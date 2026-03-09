import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class AETLossModule(nn.Module):
    """
    Multi-lane loss for token ID spaces with separate output heads.

    Supports both legacy 2-level `(B,W,L)` local sequences and the refactored
    3-level `(B,W,C,L)` shape where:
      - `W` is the semantic-window chain
      - `C` is the bounded local chunk axis within each semantic window
    """

    def __init__(
        self,
        *,
        vocab_config: dict,
        weights: dict | None = None,
        strict_routing: bool = True,
    ) -> None:
        super().__init__()
        self.vocab_config = dict(vocab_config)
        self.strict_routing = bool(strict_routing)
        self.weights = weights or {
            "struct": 5.0,
            "rvq": 1.0,
            "meas": 1.0,
            "med": 1.0,
            "val": 1.0,
            "win": 1.0,
            "transition": 1.0,
            "win_boundary": 1.0,
            "len": 0.0,
            "chunk": 0.0,
            "time": 0.0,
            "dt": 0.0,
        }

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.routing = self._build_routing(self.vocab_config)
        self._marker_info = self._build_marker_info(self.vocab_config)

    @staticmethod
    def _build_marker_info(vocab_config: dict) -> dict[str, int]:
        offsets_raw = vocab_config.get("offsets", {})
        offsets = {str(k).upper(): int(v) for k, v in offsets_raw.items()} if isinstance(offsets_raw, dict) else {}
        special_offset = int(offsets.get("SPECIAL", 0))

        markers = vocab_config.get("window_markers", {})
        markers = markers if isinstance(markers, dict) else {}
        type_rel = int(markers.get("type_token_offset", 0))
        num_types = int(markers.get("num_types", 0))
        end_rel = int(markers.get("end_token_id", type_rel + num_types))
        continue_rel = int(markers.get("continue_token_id", end_rel + 1))
        return {
            "special_offset": special_offset,
            "type_start": special_offset + type_rel,
            "type_end_excl": special_offset + type_rel + max(0, num_types),
            "num_types": max(0, num_types),
            "end_id": special_offset + end_rel,
            "continue_id": special_offset + continue_rel,
        }

    @staticmethod
    def _build_routing(vocab_config: dict) -> dict[str, list[dict]]:
        if "routing" in vocab_config and vocab_config["routing"] is not None:
            routing = vocab_config["routing"]
            if not isinstance(routing, dict):
                raise TypeError("vocab_config['routing'] must be a dict[head_key -> list[block]]")
            return routing

        offsets_raw = vocab_config.get("offsets", {})
        if not isinstance(offsets_raw, dict):
            raise TypeError("vocab_config['offsets'] must be a dict when routing is not provided")
        offsets = {str(k).upper(): int(v) for k, v in offsets_raw.items()}

        def _need(key: str) -> int:
            if key not in offsets:
                raise KeyError(f"Missing offsets['{key}']; provide vocab_config['routing'] or offsets['{key}'].")
            return offsets[key]

        size_special = int(vocab_config.get("size_special", 0))
        size_rvq = int(vocab_config.get("size_rvq", 0))
        size_meas = int(vocab_config.get("size_meas_labels", 0))
        size_med = int(vocab_config.get("size_meds", 0))

        return {
            "logits_struct": [{"offset": _need("SPECIAL"), "size": size_special, "name": "SPECIAL"}],
            "logits_rvq": [{"offset": _need("RVQ"), "size": size_rvq, "name": "RVQ"}],
            "logits_meas": [{"offset": _need("MEAS"), "size": size_meas, "name": "MEAS"}],
            "logits_medtok": [{"offset": _need("MED"), "size": size_med, "name": "MED"}],
        }

    @staticmethod
    def _default_window_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.ndim == 3:
            return attention_mask.any(dim=-1).to(dtype=torch.long)
        if attention_mask.ndim == 4:
            return attention_mask.any(dim=-1).any(dim=-1).to(dtype=torch.long)
        raise ValueError(f"attention_mask must be 3D or 4D, got shape {tuple(attention_mask.shape)}")

    @staticmethod
    def _default_chunk_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        if attention_mask.ndim != 4:
            raise ValueError(f"attention_mask must be 4D for chunk mask derivation, got shape {tuple(attention_mask.shape)}")
        return attention_mask.any(dim=-1).to(dtype=torch.long)

    @staticmethod
    def _content_mask(attention_mask: torch.Tensor, token_type_ids: torch.Tensor | None) -> torch.Tensor:
        mask = attention_mask.to(dtype=torch.bool)
        if token_type_ids is not None:
            mask = mask & (token_type_ids != 0)
        return mask

    @staticmethod
    def _chunk_end_positions(attention_mask: torch.Tensor) -> torch.Tensor:
        valid = attention_mask.to(dtype=torch.bool)
        if valid.ndim == 3:
            B, W, L = valid.shape
            lengths = valid.to(dtype=torch.long).sum(dim=-1)
            has_tokens = lengths > 0
            idx = (lengths - 1).clamp(min=0)
            out = torch.zeros_like(valid)
            b = torch.arange(B, device=valid.device)[:, None]
            w = torch.arange(W, device=valid.device)[None, :]
            out[b, w, idx] = has_tokens
            return out & valid
        if valid.ndim == 4:
            B, W, C, L = valid.shape
            lengths = valid.to(dtype=torch.long).sum(dim=-1)
            has_tokens = lengths > 0
            idx = (lengths - 1).clamp(min=0)
            out = torch.zeros_like(valid)
            b = torch.arange(B, device=valid.device)[:, None, None]
            w = torch.arange(W, device=valid.device)[None, :, None]
            c = torch.arange(C, device=valid.device)[None, None, :]
            out[b, w, c, idx] = has_tokens
            return out & valid
        raise ValueError(
            f"attention_mask must be 3D or 4D to derive chunk-end positions, got shape {tuple(valid.shape)}"
        )

    def _marker_masks(self, target_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        type_start = int(self._marker_info["type_start"])
        type_end_excl = int(self._marker_info["type_end_excl"])
        end_id = int(self._marker_info["end_id"])
        continue_id = int(self._marker_info["continue_id"])

        type_mask = (target_ids >= type_start) & (target_ids < type_end_excl)
        end_mask = target_ids == end_id
        continue_mask = target_ids == continue_id
        return type_mask, end_mask, continue_mask

    @classmethod
    def _window_targets(
        cls,
        targets_dict: dict,
        *,
        window_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        semantic_token_counts = targets_dict.get("semantic_token_counts", None)
        semantic_duration_hours = targets_dict.get("semantic_duration_hours", None)
        time_ids = targets_dict.get("time_ids", None)
        attention_mask = targets_dict.get("attention_mask", None)
        token_type_ids = targets_dict.get("token_type_ids", None)
        chunk_start_offsets = targets_dict.get("chunk_start_offsets", None)

        if semantic_token_counts is not None and semantic_duration_hours is not None:
            if semantic_token_counts.shape != semantic_duration_hours.shape:
                raise ValueError(
                    "semantic_token_counts and semantic_duration_hours must match; "
                    f"got {tuple(semantic_token_counts.shape)} vs {tuple(semantic_duration_hours.shape)}"
                )
            if window_mask is None:
                window_mask = torch.ones_like(semantic_token_counts, dtype=torch.long)
            return (
                window_mask.to(dtype=torch.bool),
                semantic_token_counts.to(dtype=torch.float32),
                semantic_duration_hours.to(dtype=torch.float32),
            )

        if time_ids is None or attention_mask is None:
            raise ValueError("Need either semantic_* targets or time_ids+attention_mask to derive window targets.")
        if window_mask is None:
            window_mask = cls._default_window_mask(attention_mask)

        content_mask = cls._content_mask(attention_mask, token_type_ids)
        if time_ids.ndim == 3:
            true_len_tokens = content_mask.to(dtype=torch.float32).sum(dim=2)
            neg_inf = torch.tensor(float("-inf"), device=time_ids.device, dtype=time_ids.dtype)
            t_masked = torch.where(content_mask, time_ids, neg_inf)
            true_len_hours = t_masked.max(dim=2).values
        elif time_ids.ndim == 4:
            true_len_tokens = content_mask.to(dtype=torch.float32).sum(dim=(2, 3))
            t_sem = time_ids
            if chunk_start_offsets is not None:
                t_sem = time_ids + chunk_start_offsets.unsqueeze(-1)
            neg_inf = torch.tensor(float("-inf"), device=time_ids.device, dtype=time_ids.dtype)
            t_masked = torch.where(content_mask, t_sem, neg_inf)
            true_len_hours = t_masked.view(t_masked.shape[0], t_masked.shape[1], -1).max(dim=2).values
        else:
            raise ValueError(f"time_ids must be 3D or 4D, got shape {tuple(time_ids.shape)}")

        true_len_hours = torch.where(torch.isfinite(true_len_hours), true_len_hours, torch.zeros_like(true_len_hours))
        true_len_hours = true_len_hours.clamp(min=0.0)
        return window_mask.to(dtype=torch.bool), true_len_tokens, true_len_hours

    @classmethod
    def _chunk_targets(cls, targets_dict: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_token_counts = targets_dict.get("chunk_token_counts", None)
        chunk_duration_hours = targets_dict.get("chunk_duration_hours", None)
        chunk_mask = targets_dict.get("chunk_mask", None)
        time_ids = targets_dict.get("time_ids", None)
        attention_mask = targets_dict.get("attention_mask", None)
        token_type_ids = targets_dict.get("token_type_ids", None)

        if chunk_token_counts is not None and chunk_duration_hours is not None and chunk_mask is not None:
            return (
                chunk_mask.to(dtype=torch.bool),
                chunk_token_counts.to(dtype=torch.float32),
                chunk_duration_hours.to(dtype=torch.float32),
            )

        if time_ids is None or attention_mask is None or time_ids.ndim != 4:
            raise ValueError("Need chunk_* targets or 4D time_ids+attention_mask to derive chunk targets.")

        if chunk_mask is None:
            chunk_mask = cls._default_chunk_mask(attention_mask)
        content_mask = cls._content_mask(attention_mask, token_type_ids)
        true_len_tokens = content_mask.to(dtype=torch.float32).sum(dim=3)
        neg_inf = torch.tensor(float("-inf"), device=time_ids.device, dtype=time_ids.dtype)
        t_masked = torch.where(content_mask, time_ids, neg_inf)
        true_len_hours = t_masked.max(dim=3).values
        true_len_hours = torch.where(torch.isfinite(true_len_hours), true_len_hours, torch.zeros_like(true_len_hours))
        true_len_hours = true_len_hours.clamp(min=0.0)
        return chunk_mask.to(dtype=torch.bool), true_len_tokens, true_len_hours

    @staticmethod
    def _flatten_local_sequences(
        time_ids: torch.Tensor,
        content_mask: torch.Tensor,
        *,
        chunk_start_offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if time_ids.ndim == 3:
            B, W, L = time_ids.shape
            return time_ids.reshape(B * W, L), content_mask.reshape(B * W, L)
        if time_ids.ndim != 4:
            raise ValueError(f"time_ids must be 3D or 4D, got shape {tuple(time_ids.shape)}")
        if chunk_start_offsets is not None:
            time_ids = time_ids + chunk_start_offsets.unsqueeze(-1)
        B, W, C, L = time_ids.shape
        return time_ids.reshape(B * W, C * L), content_mask.reshape(B * W, C * L)

    def forward(self, head_outputs, targets_dict):
        target_ids = targets_dict["input_ids"]
        attention_mask = targets_dict.get("attention_mask", None)
        valid = attention_mask.to(dtype=torch.bool) if attention_mask is not None else torch.ones_like(target_ids, dtype=torch.bool)

        total_loss = target_ids.new_zeros((), dtype=torch.float32)
        logs = {}

        head_to_weight = {
            "logits_struct": "struct",
            "logits_rvq": "rvq",
            "logits_meas": "meas",
            "logits_medtok": "med",
        }
        head_to_log = {
            "logits_struct": "loss_struct",
            "logits_rvq": "loss_rvq",
            "logits_meas": "loss_meas",
            "logits_medtok": "loss_med",
        }

        routed = torch.zeros_like(valid, dtype=torch.bool)
        has_transition_supervision = (
            ("logits_transition_boundary" in head_outputs)
            or ("logits_boundary_next_window_type" in head_outputs)
        )
        marker_type_mask_all, marker_end_mask_all, marker_continue_mask_all = self._marker_masks(target_ids)
        marker_any_mask_all = marker_type_mask_all | marker_end_mask_all | marker_continue_mask_all
        for head_key, blocks in self.routing.items():
            if head_key not in head_outputs:
                continue
            logits = torch.nan_to_num(
                head_outputs[head_key],
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            if logits.shape[:-1] != target_ids.shape:
                raise ValueError(
                    f"{head_key} logits shape {tuple(logits.shape)} incompatible with target_ids {tuple(target_ids.shape)}"
                )

            local_targets = torch.full_like(target_ids, fill_value=-1, dtype=torch.long)
            mask_head = torch.zeros_like(valid, dtype=torch.bool)
            head_vocab_size = 0

            if not isinstance(blocks, list):
                raise TypeError(f"routing[{head_key}] must be a list of blocks")

            for b in blocks:
                if not isinstance(b, dict):
                    raise TypeError(f"routing[{head_key}] blocks must be dicts, got {type(b)}")
                offset = int(b.get("offset"))
                size = int(b.get("size"))
                if size <= 0:
                    continue
                m = valid & (target_ids >= offset) & (target_ids < offset + size)
                if m.any():
                    local_targets[m] = (target_ids[m] - offset + head_vocab_size).to(torch.long)
                mask_head |= m
                head_vocab_size += size

            if head_vocab_size <= 0:
                continue
            if logits.shape[-1] != head_vocab_size:
                raise ValueError(
                    f"{head_key} expects vocab_size={head_vocab_size} from routing blocks, "
                    f"but logits last dim is {int(logits.shape[-1])}"
                )

            if head_key == "logits_struct" and has_transition_supervision:
                mask_head = mask_head & ~marker_any_mask_all

            if mask_head.any():
                loss = self.ce_loss(logits[mask_head], local_targets[mask_head])
                w = float(self.weights.get(head_to_weight.get(head_key, head_key), 1.0))
                total_loss = total_loss + (w * loss.mean())
                logs[head_to_log.get(head_key, f"loss_{head_key}")] = float(loss.mean().item())

            routed |= mask_head

        if has_transition_supervision:
            routed = routed | (valid & marker_any_mask_all)

        unrouted = valid & ~routed
        valid_count = int(valid.sum().item()) if valid.numel() else 0
        logs["frac_unrouted"] = float(unrouted.sum().float().item() / float(valid_count)) if valid_count > 0 else 0.0
        if self.strict_routing and unrouted.any():
            sample = target_ids[unrouted].detach().flatten()[:8].tolist()
            raise ValueError(
                f"Unrouted token ids encountered (n={int(unrouted.sum())}); sample={sample}. "
                "Update vocab_config['routing'] (or offsets/sizes) to cover all tokens."
            )

        attention_mask_local = targets_dict.get("attention_mask", None)
        if attention_mask_local is not None:
            boundary_positions = self._chunk_end_positions(attention_mask_local)
        else:
            boundary_positions = torch.zeros_like(target_ids, dtype=torch.bool)

        type_mask = marker_type_mask_all
        end_mask = marker_end_mask_all
        continue_mask = marker_continue_mask_all

        logits_transition_boundary = head_outputs.get("logits_transition_boundary", None)
        if logits_transition_boundary is not None:
            logits_transition_boundary = torch.nan_to_num(
                logits_transition_boundary, nan=0.0, posinf=0.0, neginf=0.0
            )
            if logits_transition_boundary.shape[:-1] != target_ids.shape or int(logits_transition_boundary.shape[-1]) != 2:
                raise ValueError(
                    "logits_transition_boundary must be (...,2) aligned with input_ids; "
                    f"got {tuple(logits_transition_boundary.shape)} vs {tuple(target_ids.shape)}"
                )

            supervise_continue = boundary_positions & continue_mask & valid
            supervise_end = boundary_positions & (end_mask | type_mask) & valid
            supervise_mask = supervise_continue | supervise_end

            if supervise_mask.any():
                targets_boundary = torch.zeros_like(target_ids, dtype=torch.long)
                targets_boundary[supervise_end] = 1
                loss_boundary = self.ce_loss(
                    logits_transition_boundary[supervise_mask],
                    targets_boundary[supervise_mask],
                ).mean()
                total_loss = total_loss + float(self.weights.get("transition", 1.0)) * loss_boundary
                logs["loss_transition_boundary"] = float(loss_boundary.item())
                pred_boundary = logits_transition_boundary.argmax(dim=-1)
                acc_boundary = (
                    (pred_boundary[supervise_mask] == targets_boundary[supervise_mask])
                    .to(dtype=torch.float32)
                    .mean()
                )
                logs["acc_transition_boundary"] = float(acc_boundary.item())
                logs["n_transition_boundary_supervised"] = int(supervise_mask.sum().item())

        logits_boundary_next_window_type = head_outputs.get("logits_boundary_next_window_type", None)
        if logits_boundary_next_window_type is not None and int(self._marker_info["num_types"]) > 0:
            logits_boundary_next_window_type = torch.nan_to_num(
                logits_boundary_next_window_type, nan=0.0, posinf=0.0, neginf=0.0
            )
            expected_shape = target_ids.shape + (int(self._marker_info["num_types"]),)
            if logits_boundary_next_window_type.shape != expected_shape:
                raise ValueError(
                    "logits_boundary_next_window_type must be (*input_shape, num_window_types); "
                    f"got {tuple(logits_boundary_next_window_type.shape)} expected {tuple(expected_shape)}"
                )

            type_boundary_mask = boundary_positions & type_mask & valid
            if type_boundary_mask.any():
                type_start = int(self._marker_info["type_start"])
                targets_next_type = (target_ids - type_start).to(dtype=torch.long)
                loss_next_type_boundary = self.ce_loss(
                    logits_boundary_next_window_type[type_boundary_mask],
                    targets_next_type[type_boundary_mask],
                ).mean()
                total_loss = total_loss + float(self.weights.get("win_boundary", 1.0)) * loss_next_type_boundary
                logs["loss_next_window_type_boundary"] = float(loss_next_type_boundary.item())
                pred_next_type = logits_boundary_next_window_type.argmax(dim=-1)
                acc_next_type_boundary = (
                    (pred_next_type[type_boundary_mask] == targets_next_type[type_boundary_mask])
                    .to(dtype=torch.float32)
                    .mean()
                )
                logs["acc_next_window_type_boundary"] = float(acc_next_type_boundary.item())
                logs["n_next_window_type_boundary_supervised"] = int(type_boundary_mask.sum().item())

        pred_val = head_outputs.get("pred_values", None)
        target_vals = targets_dict.get("numeric_values", None)
        numeric_mask = targets_dict.get("numeric_mask", None)
        if pred_val is not None and target_vals is not None:
            pred_val = torch.nan_to_num(pred_val, nan=0.0, posinf=0.0, neginf=0.0)
            if numeric_mask is not None:
                mask_val = numeric_mask.to(dtype=torch.bool) & valid
            else:
                mask_val = (target_vals != 0).squeeze(-1) & valid
            if mask_val.any():
                loss = self.mse_loss(pred_val.squeeze(-1)[mask_val], target_vals.squeeze(-1)[mask_val])
                total_loss = total_loss + float(self.weights.get("val", 1.0)) * loss.mean()
                logs["loss_val"] = float(loss.mean().item())

        logits_next_window_type = head_outputs.get("logits_next_window_type", None)
        window_type_ids = targets_dict.get("window_type_ids", None)
        window_mask = targets_dict.get("window_mask", None)
        if logits_next_window_type is not None and window_type_ids is not None:
            logits_next_window_type = torch.nan_to_num(
                logits_next_window_type, nan=0.0, posinf=0.0, neginf=0.0
            )
            if logits_next_window_type.ndim != 3:
                raise ValueError(
                    f"logits_next_window_type must be (B,W,K), got shape {tuple(logits_next_window_type.shape)}"
                )
            if window_type_ids.ndim != 2:
                raise ValueError(f"window_type_ids must be (B,W), got shape {tuple(window_type_ids.shape)}")
            if logits_next_window_type.shape[:2] != window_type_ids.shape:
                raise ValueError(
                    "logits_next_window_type and window_type_ids must match on (B,W); "
                    f"got {tuple(logits_next_window_type.shape[:2])} vs {tuple(window_type_ids.shape)}"
                )
            B, W = window_type_ids.shape
            if W >= 2:
                if window_mask is None:
                    window_mask = torch.ones((B, W), device=window_type_ids.device, dtype=torch.long)
                if window_mask.shape != (B, W):
                    raise ValueError(f"window_mask must be (B,W), got shape {tuple(window_mask.shape)}")
                transition_mask = window_mask[:, :-1].to(dtype=torch.bool) & window_mask[:, 1:].to(dtype=torch.bool)
                if transition_mask.any():
                    K = int(logits_next_window_type.shape[-1])
                    logits_flat = logits_next_window_type[:, :-1, :].reshape(B * (W - 1), K)
                    targets_flat = window_type_ids[:, 1:].reshape(B * (W - 1)).to(dtype=torch.long)
                    loss_all = self.ce_loss(logits_flat, targets_flat).view(B, W - 1)
                    loss_win = loss_all[transition_mask].mean()
                    total_loss = total_loss + float(self.weights.get("win", 1.0)) * loss_win
                    logs["loss_next_window_type"] = float(loss_win.item())
                    pred = logits_next_window_type[:, :-1, :].argmax(dim=-1)
                    acc = (pred == window_type_ids[:, 1:]).to(dtype=torch.float)[transition_mask].mean()
                    logs["acc_next_window_type"] = float(acc.item())

        pred_len_tokens = head_outputs.get("pred_window_len_tokens", None)
        pred_len_hours = head_outputs.get("pred_window_len_hours", None)
        if pred_len_tokens is not None and pred_len_hours is not None:
            pred_len_tokens = torch.nan_to_num(pred_len_tokens, nan=1.0, posinf=1e6, neginf=1.0)
            pred_len_hours = torch.nan_to_num(pred_len_hours, nan=0.0, posinf=1e6, neginf=0.0)
            win_mask, true_len_tokens, true_len_hours = self._window_targets(targets_dict, window_mask=window_mask)
            if pred_len_tokens.shape != true_len_tokens.shape or pred_len_hours.shape != true_len_hours.shape:
                raise ValueError(
                    "pred_window_len_* must match semantic window target shapes; "
                    f"got {tuple(pred_len_tokens.shape)} / {tuple(pred_len_hours.shape)} vs "
                    f"{tuple(true_len_tokens.shape)} / {tuple(true_len_hours.shape)}"
                )

            pred_len_tokens = pred_len_tokens.clamp(min=1.0)
            pred_len_hours = pred_len_hours.clamp(min=0.0)
            loss_tokens = F.mse_loss(torch.log1p(pred_len_tokens), torch.log1p(true_len_tokens), reduction="none")
            max_hours = 28.0 * 24.0
            denom = math.log1p(max_hours)
            pred_h = torch.log1p(pred_len_hours.clamp(max=max_hours)) / denom
            true_h = torch.log1p(true_len_hours.clamp(max=max_hours)) / denom
            loss_hours = F.mse_loss(pred_h, true_h, reduction="none")

            if win_mask.any():
                loss_len = (loss_tokens[win_mask].mean() + loss_hours[win_mask].mean()) * 0.5
                total_loss = total_loss + float(self.weights.get("len", 0.0)) * loss_len
                logs["loss_window_len"] = float(loss_len.item())

        pred_dur_mu = head_outputs.get("pred_window_dur_mu", None)
        pred_dur_sigma = head_outputs.get("pred_window_dur_sigma", None)
        if pred_dur_mu is not None and pred_dur_sigma is not None:
            pred_dur_mu = torch.nan_to_num(pred_dur_mu, nan=0.0, posinf=0.0, neginf=0.0)
            pred_dur_sigma = torch.nan_to_num(pred_dur_sigma, nan=1.0, posinf=1e6, neginf=1.0)
            win_mask, _, true_dur_h = self._window_targets(targets_dict, window_mask=window_mask)
            if pred_dur_mu.shape != true_dur_h.shape or pred_dur_sigma.shape != true_dur_h.shape:
                raise ValueError(
                    "pred_window_dur_* must match semantic window duration target shapes; "
                    f"got {tuple(pred_dur_mu.shape)} / {tuple(pred_dur_sigma.shape)} vs {tuple(true_dur_h.shape)}"
                )

            y_true = torch.log1p(true_dur_h.clamp(max=28.0 * 24.0))
            sigma = pred_dur_sigma.to(dtype=y_true.dtype).clamp(min=1e-4)
            mu = pred_dur_mu.to(dtype=y_true.dtype)
            nll = 0.5 * ((y_true - mu) / sigma).square() + torch.log(sigma)

            if win_mask.any():
                loss_time = nll[win_mask].mean()
                total_loss = total_loss + float(self.weights.get("time", 0.0)) * loss_time
                logs["loss_window_dur_nll"] = float(loss_time.item())

        pred_chunk_len_tokens = head_outputs.get("pred_chunk_len_tokens", None)
        pred_chunk_len_hours = head_outputs.get("pred_chunk_len_hours", None)
        if pred_chunk_len_tokens is not None and pred_chunk_len_hours is not None:
            pred_chunk_len_tokens = torch.nan_to_num(pred_chunk_len_tokens, nan=1.0, posinf=1e6, neginf=1.0)
            pred_chunk_len_hours = torch.nan_to_num(pred_chunk_len_hours, nan=0.0, posinf=1e6, neginf=0.0)
            chunk_mask, true_chunk_tokens, true_chunk_hours = self._chunk_targets(targets_dict)
            if pred_chunk_len_tokens.shape != true_chunk_tokens.shape or pred_chunk_len_hours.shape != true_chunk_hours.shape:
                raise ValueError(
                    "pred_chunk_len_* must match chunk target shapes; "
                    f"got {tuple(pred_chunk_len_tokens.shape)} / {tuple(pred_chunk_len_hours.shape)} vs "
                    f"{tuple(true_chunk_tokens.shape)} / {tuple(true_chunk_hours.shape)}"
                )
            loss_tokens = F.mse_loss(
                torch.log1p(pred_chunk_len_tokens.clamp(min=1.0)),
                torch.log1p(true_chunk_tokens),
                reduction="none",
            )
            max_hours = 28.0 * 24.0
            denom = math.log1p(max_hours)
            pred_h = torch.log1p(pred_chunk_len_hours.clamp(min=0.0, max=max_hours)) / denom
            true_h = torch.log1p(true_chunk_hours.clamp(max=max_hours)) / denom
            loss_hours = F.mse_loss(pred_h, true_h, reduction="none")
            if chunk_mask.any():
                loss_chunk = (loss_tokens[chunk_mask].mean() + loss_hours[chunk_mask].mean()) * 0.5
                total_loss = total_loss + float(self.weights.get("chunk", 0.0)) * loss_chunk
                logs["loss_chunk_len"] = float(loss_chunk.item())

        pred_dt_mu = head_outputs.get("pred_dt_next_mu", None)
        pred_dt_sigma = head_outputs.get("pred_dt_next_sigma", None)
        if pred_dt_mu is not None and pred_dt_sigma is not None:
            pred_dt_mu = torch.nan_to_num(pred_dt_mu, nan=0.0, posinf=0.0, neginf=0.0)
            pred_dt_sigma = torch.nan_to_num(pred_dt_sigma, nan=1.0, posinf=1e6, neginf=1.0)
            time_ids = targets_dict.get("time_ids", None)
            attention_mask = targets_dict.get("attention_mask", None)
            token_type_ids = targets_dict.get("token_type_ids", None)
            chunk_start_offsets = targets_dict.get("chunk_start_offsets", None)
            if time_ids is not None and attention_mask is not None and token_type_ids is not None:
                if pred_dt_mu.shape != time_ids.shape or pred_dt_sigma.shape != time_ids.shape:
                    raise ValueError(
                        "pred_dt_next_* must match time_ids shape; "
                        f"got {tuple(pred_dt_mu.shape)} / {tuple(pred_dt_sigma.shape)} vs {tuple(time_ids.shape)}"
                    )
                content_mask = attention_mask.to(dtype=torch.bool) & (token_type_ids != 0)
                t_flat, m_flat = self._flatten_local_sequences(time_ids, content_mask, chunk_start_offsets=chunk_start_offsets)
                B, W = t_flat.shape[0], 1  # placeholder for reshape only
                N, S = t_flat.shape

                next_t = torch.zeros_like(t_flat)
                next_exists = torch.zeros((N, S), device=t_flat.device, dtype=torch.bool)
                last_t = torch.zeros((N,), device=t_flat.device, dtype=t_flat.dtype)
                has = torch.zeros((N,), device=t_flat.device, dtype=torch.bool)
                for i in range(S - 1, -1, -1):
                    next_t[:, i] = last_t
                    next_exists[:, i] = has
                    cur = m_flat[:, i]
                    last_t = torch.where(cur, t_flat[:, i], last_t)
                    has = has | cur

                dt_h_flat = (next_t - t_flat).clamp(min=0.0)
                mask_flat = m_flat & next_exists & (dt_h_flat > 0.0)
                if time_ids.ndim == 3:
                    dt_h = dt_h_flat.view_as(time_ids)
                    mask = mask_flat.view_as(time_ids)
                else:
                    dt_h = dt_h_flat.view_as(time_ids)
                    mask = mask_flat.view_as(time_ids)

                if mask.any():
                    y_true = torch.log1p(dt_h.clamp(max=28.0 * 24.0))
                    mu = pred_dt_mu.to(dtype=y_true.dtype)
                    sigma = pred_dt_sigma.to(dtype=y_true.dtype).clamp(min=1e-4)
                    nll = 0.5 * ((y_true - mu) / sigma).square() + torch.log(sigma)
                    loss_dt = nll[mask].mean()
                    total_loss = total_loss + float(self.weights.get("dt", 0.0)) * loss_dt
                    logs["loss_dt_nll"] = float(loss_dt.item())

        if not torch.isfinite(total_loss):
            logs["non_finite_total_loss"] = 1.0
            raise FloatingPointError("AETLossModule produced non-finite total_loss.")

        for key, value in list(logs.items()):
            if isinstance(value, float) and not math.isfinite(value):
                logs[key] = 0.0
                logs["non_finite_log_detected"] = 1.0

        return total_loss, logs
