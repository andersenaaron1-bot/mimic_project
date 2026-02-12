import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class AETLossModule(nn.Module):
    """
    Multi-lane loss for token ID spaces with separate output heads.

    This project uses *global token ids* (one shared id space for all categories).
    The collator emits `token_type_ids` as `TokenCategory`, but head selection for
    next-token prediction must be based on the *token id ranges* (e.g. RVQ vs
    measurement code tokens both live under TokenCategory.MEASUREMENT).

    `vocab_config` provides a flexible routing table that maps global ids -> head.
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

        # Default weights prioritize structure heavily
        self.weights = weights or {
            "struct": 5.0,
            "rvq": 1.0,
            "meas": 1.0,
            "med": 1.0,
            "val": 1.0,
            "win": 1.0,
            "len": 0.0,
            "time": 0.0,
            "dt": 0.0,
        }

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # No ignore_index needed if we mask carefully
        self.mse_loss = nn.MSELoss(reduction='none')

        self.routing = self._build_routing(self.vocab_config)

    @staticmethod
    def _build_routing(vocab_config: dict) -> dict[str, list[dict]]:
        """
        Build a routing spec:
          head_key -> list of blocks, where each block is {"offset": int, "size": int, "name": str?}

        Priority:
          1) vocab_config["routing"] if present
          2) infer from vocab_config["offsets"] + size_* keys (legacy)
        """
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

        routing: dict[str, list[dict]] = {}
        routing["logits_struct"] = [{"offset": _need("SPECIAL"), "size": size_special, "name": "SPECIAL"}]
        routing["logits_rvq"] = [{"offset": _need("RVQ"), "size": size_rvq, "name": "RVQ"}]
        routing["logits_meas"] = [{"offset": _need("MEAS"), "size": size_meas, "name": "MEAS"}]
        routing["logits_medtok"] = [{"offset": _need("MED"), "size": size_med, "name": "MED"}]
        return routing

    def forward(self, head_outputs, targets_dict):
        """
        Args:
            head_outputs: Dict from AETOutputHeads
            targets_dict: From Collator containing:
                - 'input_ids': The shifted targets (next token)
                - 'token_type_ids': The mask (SPECIAL vs RVQ vs MED)
                - 'numeric_values': The regression target
        """
        target_ids = targets_dict["input_ids"]  # (B, W, L) global token ids
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

        # --- Token classification losses (range-based routing) ---
        for head_key, blocks in self.routing.items():
            if head_key not in head_outputs:
                continue
            logits = head_outputs[head_key]
            if logits.shape[:-1] != target_ids.shape:
                raise ValueError(
                    f"{head_key} logits shape {tuple(logits.shape)} incompatible with target_ids {tuple(target_ids.shape)}"
                )

            # Build local targets for this head by stitching blocks in-order.
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

            if mask_head.any():
                loss = self.ce_loss(logits[mask_head], local_targets[mask_head])
                weight_key = head_to_weight.get(head_key, head_key)
                w = float(self.weights.get(weight_key, 1.0))
                total_loss = total_loss + (w * loss.mean())
                logs[head_to_log.get(head_key, f"loss_{head_key}")] = float(loss.mean().item())

            routed |= mask_head

        # Sanity: ensure every non-pad position is routed to exactly one head
        unrouted = valid & ~routed
        valid_count = int(valid.sum().item()) if valid.numel() else 0
        if valid_count > 0:
            logs["frac_unrouted"] = float(unrouted.sum().float().item() / float(valid_count))
        else:
            logs["frac_unrouted"] = 0.0
        if self.strict_routing and unrouted.any():
            sample = target_ids[unrouted].detach().flatten()[:8].tolist()
            raise ValueError(
                f"Unrouted token ids encountered (n={int(unrouted.sum())}); sample={sample}. "
                "Update vocab_config['routing'] (or offsets/sizes) to cover all tokens."
            )

        # --- Regression Loss (side-channel) ---
        pred_val = head_outputs.get("pred_values", None)
        target_vals = targets_dict.get("numeric_values", None)
        numeric_mask = targets_dict.get("numeric_mask", None)
        if pred_val is not None and target_vals is not None:
            if numeric_mask is not None:
                mask_val = numeric_mask.to(dtype=torch.bool) & valid
            else:
                # Fallback: treat "value == 0" as missing (legacy behavior)
                mask_val = (target_vals != 0).squeeze(-1) & valid
            if mask_val.any():
                loss = self.mse_loss(pred_val.squeeze(-1)[mask_val], target_vals.squeeze(-1)[mask_val])
                total_loss = total_loss + float(self.weights.get("val", 1.0)) * loss.mean()
                logs["loss_val"] = float(loss.mean().item())

        # --- Auxiliary: next-window-type prediction (global transition modeling) ---
        logits_next_window_type = head_outputs.get("logits_next_window_type", None)
        window_type_ids = targets_dict.get("window_type_ids", None)
        window_mask = targets_dict.get("window_mask", None)
        if logits_next_window_type is not None and window_type_ids is not None:
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

        # --- Auxiliary: window length prediction (token count + elapsed hours) ---
        pred_len_tokens = head_outputs.get("pred_window_len_tokens", None)
        pred_len_hours = head_outputs.get("pred_window_len_hours", None)
        if pred_len_tokens is not None and pred_len_hours is not None:
            # Require time_ids to compute duration and token_type_ids to exclude SPECIAL markers if present.
            time_ids = targets_dict.get("time_ids", None)
            attention_mask = targets_dict.get("attention_mask", None)
            token_type_ids = targets_dict.get("token_type_ids", None)
            if time_ids is not None and attention_mask is not None:
                if pred_len_tokens.shape != pred_len_hours.shape:
                    raise ValueError(
                        "pred_window_len_tokens and pred_window_len_hours must have the same shape; "
                        f"got {tuple(pred_len_tokens.shape)} vs {tuple(pred_len_hours.shape)}"
                    )
                if time_ids.ndim != 3:
                    raise ValueError(f"time_ids must be (B,W,L), got shape {tuple(time_ids.shape)}")
                if attention_mask.shape != time_ids.shape:
                    raise ValueError(
                        f"attention_mask must match time_ids shape; got {tuple(attention_mask.shape)} vs {tuple(time_ids.shape)}"
                    )
                B, W, L = time_ids.shape
                if pred_len_tokens.shape != (B, W):
                    raise ValueError(f"pred_window_len_* must be (B,W), got {tuple(pred_len_tokens.shape)}")

                win_mask = window_mask.to(dtype=torch.bool) if window_mask is not None else torch.ones((B, W), device=time_ids.device, dtype=torch.bool)

                content_mask = attention_mask.to(dtype=torch.bool)
                # By convention, TokenCategory.SPECIAL == 0.
                if token_type_ids is not None:
                    if token_type_ids.shape != (B, W, L):
                        raise ValueError(f"token_type_ids must be (B,W,L), got shape {tuple(token_type_ids.shape)}")
                    content_mask = content_mask & (token_type_ids != 0)

                true_len_tokens = content_mask.to(dtype=torch.float32).sum(dim=2)  # (B,W)

                # Duration as max relative time among content tokens.
                neg_inf = torch.tensor(float("-inf"), device=time_ids.device, dtype=time_ids.dtype)
                t_masked = torch.where(content_mask, time_ids, neg_inf)
                true_len_hours = t_masked.max(dim=2).values  # (B,W)
                true_len_hours = torch.where(torch.isfinite(true_len_hours), true_len_hours, torch.zeros_like(true_len_hours))
                true_len_hours = true_len_hours.clamp(min=0.0)

                # Log-scale regression is more stable across variable density.
                pred_len_tokens = pred_len_tokens.clamp(min=1.0)
                pred_len_hours = pred_len_hours.clamp(min=0.0)
                loss_tokens = F.mse_loss(torch.log1p(pred_len_tokens), torch.log1p(true_len_tokens), reduction="none")

                max_hours = 28.0 * 24.0
                denom = math.log1p(max_hours)
                pred_h = torch.log1p(pred_len_hours.clamp(max=max_hours)) / denom
                true_h = torch.log1p(true_len_hours.clamp(max=max_hours)) / denom
                loss_hours = F.mse_loss(pred_h, true_h, reduction="none")

                mask = win_mask
                if mask.any():
                    loss_len = (loss_tokens[mask].mean() + loss_hours[mask].mean()) * 0.5
                    w = float(self.weights.get("len", 0.0))
                    total_loss = total_loss + w * loss_len
                    logs["loss_window_len"] = float(loss_len.item())

        # --- Auxiliary: window duration NLL (distribution over log1p(hours)) ---
        pred_dur_mu = head_outputs.get("pred_window_dur_mu", None)
        pred_dur_sigma = head_outputs.get("pred_window_dur_sigma", None)
        if pred_dur_mu is not None and pred_dur_sigma is not None:
            time_ids = targets_dict.get("time_ids", None)
            attention_mask = targets_dict.get("attention_mask", None)
            token_type_ids = targets_dict.get("token_type_ids", None)
            if time_ids is not None and attention_mask is not None:
                if time_ids.ndim != 3:
                    raise ValueError(f"time_ids must be (B,W,L), got shape {tuple(time_ids.shape)}")
                if attention_mask.shape != time_ids.shape:
                    raise ValueError(
                        f"attention_mask must match time_ids shape; got {tuple(attention_mask.shape)} vs {tuple(time_ids.shape)}"
                    )
                B, W, L = time_ids.shape
                if pred_dur_mu.shape != (B, W) or pred_dur_sigma.shape != (B, W):
                    raise ValueError(
                        "pred_window_dur_mu and pred_window_dur_sigma must be (B,W); "
                        f"got {tuple(pred_dur_mu.shape)} and {tuple(pred_dur_sigma.shape)} with time_ids={tuple(time_ids.shape)}"
                    )

                win_mask = window_mask.to(dtype=torch.bool) if window_mask is not None else torch.ones((B, W), device=time_ids.device, dtype=torch.bool)

                content_mask = attention_mask.to(dtype=torch.bool)
                if token_type_ids is not None:
                    if token_type_ids.shape != (B, W, L):
                        raise ValueError(f"token_type_ids must be (B,W,L), got shape {tuple(token_type_ids.shape)}")
                    # By convention, TokenCategory.SPECIAL == 0.
                    content_mask = content_mask & (token_type_ids != 0)

                # Duration as max relative time among content tokens.
                neg_inf = torch.tensor(float("-inf"), device=time_ids.device, dtype=time_ids.dtype)
                t_masked = torch.where(content_mask, time_ids, neg_inf)
                true_dur_h = t_masked.max(dim=2).values  # (B,W)
                true_dur_h = torch.where(torch.isfinite(true_dur_h), true_dur_h, torch.zeros_like(true_dur_h))
                true_dur_h = true_dur_h.clamp(min=0.0)

                # Only supervise windows that have at least one content token.
                has_content = content_mask.any(dim=2)
                mask = win_mask & has_content

                max_hours = 28.0 * 24.0
                y_true = torch.log1p(true_dur_h.clamp(max=max_hours))
                sigma = pred_dur_sigma.to(dtype=y_true.dtype).clamp(min=1e-4)
                mu = pred_dur_mu.to(dtype=y_true.dtype)
                nll = 0.5 * ((y_true - mu) / sigma).square() + torch.log(sigma)

                if mask.any():
                    loss_time = nll[mask].mean()
                    w = float(self.weights.get("time", 0.0))
                    total_loss = total_loss + w * loss_time
                    logs["loss_window_dur_nll"] = float(loss_time.item())

        # --- Auxiliary: per-event dt-to-next NLL (distribution over log1p(hours)) ---
        pred_dt_mu = head_outputs.get("pred_dt_next_mu", None)
        pred_dt_sigma = head_outputs.get("pred_dt_next_sigma", None)
        if pred_dt_mu is not None and pred_dt_sigma is not None:
            time_ids = targets_dict.get("time_ids", None)
            attention_mask = targets_dict.get("attention_mask", None)
            token_type_ids = targets_dict.get("token_type_ids", None)
            if time_ids is not None and attention_mask is not None and token_type_ids is not None:
                if time_ids.ndim != 3:
                    raise ValueError(f"time_ids must be (B,W,L), got shape {tuple(time_ids.shape)}")
                if attention_mask.shape != time_ids.shape:
                    raise ValueError(
                        f"attention_mask must match time_ids shape; got {tuple(attention_mask.shape)} vs {tuple(time_ids.shape)}"
                    )
                if token_type_ids.shape != time_ids.shape:
                    raise ValueError(
                        f"token_type_ids must match time_ids shape; got {tuple(token_type_ids.shape)} vs {tuple(time_ids.shape)}"
                    )
                if pred_dt_mu.shape != time_ids.shape or pred_dt_sigma.shape != time_ids.shape:
                    raise ValueError(
                        "pred_dt_next_mu and pred_dt_next_sigma must be (B,W,L); "
                        f"got {tuple(pred_dt_mu.shape)} and {tuple(pred_dt_sigma.shape)} with time_ids={tuple(time_ids.shape)}"
                    )

                # Only consider content tokens and their next content token.
                content_mask = attention_mask.to(dtype=torch.bool) & (token_type_ids != 0)

                B, W, L = time_ids.shape
                N = B * W
                t = time_ids.reshape(N, L)
                m = content_mask.reshape(N, L)

                next_t = torch.zeros_like(t)
                next_exists = torch.zeros((N, L), device=t.device, dtype=torch.bool)
                last_t = torch.zeros((N,), device=t.device, dtype=t.dtype)
                has = torch.zeros((N,), device=t.device, dtype=torch.bool)
                for i in range(L - 1, -1, -1):
                    next_t[:, i] = last_t
                    next_exists[:, i] = has
                    cur = m[:, i]
                    last_t = torch.where(cur, t[:, i], last_t)
                    has = has | cur

                dt_h = (next_t - t).clamp(min=0.0).reshape(B, W, L)
                next_exists = next_exists.reshape(B, W, L)
                mask = content_mask & next_exists & (dt_h > 0.0)

                if mask.any():
                    max_hours = 28.0 * 24.0
                    y_true = torch.log1p(dt_h.clamp(max=max_hours))
                    mu = pred_dt_mu.to(dtype=y_true.dtype)
                    sigma = pred_dt_sigma.to(dtype=y_true.dtype).clamp(min=1e-4)
                    nll = 0.5 * ((y_true - mu) / sigma).square() + torch.log(sigma)
                    loss_dt = nll[mask].mean()

                    w = float(self.weights.get("dt", 0.0))
                    total_loss = total_loss + w * loss_dt
                    logs["loss_dt_nll"] = float(loss_dt.item())

        return total_loss, logs
