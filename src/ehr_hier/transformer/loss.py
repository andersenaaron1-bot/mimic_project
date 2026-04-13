import math
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ehr_hier.data.event_frames import EVENT_PAYLOAD_KIND_TO_ID
from src.ehr_hier.transformer.precedent_memory import build_window_support_flags
from src.ehr_hier.transformer.world_model_contract import NUM_SUPPORT_FLAGS


class AETLossModule(nn.Module):
    """
    Multi-lane loss for token ID spaces with separate output heads.

    Supports both legacy 2-level `(B,W,L)` local sequences and the refactored
    3-level `(B,W,C,L)` shape where:
      - `W` is the semantic-window chain
      - `C` is the bounded local chunk axis within each semantic window

    Preferred v1 behavior:
      - use a unified dense token head (`logits_token`)
      - train with shifted autoregressive next-token targets inside each local chunk
      - keep transition/window heads as auxiliary supervision
    """

    TOKEN_FAMILY_GROUPS: Tuple[str, ...] = (
        "diagnosis",
        "diagnosis_residual",
        "procedure",
        "procedure_residual",
        "medication",
        "medication_residual",
        "measurement_code",
        "measurement_value",
        "observation_code",
        "observation_value",
        "structural",
        "special_marker",
        "unk",
    )
    EVENT_CONCEPT_FAMILY_ORDER: Tuple[str, ...] = (
        "special",
        "measurement",
        "diagnosis",
        "procedure",
        "medication",
        "structural",
    )

    def __init__(
        self,
        *,
        vocab_config: dict,
        weights: dict | None = None,
        token_family_weights: dict[str, float] | None = None,
        strict_routing: bool = True,
        prefer_unified_token_loss: bool = True,
        ignore_nonmarker_special_targets: bool = True,
        special_type_id: int = 0,
    ) -> None:
        super().__init__()
        self.vocab_config = dict(vocab_config)
        self.strict_routing = bool(strict_routing)
        self.prefer_unified_token_loss = bool(prefer_unified_token_loss)
        self.ignore_nonmarker_special_targets = bool(ignore_nonmarker_special_targets)
        self.special_type_id = int(special_type_id)
        self.weights = weights or {
            "token": 1.0,
            "event_token": 1.0,
            "event_family": 1.0,
            "event_payload": 1.0,
            "event_concept": 1.0,
            "event_dt": 1.0,
            "event_value": 1.0,
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
            "next_window_gap": 1.0,
            "next_window_duration": 1.0,
            "next_window_support": 0.5,
        }

        self.ce_loss = nn.CrossEntropyLoss(reduction="none")
        self.mse_loss = nn.MSELoss(reduction="none")
        self.routing = self._build_routing(self.vocab_config)
        self.event_concept_routing = self._build_event_concept_routing(self.vocab_config)
        self._marker_info = self._build_marker_info(self.vocab_config)
        token_family_names, token_family_ids = self._build_token_family_group_ids(self.vocab_config)
        self._token_family_group_names = token_family_names
        self.register_buffer("_token_family_group_ids", token_family_ids, persistent=False)
        token_family_loss_weights = self._build_token_family_loss_weights(
            token_family_names,
            token_family_weights,
        )
        self.register_buffer("_token_family_loss_weights", token_family_loss_weights, persistent=False)

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

    @classmethod
    def _build_event_concept_routing(cls, vocab_config: dict) -> dict[str, list[dict]]:
        dense_blocks = vocab_config.get("dense_blocks", [])
        event_routing = {
            family_name: [] for family_name in cls.EVENT_CONCEPT_FAMILY_ORDER
        }
        if not isinstance(dense_blocks, list):
            return event_routing

        block_to_family = {
            "special": "special",
            "measurement_code": "measurement",
            "observation_code": "measurement",
            "diagnosis": "diagnosis",
            "diagnosis_residual": "diagnosis",
            "procedure": "procedure",
            "procedure_residual": "procedure",
            "medication": "medication",
            "medication_residual": "medication",
            "structural": "structural",
        }
        for block in dense_blocks:
            if not isinstance(block, dict):
                continue
            block_name = str(block.get("name", ""))
            family_name = block_to_family.get(block_name, None)
            if family_name is None:
                continue
            size = int(block.get("dense_size", 0))
            if size <= 0:
                continue
            event_routing[family_name].append(
                {
                    "offset": int(block.get("dense_offset", 0)),
                    "size": size,
                    "name": block_name,
                }
            )
        return event_routing

    @classmethod
    def _build_token_family_group_ids(cls, vocab_config: dict) -> tuple[Tuple[str, ...], torch.Tensor]:
        group_names = tuple(cls.TOKEN_FAMILY_GROUPS)
        total_size = int(vocab_config.get("total_size", 0))
        if total_size <= 0:
            return group_names, torch.empty((0,), dtype=torch.long)

        group_to_idx = {name: idx for idx, name in enumerate(group_names)}
        dense_group_ids = torch.full((total_size,), fill_value=-1, dtype=torch.long)
        unk_dense_id = int(vocab_config.get("unk_dense_id", 0))
        if 0 <= unk_dense_id < total_size:
            dense_group_ids[unk_dense_id] = int(group_to_idx["unk"])

        dense_blocks = vocab_config.get("dense_blocks", [])
        if not isinstance(dense_blocks, list):
            return group_names, dense_group_ids

        sparse_contract = vocab_config.get("sparse_vocab_contract", {})
        families = sparse_contract.get("families", {}) if isinstance(sparse_contract, dict) else {}
        family_offsets: Dict[str, int] = {}
        if isinstance(families, dict):
            for name, payload in families.items():
                if isinstance(payload, dict) and "offset" in payload:
                    family_offsets[str(name)] = int(payload["offset"])

        block_to_group = {
            "diagnosis": "diagnosis",
            "diagnosis_residual": "diagnosis_residual",
            "procedure": "procedure",
            "procedure_residual": "procedure_residual",
            "medication": "medication",
            "medication_residual": "medication_residual",
            "measurement_code": "measurement_code",
            "measurement_value": "measurement_value",
            "observation_code": "observation_code",
            "observation_value": "observation_value",
            "structural": "structural",
        }

        type_start = int(vocab_config.get("offsets", {}).get("SPECIAL", 0)) + int(
            vocab_config.get("window_markers", {}).get("type_token_offset", 0)
        )
        num_types = int(vocab_config.get("window_markers", {}).get("num_types", 0))
        type_end_excl = type_start + max(0, num_types)
        end_id = int(vocab_config.get("offsets", {}).get("SPECIAL", 0)) + int(
            vocab_config.get("window_markers", {}).get("end_token_id", type_end_excl)
        )
        continue_id = int(vocab_config.get("offsets", {}).get("SPECIAL", 0)) + int(
            vocab_config.get("window_markers", {}).get("continue_token_id", end_id + 1)
        )

        def _is_marker(dense_id: int) -> bool:
            return (type_start <= dense_id < type_end_excl) or dense_id in {end_id, continue_id}

        for block in dense_blocks:
            if not isinstance(block, dict):
                continue
            block_name = str(block.get("name", ""))
            dense_offset = int(block.get("dense_offset", 0))
            dense_size = int(block.get("dense_size", 0))
            global_offset = int(block.get("global_offset", 0))
            sparse_ids = block.get("sparse_global_ids", None)
            family_offset = family_offsets.get(block_name, global_offset)
            block_group = block_to_group.get(block_name, None)

            for local_idx in range(max(0, dense_size)):
                dense_id = dense_offset + local_idx
                if dense_id < 0 or dense_id >= total_size:
                    continue
                sparse_global_id = (
                    int(sparse_ids[local_idx])
                    if isinstance(sparse_ids, list) and local_idx < len(sparse_ids)
                    else int(global_offset) + int(local_idx)
                )

                if block_name == "special":
                    if _is_marker(dense_id):
                        dense_group_ids[dense_id] = int(group_to_idx["special_marker"])
                    continue

                if int(sparse_global_id) == int(family_offset):
                    dense_group_ids[dense_id] = int(group_to_idx["unk"])
                    continue
                if block_group is not None:
                    dense_group_ids[dense_id] = int(group_to_idx[block_group])

        return group_names, dense_group_ids

    @classmethod
    def _build_token_family_loss_weights(
        cls,
        group_names: Tuple[str, ...],
        token_family_weights: dict[str, float] | None,
    ) -> torch.Tensor:
        weights = torch.ones((len(group_names),), dtype=torch.float32)
        if not token_family_weights:
            return weights

        group_to_idx = {name: idx for idx, name in enumerate(group_names)}
        for name, value in token_family_weights.items():
            if name not in group_to_idx:
                raise KeyError(
                    f"Unknown token family weight {name!r}; expected one of {sorted(group_to_idx)}"
                )
            weight_val = float(value)
            if not math.isfinite(weight_val) or weight_val <= 0.0:
                raise ValueError(f"Token family weight for {name!r} must be finite and > 0, got {value!r}")
            weights[int(group_to_idx[name])] = weight_val
        return weights

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

    def _predictive_marker_masks(
        self,
        *,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        if target_ids.ndim not in {3, 4}:
            raise ValueError(
                f"target_ids must be 3D or 4D for predictive marker masks, got shape {tuple(target_ids.shape)}"
            )
        if target_ids.shape[-1] < 2:
            empty_shape = target_ids.shape[:-1] + (0,)
            empty = torch.zeros(empty_shape, device=target_ids.device, dtype=torch.bool)
            return {
                "source_valid": empty,
                "next_type": empty,
                "next_end": empty,
                "next_continue": empty,
                "predict_continue": empty,
                "predict_end": empty,
            }

        next_targets = target_ids[..., 1:]
        next_type_mask, next_end_mask, next_continue_mask = self._marker_masks(next_targets)
        source_valid = (
            attention_mask[..., :-1].to(dtype=torch.bool) & attention_mask[..., 1:].to(dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(next_targets, dtype=torch.bool)
        )
        if token_type_ids is not None:
            source_non_special = token_type_ids[..., :-1] != int(self.special_type_id)
        else:
            source_non_special = torch.ones_like(source_valid)

        predict_continue = source_valid & source_non_special & next_continue_mask
        predict_end = source_valid & source_non_special & (next_end_mask | next_type_mask)
        return {
            "source_valid": source_valid,
            "next_type": next_type_mask,
            "next_end": next_end_mask,
            "next_continue": next_continue_mask,
            "predict_continue": predict_continue,
            "predict_end": predict_end,
        }

    @classmethod
    def _broadcast_next_window_targets(
        cls,
        *,
        window_type_ids: torch.Tensor,
        window_mask: torch.Tensor | None,
        target_shape: torch.Size,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if window_type_ids.ndim != 2:
            raise ValueError(f"window_type_ids must be (B,W), got shape {tuple(window_type_ids.shape)}")
        B, W = window_type_ids.shape
        if target_shape[:2] != (B, W):
            raise ValueError(
                "target_shape must align with window_type_ids on (B,W); "
                f"got {tuple(target_shape[:2])} vs {(B, W)}"
            )
        if window_mask is None:
            window_mask = torch.ones_like(window_type_ids, dtype=torch.long)
        if window_mask.shape != (B, W):
            raise ValueError(f"window_mask must be (B,W), got shape {tuple(window_mask.shape)}")

        next_type_ids = torch.zeros_like(window_type_ids)
        has_next_window = torch.zeros_like(window_mask, dtype=torch.bool)
        if W >= 2:
            next_type_ids[:, :-1] = window_type_ids[:, 1:]
            has_next_window[:, :-1] = (
                window_mask[:, :-1].to(dtype=torch.bool) & window_mask[:, 1:].to(dtype=torch.bool)
            )

        if len(target_shape) == 3:
            next_type_ids = next_type_ids.unsqueeze(-1).expand(target_shape)
            has_next_window = has_next_window.unsqueeze(-1).expand(target_shape)
        elif len(target_shape) == 4:
            next_type_ids = next_type_ids.unsqueeze(-1).unsqueeze(-1).expand(target_shape)
            has_next_window = has_next_window.unsqueeze(-1).unsqueeze(-1).expand(target_shape)
        else:
            raise ValueError(f"target_shape must be 3D or 4D, got {tuple(target_shape)}")
        return next_type_ids.to(dtype=torch.long), has_next_window

    def _autoregressive_targets(
        self,
        *,
        target_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        marker_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, int]]:
        if target_ids.ndim not in {3, 4}:
            raise ValueError(
                f"target_ids must be 3D or 4D for autoregressive loss, got shape {tuple(target_ids.shape)}"
            )
        if target_ids.shape[-1] < 2:
            empty_shape = target_ids.shape[:-1] + (0,)
            empty_mask = torch.zeros(empty_shape, device=target_ids.device, dtype=torch.bool)
            empty_targets = torch.zeros(empty_shape, device=target_ids.device, dtype=torch.long)
            return empty_targets, empty_mask, empty_mask, {
                "candidate_targets": 0,
                "ignored_nonmarker_special_targets": 0,
            }

        ar_targets = target_ids[..., 1:].to(dtype=torch.long)
        ar_valid = (
            attention_mask[..., 1:].to(dtype=torch.bool)
            if attention_mask is not None
            else torch.ones_like(ar_targets, dtype=torch.bool)
        )
        marker_source = marker_ids if marker_ids is not None else target_ids
        if marker_source.shape != target_ids.shape:
            raise ValueError(
                "marker_ids must match target_ids when provided; "
                f"got {tuple(marker_source.shape)} vs {tuple(target_ids.shape)}"
            )
        marker_targets = marker_source[..., 1:].to(dtype=torch.long)
        type_mask, end_mask, continue_mask = self._marker_masks(marker_targets)
        marker_mask = type_mask | end_mask | continue_mask

        ignored_special = torch.zeros_like(ar_valid)
        candidate_nonmarker_special = torch.zeros_like(ar_valid)
        if (
            token_type_ids is not None
            and self.ignore_nonmarker_special_targets
        ):
            target_token_types = token_type_ids[..., 1:]
            candidate_nonmarker_special = (
                (target_token_types == int(self.special_type_id))
                & ~marker_mask
                & ar_valid
            )
            ignored_special = candidate_nonmarker_special
            ar_valid = ar_valid & ~ignored_special

        stats = {
            "candidate_targets": int((attention_mask[..., 1:].to(dtype=torch.bool).sum().item()) if attention_mask is not None else ar_targets.numel()),
            "candidate_nonmarker_special_targets": int(candidate_nonmarker_special.sum().item()),
            "ignored_nonmarker_special_targets": int(ignored_special.sum().item()),
        }
        return ar_targets, ar_valid, marker_mask, stats

    def _target_groups_for_valid_targets(self, valid_targets: torch.Tensor) -> torch.Tensor:
        group_ids = self._token_family_group_ids
        target_groups = torch.full_like(valid_targets, fill_value=-1, dtype=torch.long)
        if group_ids.numel() == 0 or valid_targets.numel() == 0:
            return target_groups
        in_bounds = (valid_targets >= 0) & (valid_targets < int(group_ids.shape[0]))
        if in_bounds.any():
            target_groups[in_bounds] = group_ids[valid_targets[in_bounds]]
        return target_groups

    def _log_token_family_metrics(
        self,
        *,
        valid_logits: torch.Tensor,
        valid_targets: torch.Tensor,
        valid_preds: torch.Tensor,
        logs: Dict[str, float],
        prefix: str = "token",
    ) -> None:
        group_names = self._token_family_group_names
        for group_name in group_names:
            logs[f"n_{prefix}_family_{group_name}"] = 0

        if valid_targets.numel() == 0:
            return

        target_groups = self._target_groups_for_valid_targets(valid_targets)

        for group_idx, group_name in enumerate(group_names):
            group_mask = target_groups == int(group_idx)
            count = int(group_mask.sum().item())
            logs[f"n_{prefix}_family_{group_name}"] = count
            if count <= 0:
                continue
            group_loss = self.ce_loss(valid_logits[group_mask], valid_targets[group_mask]).mean()
            group_acc = (valid_preds[group_mask] == valid_targets[group_mask]).to(dtype=torch.float32).mean()
            logs[f"loss_{prefix}_family_{group_name}"] = float(group_loss.item())
            logs[f"acc_{prefix}_family_{group_name}"] = float(group_acc.item())

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

    @staticmethod
    def _flatten_sequence_feature(feature: torch.Tensor) -> torch.Tensor:
        if feature.ndim == 3:
            B, W, L = feature.shape
            return feature.reshape(B * W, L)
        if feature.ndim == 4:
            B, W, C, L = feature.shape
            return feature.reshape(B * W, C * L)
        raise ValueError(f"feature must be 3D or 4D, got shape {tuple(feature.shape)}")

    def _compute_autoregressive_ce_lane(
        self,
        *,
        logits: torch.Tensor | None,
        target_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        marker_ids: torch.Tensor | None,
        weight_key: str,
        log_prefix: str,
        apply_token_family_weights: bool = False,
        emit_token_family_metrics: bool = False,
    ) -> tuple[torch.Tensor, Dict[str, float], Dict[str, int]]:
        lane_logs: Dict[str, float] = {}
        if target_ids is not None:
            zero = target_ids.new_zeros((), dtype=torch.float32)
        elif logits is not None:
            zero = logits.new_zeros((), dtype=torch.float32)
        else:
            zero = torch.zeros((), dtype=torch.float32)
        empty_stats = {
            "candidate_targets": 0,
            "candidate_nonmarker_special_targets": 0,
            "ignored_nonmarker_special_targets": 0,
        }

        if (
            logits is None
            or target_ids is None
            or attention_mask is None
            or float(self.weights.get(weight_key, 0.0)) <= 0.0
        ):
            lane_logs[f"n_{log_prefix}_supervised"] = 0
            if emit_token_family_metrics and logits is not None:
                self._log_token_family_metrics(
                    valid_logits=logits.new_zeros((0, logits.shape[-1])),
                    valid_targets=logits.new_zeros((0,), dtype=torch.long),
                    valid_preds=logits.new_zeros((0,), dtype=torch.long),
                    logs=lane_logs,
                    prefix=log_prefix,
                )
            return zero, lane_logs, empty_stats

        logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
        if logits.shape[:-1] != target_ids.shape:
            raise ValueError(
                f"{log_prefix} logits must align with targets on all non-vocab dims; "
                f"got {tuple(logits.shape)} vs {tuple(target_ids.shape)}"
            )

        ar_targets, ar_valid, _, ar_stats = self._autoregressive_targets(
            target_ids=target_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            marker_ids=marker_ids,
        )
        ar_logits = logits[..., :-1, :]
        if ar_logits.shape[:-1] != ar_targets.shape:
            raise ValueError(
                f"{log_prefix} autoregressive logits/targets mismatch after shift; "
                f"got {tuple(ar_logits.shape[:-1])} vs {tuple(ar_targets.shape)}"
            )

        if ar_valid.any():
            pred_next = ar_logits.argmax(dim=-1)
            valid_logits = ar_logits[ar_valid]
            valid_targets = ar_targets[ar_valid]
            valid_preds = pred_next[ar_valid]
            lane_losses = self.ce_loss(valid_logits, valid_targets)
            loss_unweighted = lane_losses.mean()

            if apply_token_family_weights:
                target_groups = self._target_groups_for_valid_targets(valid_targets)
                loss_weights = torch.ones_like(lane_losses)
                if self._token_family_loss_weights.numel() > 0:
                    in_groups = target_groups >= 0
                    if in_groups.any():
                        loss_weights[in_groups] = self._token_family_loss_weights[
                            target_groups[in_groups]
                        ].to(dtype=lane_losses.dtype, device=lane_losses.device)
                loss_weighted = (lane_losses * loss_weights).sum() / loss_weights.sum().clamp(min=1.0)
                if not torch.allclose(loss_weights, torch.ones_like(loss_weights)):
                    lane_logs[f"loss_{log_prefix}_weighted"] = float(loss_weighted.item())
            else:
                loss_weighted = loss_unweighted

            total_contrib = float(self.weights.get(weight_key, 1.0)) * loss_weighted
            lane_logs[f"loss_{log_prefix}"] = float(loss_unweighted.item())
            lane_logs[f"acc_{log_prefix}"] = float(
                (valid_preds == valid_targets).to(dtype=torch.float32).mean().item()
            )
            lane_logs[f"n_{log_prefix}_supervised"] = int(ar_valid.sum().item())
            if emit_token_family_metrics:
                self._log_token_family_metrics(
                    valid_logits=valid_logits,
                    valid_targets=valid_targets,
                    valid_preds=valid_preds,
                    logs=lane_logs,
                    prefix=log_prefix,
                )
            return total_contrib, lane_logs, ar_stats

        lane_logs[f"n_{log_prefix}_supervised"] = 0
        if emit_token_family_metrics:
            self._log_token_family_metrics(
                valid_logits=ar_logits.new_zeros((0, ar_logits.shape[-1])),
                valid_targets=ar_targets.new_zeros((0,), dtype=torch.long),
                valid_preds=ar_targets.new_zeros((0,), dtype=torch.long),
                logs=lane_logs,
                prefix=log_prefix,
            )
        return zero, lane_logs, ar_stats

    def _compute_autoregressive_routed_ce_lane(
        self,
        *,
        head_outputs: dict,
        target_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        marker_ids: torch.Tensor | None,
        routing: dict[str, list[dict]],
        head_key_prefix: str,
        weight_key: str,
        log_prefix: str,
    ) -> tuple[torch.Tensor, Dict[str, float], Dict[str, int]]:
        lane_logs: Dict[str, float] = {}
        if target_ids is not None:
            zero = target_ids.new_zeros((), dtype=torch.float32)
        else:
            zero = torch.zeros((), dtype=torch.float32)
        empty_stats = {
            "candidate_targets": 0,
            "candidate_nonmarker_special_targets": 0,
            "ignored_nonmarker_special_targets": 0,
        }
        if (
            target_ids is None
            or attention_mask is None
            or token_type_ids is None
            or float(self.weights.get(weight_key, 0.0)) <= 0.0
        ):
            lane_logs[f"n_{log_prefix}_supervised"] = 0
            lane_logs[f"frac_{log_prefix}_unrouted"] = 0.0
            return zero, lane_logs, empty_stats

        ar_targets, ar_valid, _, ar_stats = self._autoregressive_targets(
            target_ids=target_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            marker_ids=marker_ids,
        )
        if ar_targets.numel() == 0:
            lane_logs[f"n_{log_prefix}_supervised"] = 0
            lane_logs[f"frac_{log_prefix}_unrouted"] = 0.0
            return zero, lane_logs, ar_stats

        routed = torch.zeros_like(ar_valid, dtype=torch.bool)
        weighted_loss_sum = zero.clone()
        total_count = 0
        for family_name, blocks in routing.items():
            if not isinstance(blocks, list) or not blocks:
                continue
            head_key = f"{head_key_prefix}{family_name}"
            logits = head_outputs.get(head_key, None)
            if logits is None:
                continue
            logits = torch.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
            if logits.shape[:-1] != target_ids.shape:
                raise ValueError(
                    f"{head_key} logits must align with target_ids before shift; "
                    f"got {tuple(logits.shape)} vs {tuple(target_ids.shape)}"
                )
            ar_logits = logits[..., :-1, :]
            if ar_logits.shape[:-1] != ar_targets.shape:
                raise ValueError(
                    f"{head_key} autoregressive logits mismatch after shift; "
                    f"got {tuple(ar_logits.shape[:-1])} vs {tuple(ar_targets.shape)}"
                )

            local_targets = torch.full_like(ar_targets, fill_value=-1, dtype=torch.long)
            mask_family = torch.zeros_like(ar_valid, dtype=torch.bool)
            head_vocab_size = 0
            for block in blocks:
                if not isinstance(block, dict):
                    raise TypeError(f"event concept routing[{family_name}] blocks must be dicts")
                offset = int(block.get("offset", 0))
                size = int(block.get("size", 0))
                if size <= 0:
                    continue
                mask_block = ar_valid & (ar_targets >= offset) & (ar_targets < offset + size)
                if mask_block.any():
                    local_targets[mask_block] = (ar_targets[mask_block] - offset + head_vocab_size).to(torch.long)
                mask_family |= mask_block
                head_vocab_size += size

            if head_vocab_size <= 0:
                continue
            if int(ar_logits.shape[-1]) != int(head_vocab_size):
                raise ValueError(
                    f"{head_key} expects vocab_size={head_vocab_size} from event routing, "
                    f"but logits last dim is {int(ar_logits.shape[-1])}"
                )
            if mask_family.any():
                losses = self.ce_loss(ar_logits[mask_family], local_targets[mask_family])
                weighted_loss_sum = weighted_loss_sum + losses.sum()
                count = int(mask_family.sum().item())
                total_count += count
                lane_logs[f"loss_{log_prefix}_{family_name}"] = float(losses.mean().item())
                lane_logs[f"n_{log_prefix}_{family_name}"] = count
            else:
                lane_logs[f"n_{log_prefix}_{family_name}"] = 0
            routed |= mask_family

        unrouted = ar_valid & ~routed
        total_valid = int(ar_valid.sum().item())
        lane_logs[f"frac_{log_prefix}_unrouted"] = (
            float(unrouted.sum().item()) / float(total_valid) if total_valid > 0 else 0.0
        )
        if self.strict_routing and unrouted.any():
            sample = ar_targets[unrouted].detach().flatten()[:8].tolist()
            raise ValueError(
                f"Unrouted {log_prefix} targets encountered (n={int(unrouted.sum())}); sample={sample}."
            )

        lane_logs[f"n_{log_prefix}_supervised"] = int(total_count)
        if total_count <= 0:
            return zero, lane_logs, ar_stats

        loss = weighted_loss_sum / float(total_count)
        total_contrib = float(self.weights.get(weight_key, 1.0)) * loss
        lane_logs[f"loss_{log_prefix}"] = float(loss.item())
        return total_contrib, lane_logs, ar_stats

    def _compute_dt_nll_lane(
        self,
        *,
        pred_mu: torch.Tensor | None,
        pred_sigma: torch.Tensor | None,
        time_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        chunk_start_offsets: torch.Tensor | None,
        weight_key: str,
        log_key: str,
        count_key: str | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        lane_logs: Dict[str, float] = {}
        if pred_mu is None or pred_sigma is None or time_ids is None or attention_mask is None or token_type_ids is None:
            return torch.zeros((), dtype=torch.float32), lane_logs
        if float(self.weights.get(weight_key, 0.0)) <= 0.0:
            return time_ids.new_zeros((), dtype=torch.float32), lane_logs

        pred_mu = torch.nan_to_num(pred_mu, nan=0.0, posinf=0.0, neginf=0.0)
        pred_sigma = torch.nan_to_num(pred_sigma, nan=1.0, posinf=1e6, neginf=1.0)
        if pred_mu.shape != time_ids.shape or pred_sigma.shape != time_ids.shape:
            raise ValueError(
                f"{log_key} predictions must match time_ids shape; "
                f"got {tuple(pred_mu.shape)} / {tuple(pred_sigma.shape)} vs {tuple(time_ids.shape)}"
            )

        content_mask = attention_mask.to(dtype=torch.bool) & (token_type_ids != 0)
        t_flat, m_flat = self._flatten_local_sequences(
            time_ids,
            content_mask,
            chunk_start_offsets=chunk_start_offsets,
        )
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
        dt_h = dt_h_flat.view_as(time_ids)
        mask = mask_flat.view_as(time_ids)
        if not mask.any():
            return time_ids.new_zeros((), dtype=torch.float32), lane_logs

        y_true = torch.log1p(dt_h.clamp(max=28.0 * 24.0))
        mu = pred_mu.to(dtype=y_true.dtype)
        sigma = pred_sigma.to(dtype=y_true.dtype).clamp(min=1e-4)
        nll = 0.5 * ((y_true - mu) / sigma).square() + torch.log(sigma)
        loss = nll[mask].mean()
        total_contrib = float(self.weights.get(weight_key, 1.0)) * loss
        lane_logs[log_key] = float(loss.item())
        lane_logs[str(count_key or f"n_{log_key}_supervised")] = int(mask.sum().item())
        return total_contrib, lane_logs

    def _compute_event_numeric_value_nll_lane(
        self,
        *,
        pred_mu: torch.Tensor | None,
        pred_sigma: torch.Tensor | None,
        event_numeric_values: torch.Tensor | None,
        event_numeric_mask: torch.Tensor | None,
        event_payload_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        token_type_ids: torch.Tensor | None,
        weight_key: str,
        log_key: str,
        count_key: str | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        lane_logs: Dict[str, float] = {}
        if (
            pred_mu is None
            or pred_sigma is None
            or event_numeric_values is None
            or event_numeric_mask is None
            or event_payload_ids is None
            or attention_mask is None
            or token_type_ids is None
        ):
            return torch.zeros((), dtype=torch.float32), lane_logs
        if float(self.weights.get(weight_key, 0.0)) <= 0.0:
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        pred_mu = torch.nan_to_num(pred_mu, nan=0.0, posinf=0.0, neginf=0.0)
        pred_sigma = torch.nan_to_num(pred_sigma, nan=1.0, posinf=1e6, neginf=1.0)
        target_values = event_numeric_values.squeeze(-1) if event_numeric_values.ndim == pred_mu.ndim + 1 else event_numeric_values
        if pred_mu.shape != pred_sigma.shape or pred_mu.shape != target_values.shape:
            raise ValueError(
                f"{log_key} predictions must match event numeric target shape; "
                f"got {tuple(pred_mu.shape)} / {tuple(pred_sigma.shape)} vs {tuple(target_values.shape)}"
            )
        if (
            event_numeric_mask.shape != pred_mu.shape
            or event_payload_ids.shape != pred_mu.shape
            or attention_mask.shape != pred_mu.shape
            or token_type_ids.shape != pred_mu.shape
        ):
            raise ValueError(
                f"{log_key} masks and ids must match prediction shape {tuple(pred_mu.shape)}"
            )

        content_mask = attention_mask.to(dtype=torch.bool) & (token_type_ids != int(self.special_type_id))
        numeric_payload_id = int(EVENT_PAYLOAD_KIND_TO_ID["numeric_measurement"])
        target_numeric_mask = (
            event_numeric_mask.to(dtype=torch.bool)
            & attention_mask.to(dtype=torch.bool)
            & (event_payload_ids == numeric_payload_id)
        )

        value_flat = self._flatten_sequence_feature(target_values)
        content_mask_flat = self._flatten_sequence_feature(content_mask)
        target_numeric_mask_flat = self._flatten_sequence_feature(target_numeric_mask)
        mu_flat = self._flatten_sequence_feature(pred_mu)
        sigma_flat = self._flatten_sequence_feature(pred_sigma).clamp(min=1e-4)

        N, S = value_flat.shape
        next_value = torch.zeros_like(value_flat)
        next_is_numeric = torch.zeros((N, S), device=value_flat.device, dtype=torch.bool)
        next_exists = torch.zeros((N, S), device=value_flat.device, dtype=torch.bool)
        last_value = torch.zeros((N,), device=value_flat.device, dtype=value_flat.dtype)
        last_is_numeric = torch.zeros((N,), device=value_flat.device, dtype=torch.bool)
        has = torch.zeros((N,), device=value_flat.device, dtype=torch.bool)
        for i in range(S - 1, -1, -1):
            next_value[:, i] = last_value
            next_is_numeric[:, i] = last_is_numeric
            next_exists[:, i] = has
            cur = content_mask_flat[:, i]
            last_value = torch.where(cur, value_flat[:, i], last_value)
            last_is_numeric = torch.where(cur, target_numeric_mask_flat[:, i], last_is_numeric)
            has = has | cur

        supervise_mask = content_mask_flat & next_exists & next_is_numeric
        if not supervise_mask.any():
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        y_true = next_value.to(dtype=mu_flat.dtype)
        nll = 0.5 * ((y_true - mu_flat) / sigma_flat).square() + torch.log(sigma_flat)
        loss = nll[supervise_mask].mean()
        total_contrib = float(self.weights.get(weight_key, 1.0)) * loss
        lane_logs[log_key] = float(loss.item())
        lane_logs[str(count_key or f"n_{log_key}_supervised")] = int(supervise_mask.sum().item())
        return total_contrib, lane_logs

    def _compute_next_window_gap_nll_lane(
        self,
        *,
        pred_mu: torch.Tensor | None,
        pred_sigma: torch.Tensor | None,
        targets_dict: dict,
        window_mask: torch.Tensor | None,
        weight_key: str,
        log_key: str,
        count_key: str | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        lane_logs: Dict[str, float] = {}
        window_start_times = targets_dict.get("window_start_times", None)
        if pred_mu is None or pred_sigma is None or window_start_times is None:
            return torch.zeros((), dtype=torch.float32), lane_logs
        if float(self.weights.get(weight_key, 0.0)) <= 0.0:
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        pred_mu = torch.nan_to_num(pred_mu, nan=0.0, posinf=0.0, neginf=0.0)
        pred_sigma = torch.nan_to_num(pred_sigma, nan=1.0, posinf=1e6, neginf=1.0)
        if pred_mu.shape != pred_sigma.shape or pred_mu.shape != window_start_times.shape:
            raise ValueError(
                f"{log_key} predictions must match window_start_times shape; "
                f"got {tuple(pred_mu.shape)} / {tuple(pred_sigma.shape)} vs {tuple(window_start_times.shape)}"
            )

        win_mask, _, true_dur_h = self._window_targets(targets_dict, window_mask=window_mask)
        if true_dur_h.shape != pred_mu.shape:
            raise ValueError(
                f"{log_key} derived window durations must match prediction shape; "
                f"got {tuple(true_dur_h.shape)} vs {tuple(pred_mu.shape)}"
            )
        if pred_mu.shape[1] < 2:
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        next_gap_h = (
            window_start_times[:, 1:].to(dtype=true_dur_h.dtype)
            - (
                window_start_times[:, :-1].to(dtype=true_dur_h.dtype)
                + true_dur_h[:, :-1]
            )
        ).clamp(min=0.0)
        supervise_mask = win_mask[:, :-1] & win_mask[:, 1:]
        if not supervise_mask.any():
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        max_gap_hours = 365.25 * 24.0 * 10.0
        y_true = torch.log1p(next_gap_h.clamp(max=max_gap_hours))
        mu = pred_mu[:, :-1].to(dtype=y_true.dtype)
        sigma = pred_sigma[:, :-1].to(dtype=y_true.dtype).clamp(min=1e-4)
        nll = 0.5 * ((y_true - mu) / sigma).square() + torch.log(sigma)
        loss = nll[supervise_mask].mean()
        total_contrib = float(self.weights.get(weight_key, 1.0)) * loss
        lane_logs[log_key] = float(loss.item())
        lane_logs[str(count_key or f"n_{log_key}_supervised")] = int(supervise_mask.sum().item())
        return total_contrib, lane_logs

    def _compute_next_window_duration_nll_lane(
        self,
        *,
        pred_mu: torch.Tensor | None,
        pred_sigma: torch.Tensor | None,
        targets_dict: dict,
        window_mask: torch.Tensor | None,
        weight_key: str,
        log_key: str,
        count_key: str | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        lane_logs: Dict[str, float] = {}
        if pred_mu is None or pred_sigma is None:
            return torch.zeros((), dtype=torch.float32), lane_logs
        if float(self.weights.get(weight_key, 0.0)) <= 0.0:
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        pred_mu = torch.nan_to_num(pred_mu, nan=0.0, posinf=0.0, neginf=0.0)
        pred_sigma = torch.nan_to_num(pred_sigma, nan=1.0, posinf=1e6, neginf=1.0)

        win_mask, _, true_dur_h = self._window_targets(targets_dict, window_mask=window_mask)
        if pred_mu.shape != pred_sigma.shape or pred_mu.shape != true_dur_h.shape:
            raise ValueError(
                f"{log_key} predictions must match semantic duration targets; "
                f"got {tuple(pred_mu.shape)} / {tuple(pred_sigma.shape)} vs {tuple(true_dur_h.shape)}"
            )
        if pred_mu.shape[1] < 2:
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        supervise_mask = win_mask[:, :-1] & win_mask[:, 1:]
        if not supervise_mask.any():
            return pred_mu.new_zeros((), dtype=torch.float32), lane_logs

        max_duration_hours = 28.0 * 24.0
        y_true = torch.log1p(true_dur_h[:, 1:].clamp(max=max_duration_hours))
        mu = pred_mu[:, :-1].to(dtype=y_true.dtype)
        sigma = pred_sigma[:, :-1].to(dtype=y_true.dtype).clamp(min=1e-4)
        nll = 0.5 * ((y_true - mu) / sigma).square() + torch.log(sigma)
        loss = nll[supervise_mask].mean()
        total_contrib = float(self.weights.get(weight_key, 1.0)) * loss
        lane_logs[log_key] = float(loss.item())
        lane_logs[str(count_key or f"n_{log_key}_supervised")] = int(supervise_mask.sum().item())
        return total_contrib, lane_logs

    def _compute_next_window_support_bce_lane(
        self,
        *,
        pred_logits: torch.Tensor | None,
        targets_dict: dict,
        window_mask: torch.Tensor | None,
        weight_key: str,
        log_key: str,
        count_key: str | None = None,
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        lane_logs: Dict[str, float] = {}
        event_type_ids = targets_dict.get("event_type_ids", None)
        event_attention_mask = targets_dict.get("event_attention_mask", None)
        if pred_logits is None or event_type_ids is None or event_attention_mask is None:
            return torch.zeros((), dtype=torch.float32), lane_logs
        if float(self.weights.get(weight_key, 0.0)) <= 0.0:
            return pred_logits.new_zeros((), dtype=torch.float32), lane_logs

        pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=0.0, neginf=0.0)
        if pred_logits.ndim != 3 or pred_logits.shape[-1] != int(NUM_SUPPORT_FLAGS):
            raise ValueError(
                f"{log_key} logits must be (B,W,{int(NUM_SUPPORT_FLAGS)}), got shape {tuple(pred_logits.shape)}"
            )

        if window_mask is None:
            window_mask = torch.ones(pred_logits.shape[:2], device=pred_logits.device, dtype=torch.long)
        if tuple(window_mask.shape) != tuple(pred_logits.shape[:2]):
            raise ValueError(
                f"{log_key} window_mask must match logits (B,W), got {tuple(window_mask.shape)} vs {tuple(pred_logits.shape[:2])}"
            )
        if pred_logits.shape[1] < 2:
            return pred_logits.new_zeros((), dtype=torch.float32), lane_logs

        support_flags = build_window_support_flags(
            event_type_ids=event_type_ids,
            event_attention_mask=event_attention_mask,
            event_memory_chronic_flags=targets_dict.get("event_memory_chronic_flags", None),
            event_numeric_values=targets_dict.get("event_numeric_values", None),
            event_numeric_mask=targets_dict.get("event_numeric_mask", None),
        ).to(device=pred_logits.device, dtype=pred_logits.dtype)
        if tuple(support_flags.shape) != tuple(pred_logits.shape):
            raise ValueError(
                f"{log_key} derived support flags must match logits shape; "
                f"got {tuple(support_flags.shape)} vs {tuple(pred_logits.shape)}"
            )

        supervise_mask = window_mask[:, :-1].to(dtype=torch.bool) & window_mask[:, 1:].to(dtype=torch.bool)
        if not supervise_mask.any():
            return pred_logits.new_zeros((), dtype=torch.float32), lane_logs

        target_support = support_flags[:, 1:, :]
        logits = pred_logits[:, :-1, :]
        expanded_mask = supervise_mask.unsqueeze(-1).expand_as(logits)
        raw_loss = F.binary_cross_entropy_with_logits(
            logits[expanded_mask],
            target_support[expanded_mask],
            reduction="mean",
        )
        total_contrib = float(self.weights.get(weight_key, 1.0)) * raw_loss
        lane_logs[log_key] = float(raw_loss.item())
        pred_binary = (logits > 0.0).to(dtype=target_support.dtype)
        acc = (pred_binary[expanded_mask] == target_support[expanded_mask]).to(dtype=torch.float32).mean()
        lane_logs["acc_next_window_support"] = float(acc.item())
        lane_logs[str(count_key or f"n_{log_key}_supervised")] = int(supervise_mask.sum().item())
        return total_contrib, lane_logs

    def _compute_precedent_phase4_losses(
        self,
        *,
        head_outputs: dict,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        precedent_future_summary = head_outputs.get("precedent_future_summary", None)
        precedent_future_embedding = head_outputs.get("precedent_future_embedding", None)
        precedent_query_embedding = head_outputs.get("precedent_query_embedding", None)
        precedent_retrieval_scores = head_outputs.get("precedent_retrieval_scores", None)
        precedent_candidate_future_summaries = head_outputs.get("precedent_candidate_future_summaries", None)
        precedent_candidate_future_embeddings = head_outputs.get("precedent_candidate_future_embeddings", None)
        precedent_matched_item_ids = head_outputs.get("precedent_matched_item_ids", None)
        precedent_target_future_summary = head_outputs.get("precedent_target_future_summary", None)
        precedent_target_future_embedding = head_outputs.get("precedent_target_future_embedding", None)
        precedent_target_future_mask = head_outputs.get("precedent_target_future_mask", None)
        precedent_anchor_item_ids = head_outputs.get("precedent_anchor_item_ids", None)

        base = None
        for tensor in (
            precedent_future_summary,
            precedent_future_embedding,
            precedent_query_embedding,
            precedent_retrieval_scores,
            precedent_target_future_summary,
        ):
            if torch.is_tensor(tensor):
                base = tensor
                break
        if base is None:
            return torch.zeros((), dtype=torch.float32), {}

        total = base.new_zeros((), dtype=torch.float32)
        logs: dict[str, float] = {}

        if (
            precedent_future_summary is not None
            and precedent_target_future_summary is not None
            and precedent_target_future_mask is not None
        ):
            future_mask = precedent_target_future_mask.to(dtype=torch.bool)
            if future_mask.any():
                pred_summary = torch.nan_to_num(
                    precedent_future_summary,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).to(dtype=torch.float32)
                target_summary = torch.nan_to_num(
                    precedent_target_future_summary,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ).to(dtype=torch.float32)
                expanded_mask = future_mask.unsqueeze(-1).expand_as(pred_summary)
                raw_loss = F.smooth_l1_loss(
                    pred_summary[expanded_mask],
                    target_summary[expanded_mask],
                    reduction="mean",
                )
                future_loss = raw_loss
                logs["loss_precedent_future_summary"] = float(raw_loss.item())

                if (
                    precedent_future_embedding is not None
                    and precedent_target_future_embedding is not None
                ):
                    pred_emb = torch.nan_to_num(
                        precedent_future_embedding,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).to(dtype=torch.float32)
                    target_emb = torch.nan_to_num(
                        precedent_target_future_embedding,
                        nan=0.0,
                        posinf=0.0,
                        neginf=0.0,
                    ).to(dtype=torch.float32)
                    cos_loss = (
                        1.0
                        - F.cosine_similarity(
                            pred_emb[future_mask],
                            target_emb[future_mask],
                            dim=-1,
                            eps=1e-6,
                        )
                    ).mean()
                    future_loss = 0.5 * (future_loss + cos_loss)
                    logs["loss_precedent_future_embedding"] = float(cos_loss.item())

                total = total + float(self.weights.get("precedent_future", 0.0)) * future_loss
                logs["loss_precedent_future"] = float(future_loss.item())
                logs["n_precedent_future_supervised"] = int(future_mask.sum().item())

        if (
            precedent_retrieval_scores is not None
            and precedent_candidate_future_summaries is not None
            and precedent_target_future_summary is not None
            and precedent_target_future_mask is not None
        ):
            retrieval_scores = torch.nan_to_num(
                precedent_retrieval_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(dtype=torch.float32)
            candidate_summaries = torch.nan_to_num(
                precedent_candidate_future_summaries,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(dtype=torch.float32)
            target_summary = torch.nan_to_num(
                precedent_target_future_summary,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(dtype=torch.float32)
            future_mask = precedent_target_future_mask.to(dtype=torch.bool)
            if precedent_matched_item_ids is not None:
                candidate_valid = precedent_matched_item_ids >= 0
            else:
                candidate_valid = torch.ones_like(retrieval_scores, dtype=torch.bool)
            positive_mask = future_mask & candidate_valid.any(dim=-1)
            if positive_mask.any():
                target_summary_exp = target_summary.unsqueeze(-2).expand_as(candidate_summaries)
                candidate_sim = F.cosine_similarity(
                    candidate_summaries,
                    target_summary_exp,
                    dim=-1,
                    eps=1e-6,
                )
                candidate_sim = candidate_sim.masked_fill(~candidate_valid, float("-inf"))
                positive_idx = candidate_sim.argmax(dim=-1)
                contrast_loss = F.cross_entropy(
                    retrieval_scores[positive_mask],
                    positive_idx[positive_mask].to(dtype=torch.long),
                    reduction="mean",
                )
                total = total + float(self.weights.get("precedent_contrast", 0.0)) * contrast_loss
                logs["loss_precedent_contrast"] = float(contrast_loss.item())
                pred_idx = retrieval_scores.argmax(dim=-1)
                contrast_acc = (
                    pred_idx[positive_mask] == positive_idx[positive_mask]
                ).to(dtype=torch.float32).mean()
                logs["acc_precedent_contrast"] = float(contrast_acc.item())
                logs["n_precedent_contrast_supervised"] = int(positive_mask.sum().item())

        if (
            precedent_retrieval_scores is not None
            and precedent_matched_item_ids is not None
            and precedent_anchor_item_ids is not None
        ):
            retrieval_scores = torch.nan_to_num(
                precedent_retrieval_scores,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).to(dtype=torch.float32)
            anchor_ids = precedent_anchor_item_ids.to(dtype=torch.long)
            candidate_match = precedent_matched_item_ids.to(dtype=torch.long) == anchor_ids.unsqueeze(-1)
            anchor_mask = (anchor_ids >= 0) & candidate_match.any(dim=-1)
            if anchor_mask.any():
                anchor_idx = candidate_match.to(dtype=torch.long).argmax(dim=-1)
                anchor_loss = F.cross_entropy(
                    retrieval_scores[anchor_mask],
                    anchor_idx[anchor_mask],
                    reduction="mean",
                )
                total = total + float(self.weights.get("precedent_anchor", 0.0)) * anchor_loss
                logs["loss_precedent_anchor"] = float(anchor_loss.item())
                pred_idx = retrieval_scores.argmax(dim=-1)
                anchor_acc = (
                    pred_idx[anchor_mask] == anchor_idx[anchor_mask]
                ).to(dtype=torch.float32).mean()
                logs["acc_precedent_anchor"] = float(anchor_acc.item())
                logs["n_precedent_anchor_supervised"] = int(anchor_mask.sum().item())

        return total, logs

    def forward(self, head_outputs, targets_dict):
        target_ids = targets_dict["input_ids"]
        attention_mask = targets_dict.get("attention_mask", None)
        valid = attention_mask.to(dtype=torch.bool) if attention_mask is not None else torch.ones_like(target_ids, dtype=torch.bool)
        token_type_ids = targets_dict.get("token_type_ids", None)
        window_type_ids = targets_dict.get("window_type_ids", None)
        window_mask = targets_dict.get("window_mask", None)

        total_loss = target_ids.new_zeros((), dtype=torch.float32)
        logs = {}

        logits_token = head_outputs.get("logits_token", None)
        using_unified_token_loss = (
            logits_token is not None and bool(self.prefer_unified_token_loss)
        )

        marker_type_mask_all, marker_end_mask_all, marker_continue_mask_all = self._marker_masks(target_ids)
        marker_any_mask_all = marker_type_mask_all | marker_end_mask_all | marker_continue_mask_all

        if using_unified_token_loss:
            token_loss, token_logs, ar_stats = self._compute_autoregressive_ce_lane(
                logits=logits_token,
                target_ids=target_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                marker_ids=None,
                weight_key="token",
                log_prefix="token",
                apply_token_family_weights=True,
                emit_token_family_metrics=True,
            )
            total_loss = total_loss + token_loss
            logs.update(token_logs)
            logs["candidate_nonmarker_special_targets"] = int(ar_stats["candidate_nonmarker_special_targets"])
            logs["ignored_nonmarker_special_targets"] = int(ar_stats["ignored_nonmarker_special_targets"])
            logs["frac_unrouted"] = 0.0

        event_target_ids = targets_dict.get("event_input_ids", None)
        event_attention_mask = targets_dict.get("event_attention_mask", None)
        event_token_type_ids = targets_dict.get("event_type_ids", None)
        event_payload_ids = targets_dict.get("event_payload_ids", None)
        event_time_ids = targets_dict.get("event_time_ids", None)
        event_numeric_values = targets_dict.get("event_numeric_values", None)
        event_numeric_mask = targets_dict.get("event_numeric_mask", None)
        if (
            event_target_ids is not None
            and event_attention_mask is not None
            and event_token_type_ids is not None
        ):
            event_token_loss, event_token_logs, event_stats = self._compute_autoregressive_ce_lane(
                logits=head_outputs.get("logits_event_token", None),
                target_ids=event_target_ids,
                attention_mask=event_attention_mask,
                token_type_ids=event_token_type_ids,
                marker_ids=event_target_ids,
                weight_key="event_token",
                log_prefix="event_token",
                apply_token_family_weights=True,
                emit_token_family_metrics=True,
            )
            total_loss = total_loss + event_token_loss
            logs.update(event_token_logs)
            logs["candidate_nonmarker_special_event_targets"] = int(
                event_stats["candidate_nonmarker_special_targets"]
            )
            logs["ignored_nonmarker_special_event_targets"] = int(
                event_stats["ignored_nonmarker_special_targets"]
            )

            event_family_loss, event_family_logs, event_family_stats = self._compute_autoregressive_ce_lane(
                logits=head_outputs.get("logits_event_family", None),
                target_ids=event_token_type_ids,
                attention_mask=event_attention_mask,
                token_type_ids=event_token_type_ids,
                marker_ids=event_target_ids,
                weight_key="event_family",
                log_prefix="event_family",
            )
            total_loss = total_loss + event_family_loss
            logs.update(event_family_logs)
            logs["candidate_nonmarker_special_event_family_targets"] = int(
                event_family_stats["candidate_nonmarker_special_targets"]
            )
            logs["ignored_nonmarker_special_event_family_targets"] = int(
                event_family_stats["ignored_nonmarker_special_targets"]
            )

            event_payload_loss, event_payload_logs, event_payload_stats = self._compute_autoregressive_ce_lane(
                logits=head_outputs.get("logits_event_payload", None),
                target_ids=event_payload_ids,
                attention_mask=event_attention_mask,
                token_type_ids=event_token_type_ids,
                marker_ids=event_target_ids,
                weight_key="event_payload",
                log_prefix="event_payload",
            )
            total_loss = total_loss + event_payload_loss
            logs.update(event_payload_logs)
            logs["candidate_nonmarker_special_event_payload_targets"] = int(
                event_payload_stats["candidate_nonmarker_special_targets"]
            )
            logs["ignored_nonmarker_special_event_payload_targets"] = int(
                event_payload_stats["ignored_nonmarker_special_targets"]
            )

            event_concept_loss, event_concept_logs, event_concept_stats = self._compute_autoregressive_routed_ce_lane(
                head_outputs=head_outputs,
                target_ids=event_target_ids,
                attention_mask=event_attention_mask,
                token_type_ids=event_token_type_ids,
                marker_ids=event_target_ids,
                routing=self.event_concept_routing,
                head_key_prefix="logits_event_concept_",
                weight_key="event_concept",
                log_prefix="event_concept",
            )
            total_loss = total_loss + event_concept_loss
            logs.update(event_concept_logs)
            logs["candidate_nonmarker_special_event_concept_targets"] = int(
                event_concept_stats["candidate_nonmarker_special_targets"]
            )
            logs["ignored_nonmarker_special_event_concept_targets"] = int(
                event_concept_stats["ignored_nonmarker_special_targets"]
            )

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
        if not using_unified_token_loss:
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

        predictive_markers = self._predictive_marker_masks(
            target_ids=target_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        predict_continue_mask = predictive_markers["predict_continue"]
        predict_end_mask = predictive_markers["predict_end"]
        predict_next_type_mask = predictive_markers["next_type"]

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
            logits_transition_boundary = logits_transition_boundary[..., :-1, :]
            if logits_transition_boundary.shape[:-1] != predict_end_mask.shape:
                raise ValueError(
                    "logits_transition_boundary autoregressive slice must align with predictive masks; "
                    f"got {tuple(logits_transition_boundary.shape[:-1])} vs {tuple(predict_end_mask.shape)}"
                )

            supervise_continue = predict_continue_mask
            supervise_end = predict_end_mask
            supervise_mask = supervise_continue | supervise_end

            if supervise_mask.any():
                targets_boundary = torch.zeros_like(predict_end_mask, dtype=torch.long)
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
            logits_boundary_next_window_type = logits_boundary_next_window_type[..., :-1, :]
            if logits_boundary_next_window_type.shape[:-1] != predict_end_mask.shape:
                raise ValueError(
                    "logits_boundary_next_window_type autoregressive slice must align with predictive masks; "
                    f"got {tuple(logits_boundary_next_window_type.shape[:-1])} vs {tuple(predict_end_mask.shape)}"
                )

            if window_type_ids is not None:
                if window_mask is None:
                    window_mask = (
                        self._default_window_mask(attention_mask)
                        if attention_mask is not None
                        else torch.ones_like(window_type_ids, dtype=torch.long)
                    )
                targets_next_type, has_next_window = self._broadcast_next_window_targets(
                    window_type_ids=window_type_ids,
                    window_mask=window_mask,
                    target_shape=predict_end_mask.shape,
                )
                type_boundary_mask = predict_end_mask & has_next_window
            else:
                type_start = int(self._marker_info["type_start"])
                targets_next_type = (target_ids[..., 1:] - type_start).to(dtype=torch.long)
                type_boundary_mask = predict_end_mask & predict_next_type_mask

            if type_boundary_mask.any():
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
        if pred_val is not None and target_vals is not None and float(self.weights.get("val", 1.0)) != 0.0:
            pred_val = torch.nan_to_num(pred_val, nan=0.0, posinf=0.0, neginf=0.0)
            if using_unified_token_loss:
                pred_val_shift = pred_val[..., :-1, :].squeeze(-1)
                target_vals_shift = target_vals[..., 1:, :].squeeze(-1)
                if numeric_mask is not None:
                    mask_val = numeric_mask[..., 1:].to(dtype=torch.bool)
                else:
                    mask_val = target_vals_shift != 0
                if attention_mask is not None:
                    mask_val = mask_val & attention_mask[..., 1:].to(dtype=torch.bool)
                if token_type_ids is not None and self.ignore_nonmarker_special_targets:
                    target_type_ids = token_type_ids[..., 1:]
                    target_marker_mask = marker_any_mask_all[..., 1:]
                    mask_val = mask_val & ~(
                        (target_type_ids == int(self.special_type_id)) & ~target_marker_mask
                    )
                target_vals_used = target_vals_shift
            else:
                pred_val_shift = pred_val.squeeze(-1)
                target_vals_used = target_vals.squeeze(-1)
                if numeric_mask is not None:
                    mask_val = numeric_mask.to(dtype=torch.bool) & valid
                else:
                    mask_val = (target_vals != 0).squeeze(-1) & valid
            if mask_val.any():
                loss = self.mse_loss(pred_val_shift[mask_val], target_vals_used[mask_val])
                total_loss = total_loss + float(self.weights.get("val", 1.0)) * loss.mean()
                logs["loss_val"] = float(loss.mean().item())

        logits_next_window_type = head_outputs.get("logits_next_window_type", None)
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

        time_ids = targets_dict.get("time_ids", None)
        chunk_start_offsets = targets_dict.get("chunk_start_offsets", None)
        dt_loss, dt_logs = self._compute_dt_nll_lane(
            pred_mu=head_outputs.get("pred_dt_next_mu", None),
            pred_sigma=head_outputs.get("pred_dt_next_sigma", None),
            time_ids=time_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            chunk_start_offsets=chunk_start_offsets,
            weight_key="dt",
            log_key="loss_dt_nll",
            count_key="n_dt_supervised",
        )
        total_loss = total_loss + dt_loss.to(device=total_loss.device)
        logs.update(dt_logs)

        event_dt_loss, event_dt_logs = self._compute_dt_nll_lane(
            pred_mu=head_outputs.get("pred_event_dt_next_mu", None),
            pred_sigma=head_outputs.get("pred_event_dt_next_sigma", None),
            time_ids=event_time_ids,
            attention_mask=event_attention_mask,
            token_type_ids=event_token_type_ids,
            chunk_start_offsets=chunk_start_offsets,
            weight_key="event_dt",
            log_key="loss_event_dt_nll",
            count_key="n_event_dt_supervised",
        )
        total_loss = total_loss + event_dt_loss.to(device=total_loss.device)
        logs.update(event_dt_logs)

        event_value_loss, event_value_logs = self._compute_event_numeric_value_nll_lane(
            pred_mu=head_outputs.get("pred_event_value_mu", None),
            pred_sigma=head_outputs.get("pred_event_value_sigma", None),
            event_numeric_values=event_numeric_values,
            event_numeric_mask=event_numeric_mask,
            event_payload_ids=event_payload_ids,
            attention_mask=event_attention_mask,
            token_type_ids=event_token_type_ids,
            weight_key="event_value",
            log_key="loss_event_value_nll",
            count_key="n_event_value_supervised",
        )
        total_loss = total_loss + event_value_loss.to(device=total_loss.device)
        logs.update(event_value_logs)

        next_window_gap_loss, next_window_gap_logs = self._compute_next_window_gap_nll_lane(
            pred_mu=head_outputs.get("pred_next_window_gap_mu", None),
            pred_sigma=head_outputs.get("pred_next_window_gap_sigma", None),
            targets_dict=targets_dict,
            window_mask=window_mask,
            weight_key="next_window_gap",
            log_key="loss_next_window_gap_nll",
            count_key="n_next_window_gap_supervised",
        )
        total_loss = total_loss + next_window_gap_loss.to(device=total_loss.device)
        logs.update(next_window_gap_logs)

        next_window_duration_loss, next_window_duration_logs = self._compute_next_window_duration_nll_lane(
            pred_mu=head_outputs.get("pred_next_window_duration_mu", None),
            pred_sigma=head_outputs.get("pred_next_window_duration_sigma", None),
            targets_dict=targets_dict,
            window_mask=window_mask,
            weight_key="next_window_duration",
            log_key="loss_next_window_duration_nll",
            count_key="n_next_window_duration_supervised",
        )
        total_loss = total_loss + next_window_duration_loss.to(device=total_loss.device)
        logs.update(next_window_duration_logs)

        next_window_support_loss, next_window_support_logs = self._compute_next_window_support_bce_lane(
            pred_logits=head_outputs.get("logits_next_window_support", None),
            targets_dict=targets_dict,
            window_mask=window_mask,
            weight_key="next_window_support",
            log_key="loss_next_window_support_bce",
            count_key="n_next_window_support_supervised",
        )
        total_loss = total_loss + next_window_support_loss.to(device=total_loss.device)
        logs.update(next_window_support_logs)

        precedent_loss, precedent_logs = self._compute_precedent_phase4_losses(
            head_outputs=head_outputs,
        )
        total_loss = total_loss + precedent_loss.to(device=total_loss.device)
        logs.update(precedent_logs)

        if not torch.isfinite(total_loss):
            logs["non_finite_total_loss"] = 1.0
            raise FloatingPointError("AETLossModule produced non-finite total_loss.")

        for key, value in list(logs.items()):
            if isinstance(value, float) and not math.isfinite(value):
                logs[key] = 0.0
                logs["non_finite_log_detected"] = 1.0

        return total_loss, logs
