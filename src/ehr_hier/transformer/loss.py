import torch
import torch.nn as nn


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
        self.weights = weights or {"struct": 5.0, "rvq": 1.0, "meas": 1.0, "med": 1.0, "val": 1.0, "win": 1.0}

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

        return total_loss, logs
