import torch
import torch.nn as nn


class AETLossModule(nn.Module):
    def __init__(self, vocab_offsets, weights=None):
        super().__init__()
        self.offsets = vocab_offsets
        # Default weights prioritize structure heavily
        self.weights = weights or {'struct': 5.0, 'rvq': 1.0, 'med': 1.0, 'val': 1.0}

        self.ce_loss = nn.CrossEntropyLoss(reduction='none')  # No ignore_index needed if we mask carefully
        self.mse_loss = nn.MSELoss(reduction='none')

    def forward(self, head_outputs, targets_dict):
        """
        Args:
            head_outputs: Dict from AETOutputHeads
            targets_dict: From Collator containing:
                - 'input_ids': The shifted targets (next token)
                - 'token_type_ids': The mask (SPECIAL vs RVQ vs MED)
                - 'numeric_values': The regression target
        """
        # Unpack
        logits_struct = head_outputs['logits_struct']
        logits_rvq = head_outputs['logits_rvq']
        logits_med = head_outputs['logits_medtok']
        pred_val = head_outputs['pred_values']

        target_ids = targets_dict['input_ids']  # (B, S) - Global IDs
        token_types = targets_dict['token_type_ids']  # (B, S)
        target_vals = targets_dict['numeric_values']  # (B, S, 1)

        total_loss = 0.0
        logs = {}

        # --- 1. Structural Loss ---
        # Mask: Where type is SPECIAL (0)
        mask_struct = (token_types == 0)
        if mask_struct.any():
            # Shift Global ID to Local ID: Target = Global - Offset
            # Since SPECIAL usually starts at 0, Global == Local
            local_targets = target_ids[mask_struct] - self.offsets['SPECIAL']

            # Extract relevant logits (flattening batch/seq)
            relevant_logits = logits_struct[mask_struct]  # (N_valid, Vocab_Struct)

            loss = self.ce_loss(relevant_logits, local_targets)
            total_loss += self.weights['struct'] * loss.mean()
            logs['loss_struct'] = loss.mean().item()

        # --- 2. RVQ Loss ---
        # Mask: Where type is RVQ
        mask_rvq = (token_types == 1)  # Assuming 1 is mapped to RVQ type
        if mask_rvq.any():
            local_targets = target_ids[mask_rvq] - self.offsets['RVQ']
            relevant_logits = logits_rvq[mask_rvq]

            loss = self.ce_loss(relevant_logits, local_targets)
            total_loss += self.weights['rvq'] * loss.mean()
            logs['loss_rvq'] = loss.mean().item()

        # --- 3. MedTok/Clinical Loss ---
        # Mask: Where type is MED, DIAG, or PROC (Shared Head)
        # You might map these all to type_id=2 in your Collator
        mask_med = (token_types >= 2)
        if mask_med.any():
            local_targets = target_ids[mask_med] - self.offsets['MED']  # Assuming MED is start of block
            relevant_logits = logits_med[mask_med]

            loss = self.ce_loss(relevant_logits, local_targets)
            total_loss += self.weights['med'] * loss.mean()
            logs['loss_med'] = loss.mean().item()

        # --- 4. Regression Loss ---
        # Only where value is non-zero (or explicit mask provided)
        mask_val = (target_vals != 0).squeeze()
        if mask_val.any():
            loss = self.mse_loss(pred_val.squeeze()[mask_val], target_vals.squeeze()[mask_val])
            total_loss += self.weights['val'] * loss.mean()
            logs['loss_val'] = loss.mean().item()

        return total_loss, logs