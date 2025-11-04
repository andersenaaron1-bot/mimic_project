from dataclasses import dataclass
from typing import Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ehr_hier.models.value_encoders.rvq import RVQConfig, ResidualVQ
from src.ehr_hier.models.value_encoders.attn_quant import AttnSelConfig, AttnSelector


@dataclass
class TokenizerConfig:
    # Latent & RVQ
    d_val: int
    num_codebooks: int = 2
    codebook_size: int = 256
    beta_commit: float = 0.25

    # Attention shortlist
    attn_topk: int = 8
    attn_temp: float = 0.5

    # EMA global usage entropy (CODA-style)
    attn_usage_ema_decay: float = 0.99    # EMA smoothing of usage histogram
    attn_usage_loss_mix: float = 0.10     # small batch mix-in to carry gradients
    attn_eps: float = 1e-8

    # Loss weights
    aux_soft_weight: float = 0.05         # residual-aligned soft mixture aux loss
    aux_entropy_weight: float = 0.01      # maximize global usage entropy


class ValueDiscretizer(nn.Module):
    """
    CODA-inspired tokenizer:
      - Per-level attention shortlist computed from the *current residual*.
      - Residual VQ with straight-through estimator.
      - Loss = VQ(codebook + beta*commit) + λ_soft * soft(residual) - λ_e * global-usage-entropy.
    Returns:
      indices: [B,T,L]  (discrete codes per level)
      z_hat:   [B,T,D]  (quantized reconstruction)
      loss:    scalar
      stats:   dict (includes attn & rvq diagnostics; attn_entropy is differentiable)
    """
    def __init__(self, cfg: TokenizerConfig):
        super().__init__()
        self.cfg = cfg

        self.rvq = ResidualVQ(RVQConfig(
            d_val=cfg.d_val,
            num_codebooks=cfg.num_codebooks,
            codebook_size=cfg.codebook_size,
            beta_commit=cfg.beta_commit,
        ))

        self.attn = AttnSelector(
            AttnSelConfig(
                d_val=cfg.d_val,
                codebook_size=cfg.codebook_size,
                num_codebooks=cfg.num_codebooks,
                topk=cfg.attn_topk,
                temperature=cfg.attn_temp,
                use_proj=True,
                usage_ema_decay=cfg.attn_usage_ema_decay,
                usage_loss_mix=cfg.attn_usage_loss_mix,
                eps=cfg.attn_eps,
            ),
            self.rvq.levels
        )

    def forward(self, z: torch.Tensor) -> Dict[str, Any]:
        """
        z: [B,T,D] CVAE latent. (Detach upstream if CVAE is frozen.)
        """
        # 1) Per-level shortlist from residuals + soft mixtures (q_soft_ℓ)
        topk_ids, soft_mix, attn_stats = self.attn(z)  # lists of length L

        # 2) RVQ with shortlist restriction
        indices, z_hat, vq_loss, rvq_stats = self.rvq(z, restrict_ids=topk_ids)

        # 3) Residual-aligned soft auxiliary:
        #    q_soft_ℓ should approximate r_ℓ (the running residual).
        B, T, D = z.shape
        r = z.reshape(B * T, D)
        soft_rec_loss = 0.0
        for q_soft in soft_mix:                 # q_soft: [N,D], N=B*T
            soft_rec_loss = soft_rec_loss + F.mse_loss(q_soft, r.detach())
            r = r - q_soft

        # 4) Final loss: VQ + λ_soft * soft(residual) - λ_e * global-usage-entropy
        loss = vq_loss \
             + self.cfg.aux_soft_weight * soft_rec_loss \
             - self.cfg.aux_entropy_weight * attn_stats["attn_entropy"]

        stats = {
            **rvq_stats,
            # attn_entropy is differentiable; attn_entropy_det is safe for logging
            **{k: (v.detach() if torch.is_tensor(v) else v) for k, v in attn_stats.items() if k != "attn_entropy"},
            "attn_entropy": attn_stats["attn_entropy"].detach(),
            "vq_loss": vq_loss.detach(),
            "soft_rec": soft_rec_loss.detach(),
        }
        return {"indices": indices, "z_hat": z_hat, "loss": loss, "stats": stats}


# ---- helpers to pack/unpack L indices into one value_state id (base-K) ----

def pack_rvq_indices(indices: torch.Tensor, K: int) -> torch.Tensor:
    """
    indices: [B,T,L] with digits in [0, K-1] (least significant at level 0)
    Returns: value_state [B,T] encoding base-K digits.
    """
    assert indices.dtype in (torch.long, torch.int64)
    L = indices.shape[-1]
    device = indices.device
    base = torch.cumprod(torch.full((L,), K, device=device, dtype=torch.long), dim=0)
    base[0] = 1
    return (indices * base.view(1, 1, L)).sum(dim=-1)  # [B,T]


def unpack_rvq_indices(value_state: torch.Tensor, L: int, K: int) -> torch.Tensor:
    """
    value_state: [B,T] long
    Returns: indices: [B,T,L] with digits in [0, K-1] (least significant at level 0)
    """
    assert value_state.dtype in (torch.long, torch.int64)
    device = value_state.device
    base = torch.cumprod(torch.full((L,), K, device=device, dtype=torch.long), dim=0)
    base[0] = 1
    return (value_state.unsqueeze(-1) // base.view(1, 1, L)) % K  # [B,T,L]
