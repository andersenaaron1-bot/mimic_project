from dataclasses import dataclass
from typing import Dict, Any, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ehr_hier.models.value_encoders.rvq import RVQConfig, ResidualVQ
from src.ehr_hier.models.value_encoders.attn_quant import AttnSelConfig, AttnSelector


@dataclass
class TokenizerConfig:
    d_val: int
    num_codebooks: int = 2
    codebook_size: int = 256
    beta_commit: float = 0.25
    attn_topk: int = 8
    attn_temp: float = 0.5
    aux_soft_weight: float = 0.05   # weight on soft reconstruction aux loss
    aux_entropy_weight: float = 0.01


class ValueDiscretizer(nn.Module):
    """
    CODA-inspired: attention shortlist per level -> RVQ quantization of residuals.
    Returns discrete indices [B,T,L], reconstructed z_hat, scalar loss, and stats.
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
        self.attn = AttnSelector(AttnSelConfig(
            d_val=cfg.d_val,
            codebook_size=cfg.codebook_size,
            num_codebooks=cfg.num_codebooks,
            topk=cfg.attn_topk,
            temperature=cfg.attn_temp,
        ), self.rvq.levels)

    def forward(self, z: torch.Tensor) -> Dict[str, Any]:
        """
        z: [B,T,D] latent from cVAE (detach upstream in trainer).
        """
        # Attention shortlist per level (on current z; using raw residual here is fine for now)
        topk_ids, soft_mix, attn_stats = self.attn(z)           # lists length L
        # RVQ with restriction to top-k per level
        indices, z_hat, vq_loss, rvq_stats = self.rvq(z, restrict_ids=topk_ids)

        # Aux: soft mixture should be close to z or to per-level residuals (use z for simplicity)
        soft_rec_loss = 0.0
        for q_soft in soft_mix:
            soft_rec_loss = soft_rec_loss + F.mse_loss(q_soft, z.reshape(-1, z.shape[-1]).detach())

        loss = vq_loss + self.cfg.aux_soft_weight * soft_rec_loss \
                       - self.cfg.aux_entropy_weight * attn_stats["attn_entropy"]

        stats = {
            **attn_stats,
            **rvq_stats,
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
    # base = [1, K, K^2, ..., K^(L-1)]
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
    # base = [1, K, K^2, ..., K^(L-1)]
    base = torch.cumprod(torch.full((L,), K, device=device, dtype=torch.long), dim=0)
    base[0] = 1
    # digits_i = floor((value_state / base[i])) % K
    return (value_state.unsqueeze(-1) // base.view(1, 1, L)) % K  # [B,T,L]

