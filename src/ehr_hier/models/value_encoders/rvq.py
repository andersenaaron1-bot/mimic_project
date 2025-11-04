# src/ehr_hier/models/value_encoders/rvq.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RVQConfig:
    d_val: int                # latent dim coming from cVAE (z_dim)
    num_codebooks: int = 2    # L levels
    codebook_size: int = 256  # K entries per level
    beta_commit: float = 0.25 # commitment loss weight
    use_ema: bool = False     # hook left for alt. updates (not used here)
    eps: float = 1e-5


class VectorQuantizer(nn.Module):
    """Single codebook with straight-through nearest-neighbor quantization."""
    def __init__(self, dim: int, n_codes: int):
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        self.code = nn.Embedding(n_codes, dim)
        nn.init.uniform_(self.code.weight, a=-1.0 / dim, b=1.0 / dim)

    def forward(
        self,
        r: torch.Tensor,                            # [N, D]
        restrict_id: Optional[torch.Tensor] = None  # [N, K_top]
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        if restrict_id is None:
            r2 = (r ** 2).sum(dim=-1, keepdim=True)         # [N,1]
            e2 = (self.code.weight ** 2).sum(dim=-1)        # [K]
            re = r @ self.code.weight.t()                   # [N,K]
            dist = r2 - 2 * re + e2.unsqueeze(0)            # [N,K]
            idx = dist.argmin(dim=-1)                       # [N]
            q = self.code(idx)                              # [N,D]
        else:
            e_subset = self.code(restrict_id)               # [N,K_top,D]
            d = (r.unsqueeze(1) - e_subset)                 # [N,K_top,D]
            dist = (d ** 2).sum(dim=-1)                     # [N,K_top]
            sel = dist.argmin(dim=-1)                       # [N]
            idx = restrict_id.gather(1, sel.unsqueeze(-1)).squeeze(-1)  # [N]
            q = self.code(idx)                              # [N,D]

        with torch.no_grad():
            uniq = torch.unique(idx).numel()
            stats = {"uniq_codes": uniq, "usage_frac": uniq / float(self.n_codes)}
        return q, idx, stats


class ResidualVQ(nn.Module):
    """L-level Residual Vector Quantizer (quantize the current residual at each level)."""
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        self.levels = nn.ModuleList(
            [VectorQuantizer(cfg.d_val, cfg.codebook_size) for _ in range(cfg.num_codebooks)]
        )

    def forward(
        self,
        z: torch.Tensor,                                # [B, T, D]
        restrict_ids: Optional[List[torch.Tensor]] = None  # list[L] of [N, K_top]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        B, T, D = z.shape
        N = B * T
        r = z.reshape(N, D)          # current residual
        z_hat = torch.zeros_like(r)

        all_idx: List[torch.Tensor] = []
        code_loss = 0.0
        commit_loss = 0.0
        uniq_used = 0

        for l, vq in enumerate(self.levels):
            restr = None if restrict_ids is None else restrict_ids[l]      # [N,K_top] or None
            q, idx, st = vq(r, restrict_id=restr)                          # [N,D], [N]

            # Straight-through estimator
            q_st = r + (q - r).detach()
            z_hat = z_hat + q_st

            # CODA/VQGAN-style losses (per level)
            # codebook: || sg[r] - q ||^2      (updates code vectors)
            # commit:   || r - sg[q] ||^2      (encourages encoder outputs near codes)
            code_loss   = code_loss   + F.mse_loss(r.detach(), q, reduction="mean")
            commit_loss = commit_loss + F.mse_loss(r, q.detach(), reduction="mean")

            all_idx.append(idx.view(B, T))
            uniq_used += st["uniq_codes"]

            # residual for next level
            r = r - q_st

        # Combine as in CODA/VQGAN: codebook (weight 1.0) + beta * commitment
        vq_loss = code_loss + self.cfg.beta_commit * commit_loss

        indices = torch.stack(all_idx, dim=-1)  # [B,T,L]
        z_hat = z_hat.view(B, T, D)

        stats = {
            "avg_uniq_codes_per_level": uniq_used / len(self.levels),
            "code_loss":   code_loss.detach(),
            "commit_loss": commit_loss.detach(),
        }
        return indices, z_hat, vq_loss, stats
