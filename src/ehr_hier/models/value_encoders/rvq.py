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
    use_ema: bool = False     # (hook left in place; this impl uses ST by default)
    eps: float = 1e-5


class VectorQuantizer(nn.Module):
    """Single codebook with straight-through nearest-neighbor quantization.
       Optionally can be extended to EMA updates (not enabled here for brevity)."""
    def __init__(self, dim: int, n_codes: int):
        super().__init__()
        self.dim = dim
        self.n_codes = n_codes
        # nn.Embedding makes it easy to gather code vectors by index
        self.code = nn.Embedding(n_codes, dim)
        nn.init.uniform_(self.code.weight, a=-1.0 / dim, b=1.0 / dim)

    def forward(self, r: torch.Tensor,
                restrict_id: Optional[torch.Tensor] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        r: [N, D] residual to quantize
        restrict_id: [N, K_top] optional shortlist of candidate ids per sample
        Returns: q [N, D], idx [N], stats dict
        """
        # Compute nearest neighbor (either over full codebook or shortlist)
        if restrict_id is None:
            # distances to all codes
            # dist = ||r||^2 - 2 r·e + ||e||^2; nearest = argmin dist
            r2 = (r ** 2).sum(dim=-1, keepdim=True)            # [N,1]
            e2 = (self.code.weight ** 2).sum(dim=-1)           # [K]
            re = r @ self.code.weight.t()                      # [N,K]
            dist = r2 - 2*re + e2.unsqueeze(0)                 # [N,K]
            idx = dist.argmin(dim=-1)                          # [N]
            q = self.code(idx)                                 # [N,D]
        else:
            # restrict per sample to top-k ids
            # gather code vectors and find nearest within subset
            # restrict_id: [N, K_top]
            e_subset = self.code(restrict_id)                  # [N,K_top,D]
            # compute squared distances
            d = (r.unsqueeze(1) - e_subset)                    # [N, K_top, D]
            dist = (d ** 2).sum(dim=-1)                        # [N, K_top]
            sel = dist.argmin(dim=-1)                          # [N]
            idx = restrict_id.gather(1, sel.unsqueeze(-1)).squeeze(-1)  # [N]
            q = self.code(idx)                                 # [N,D]

        # Stats
        with torch.no_grad():
            # batch perplexity proxy: how many unique codes used
            uniq = torch.unique(idx).numel()
            stats = {"uniq_codes": uniq, "usage_frac": uniq / float(self.n_codes)}
        return q, idx, stats


class ResidualVQ(nn.Module):
    """L-level Residual Vector Quantizer. Each level quantizes the current residual."""
    def __init__(self, cfg: RVQConfig):
        super().__init__()
        self.cfg = cfg
        self.levels = nn.ModuleList([VectorQuantizer(cfg.d_val, cfg.codebook_size)
                                     for _ in range(cfg.num_codebooks)])

    def forward(self, z: torch.Tensor,
                restrict_ids: Optional[List[torch.Tensor]] = None
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        z: [B, T, D] latent from cVAE
        restrict_ids: optional list length L; each is [N, K_top] shortlist per level
        Returns:
          indices: [B, T, L] long
          z_hat:   [B, T, D]
          vq_loss: scalar
          stats:   dict
        """
        B, T, D = z.shape
        N = B * T
        r = z.reshape(N, D)                 # residual at current level
        z_hat = torch.zeros_like(r)
        all_idx = []
        total_commit = 0.0
        uniq_used = 0

        for l, vq in enumerate(self.levels):
            restr = None if restrict_ids is None else restrict_ids[l]  # [N,K_top]
            q, idx, st = vq(r, restrict_id=restr)                      # [N,D], [N]
            # Straight-through estimator: preserve gradient to r
            q_st = r + (q - r).detach()
            z_hat = z_hat + q_st
            # commitment losses (VQ-VAE style)
            commit = F.mse_loss(r.detach(), q, reduction="mean") + F.mse_loss(r, q.detach(), reduction="mean")
            total_commit = total_commit + commit
            all_idx.append(idx.view(B, T))
            uniq_used += st["uniq_codes"]

            # update residual for next level
            r = r - q_st

        vq_loss = self.cfg.beta_commit * total_commit
        indices = torch.stack(all_idx, dim=-1)  # [B,T,L]
        z_hat = z_hat.view(B, T, D)

        stats = {
            "avg_uniq_codes_per_level": uniq_used / len(self.levels),
            "commit_loss": total_commit.detach(),
        }
        return indices, z_hat, vq_loss, stats
