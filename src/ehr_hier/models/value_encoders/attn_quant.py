# src/ehr_hier/models/value_encoders/attn_quant.py
from dataclasses import dataclass
from typing import Dict, Any, Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttnSelConfig:
    d_val: int
    codebook_size: int
    num_codebooks: int = 2
    topk: int = 8
    temperature: float = 0.5
    use_proj: bool = True
    # EMA of code usage (CODA-style global statistics)
    usage_ema_decay: float = 0.99   # higher -> slower, stabler
    usage_loss_mix: float = 0.10    # β in text; small >0 keeps gradients via batch
    eps: float = 1e-8


class AttnSelector(nn.Module):
    """
    Attention shortlist per RVQ level using the *current residual*.
    Entropy regularizer is computed on an EMA-smoothed global usage distribution.
    """
    def __init__(self, cfg: AttnSelConfig, codebooks: nn.ModuleList):
        super().__init__()
        self.cfg = cfg
        self.codebooks = codebooks
        self.q_proj = nn.Linear(cfg.d_val, cfg.d_val, bias=False) if cfg.use_proj else nn.Identity()

        L = len(self.codebooks)
        K = cfg.codebook_size
        # EMA buffers: per-level unnormalized usage counts and their totals
        self.register_buffer("usage_ema", torch.zeros(L, K))
        self.register_buffer("usage_ema_sum", torch.full((L,), cfg.eps))

    def forward(self, z: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        """
        z: [B,T,D]
        Returns:
          topk_ids: list[L] of [N, topk] LongTensors
          soft_mix: list[L] of [N, D] tensors (level contributions)
          stats: {"attn_entropy": scalar tensor used in loss,
                  "attn_entropy_det": detached copy for logging}
        """
        B, T, D = z.shape
        N = B * T
        eps = self.cfg.eps
        r = z.reshape(N, D)

        topk_ids: List[torch.Tensor] = []
        soft_mix: List[torch.Tensor] = []
        entropies: List[torch.Tensor] = []

        for l, vq in enumerate(self.codebooks):
            E = vq.code.weight                  # [K, D]
            K = E.shape[0]

            # Query from current residual
            q = self.q_proj(r)                  # [N, D]
            logits = q @ E.t() / (D ** 0.5)     # [N, K]

            k = max(1, min(self.cfg.topk, K))
            vals, ids = torch.topk(logits, k=k, dim=-1)  # [N, k], [N, k]
            w = F.softmax(vals / (self.cfg.temperature + eps), dim=-1)   # [N, k]

            # Level contribution and residual update
            E_top = self._gather_codes(E, ids)                                   # [N, k, D]
            q_soft = torch.sum(E_top * w.unsqueeze(-1), dim=1)                   # [N, D]
            r = r - q_soft

            # --- EMA-smoothed global usage entropy (CODA-style) ---
            # Batch soft usage counts (differentiable):
            usage_batch = torch.zeros(K, device=E.device, dtype=w.dtype)         # [K]
            usage_batch = usage_batch.scatter_add(0, ids.reshape(-1), w.reshape(-1))

            # Update EMA (no grad)
            with torch.no_grad():
                d = self.cfg.usage_ema_decay
                self.usage_ema[l] = d * self.usage_ema[l] + (1 - d) * usage_batch
                self.usage_ema_sum[l] = d * self.usage_ema_sum[l] + (1 - d) * usage_batch.sum().clamp_min(eps)

            # Smoothed usage for loss (keeps gradient via usage_batch only)
            beta = self.cfg.usage_loss_mix
            num = (1 - beta) * self.usage_ema[l].detach() + beta * usage_batch   # [K]
            p_smooth = (num + eps) / (num.sum() + eps * K)                        # [K]
            H = -(p_smooth * (p_smooth.clamp_min(eps)).log()).sum()              # scalar

            topk_ids.append(ids)
            soft_mix.append(q_soft)
            entropies.append(H)

        H_usage = torch.stack(entropies).mean()
        stats: Dict[str, Any] = {
            "attn_entropy": H_usage,               # use this in your tokenizer loss
            "attn_entropy_det": H_usage.detach(),  # for logging
        }
        return topk_ids, soft_mix, stats

    @staticmethod
    def _gather_codes(E: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # E: [K, D], ids: [N, k] -> [N, k, D]
        return E.index_select(0, ids.reshape(-1)).view(ids.shape[0], ids.shape[1], -1)
