# src/ehr_hier/models/value_encoders/attn_quant.py
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List
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
    use_proj: bool = True    # learnable query projection


class AttnSelector(nn.Module):
    """
    Attention-based shortlist over codewords per level.
    It computes logits ~ <W r, E_l> and returns top-k indices per sample.
    Optionally returns a soft mixture q_soft for an auxiliary loss.
    """
    def __init__(self, cfg: AttnSelConfig, codebooks: nn.ModuleList):
        super().__init__()
        self.cfg = cfg
        self.codebooks = codebooks  # the RVQ codebooks (to read their embeddings)
        self.q_proj = nn.Linear(cfg.d_val, cfg.d_val, bias=False) if cfg.use_proj else nn.Identity()

    def forward(self, z: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor], Dict[str, Any]]:
        """
        z: [B,T,D] latent (or current residual).
        Returns:
          topk_ids:  list of length L with [N, topk] LongTensor
          q_soft:    list of length L with [N, D] soft mixtures (optional)
          stats:     dict (entropy etc.)
        """
        B, T, D = z.shape
        N = B * T
        r = z.reshape(N, D)
        q = self.q_proj(r)                                     # [N,D]
        topk_ids, soft_mix, entropies = [], [], []

        for l, vq in enumerate(self.codebooks):
            # codebook embeddings: [K,D]
            E = vq.code.weight                                 # nn.Embedding
            # logits ~ dot-product; scale by sqrt(D)
            logits = q @ E.t() / (D ** 0.5)                    # [N,K]
            # top-k shortlist
            topk = torch.topk(logits, k=min(self.cfg.topk, E.shape[0]), dim=-1)
            ids = topk.indices                                 # [N,topk]
            vals = topk.values                                 # [N,topk]
            # soft mixture for aux loss (optional)
            w = F.softmax(vals / self.cfg.temperature, dim=-1) # [N,topk]
            q_soft = torch.sum(self._gather_codes(E, ids) * w.unsqueeze(-1), dim=1)  # [N,D]
            # entropy (higher is better utilization)
            entropy = -(w * (w.clamp_min(1e-8)).log()).sum(dim=-1).mean()

            topk_ids.append(ids)
            soft_mix.append(q_soft)
            entropies.append(entropy)

        stats = {"attn_entropy": torch.stack(entropies).mean().detach()}
        return topk_ids, soft_mix, stats

    @staticmethod
    def _gather_codes(E: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        # E: [K,D], ids: [N,topk] -> [N,topk,D]
        return E.index_select(0, ids.reshape(-1)).view(ids.shape[0], ids.shape[1], -1)
