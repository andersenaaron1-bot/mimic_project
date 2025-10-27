from dataclasses import dataclass
from typing import Optional
import torch
import torch.nn as nn

from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig
from src.ehr_hier.tokenizers.value_tokenizer import ValueDiscretizer, TokenizerConfig, pack_rvq_indices


@dataclass
class TokenOnlyConfig:
    # embedding dims
    d_code: int = 256
    d_time: int = 64
    d_model: int = 512
    value_token_dim: int = 64

    # base code vocab (MEDS code_ids)
    vocab_size: int = 200000

    # cVAE (used to produce z for quantization; not concatenated)
    value_z_dim: int = 64
    cvae_ckpt: str = "/dss/.../value_cvae.pt"
    n_vars: int = 200000
    use_dt_prev: bool = True
    use_age: bool = True
    use_sex: bool = True

    # discretizer (CODA-style RVQ)
    tokenizer_ckpt: str = "/dss/.../value_tokenizer.pt"
    num_codebooks: int = 2     # L
    codebook_size: int = 256   # K
    pad_value_state_id: int = 0  # 0 reserved for "no numeric value"

    # fold (code,value_state) into a single event id/token embedding
    fold_eic: bool = False  # True → use event embedding; False → separate code + value token


class TokenOnlyEncoder(nn.Module):
    """
    Pure token input:
      - If fold_eic=False: concat [code_emb ⊕ time_emb ⊕ value_token_emb] → proj
      - If fold_eic=True:  build event_id = code * (K^L + 1) + value_state; use event_emb ⊕ time_emb → proj
    Value tokens come from RVQ over the cVAE latent; no continuous z is concatenated.
    """
    def __init__(self, cfg: TokenOnlyConfig, code_encoder: nn.Module, time_encoder: nn.Module):
        super().__init__()
        self.cfg = cfg
        self.code_enc = code_encoder        # should map code_id -> [B,T,d_code]
        self.time_enc = time_encoder        # should map time -> [B,T,d_time]

        # ---- load frozen cVAE (to compute z for quantization) ----
        self.cvae = ValueCVAE(
            n_vars=cfg.n_vars,
            cfg=ValueCVAEConfig(
                z_dim=cfg.value_z_dim,
                use_dt_prev=cfg.use_dt_prev,
                use_age=cfg.use_age,
                use_sex=cfg.use_sex,
            ),
        )
        cvae_ckpt = torch.load(cfg.cvae_ckpt, map_location="cpu")
        self.cvae.load_state_dict(cvae_ckpt["state_dict"], strict=False)
        self.cvae.eval()
        for p in self.cvae.parameters():
            p.requires_grad = False

        # ---- load frozen discretizer (RVQ + attention shortlist) ----
        tok_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
        tcfg = TokenizerConfig(
            d_val=cfg.value_z_dim,
            num_codebooks=cfg.num_codebooks,
            codebook_size=cfg.codebook_size,
            beta_commit=tok_ckpt.get("cfg", {}).get("beta_commit", 0.25),
            attn_topk=tok_ckpt.get("cfg", {}).get("attn_topk", 8),
            attn_temp=tok_ckpt.get("cfg", {}).get("attn_temp", 0.5),
            aux_soft_weight=tok_ckpt.get("cfg", {}).get("aux_soft_weight", 0.05),
            aux_entropy_weight=tok_ckpt.get("cfg", {}).get("aux_entropy_weight", 0.01),
        )
        self.tokenizer = ValueDiscretizer(tcfg)
        self.tokenizer.load_state_dict(tok_ckpt["state_dict"], strict=False)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # ---- embeddings for tokens ----
        # value token vocab size: (K^L) + 1 (PAD=0 for non-numeric)
        self.KpowL = cfg.codebook_size ** cfg.num_codebooks
        self.value_vocab_size = self.KpowL + 1
        self.value_token_emb = nn.Embedding(self.value_vocab_size, cfg.value_token_dim)

        if cfg.fold_eic:
            # event vocab (upper bound): base_code_vocab * value_vocab_size
            self.event_vocab_size = cfg.vocab_size * self.value_vocab_size
            self.event_emb = nn.Embedding(self.event_vocab_size, cfg.d_code)

        # ---- projection to model width ----
        if cfg.fold_eic:
            in_dim = cfg.d_time + cfg.d_code
        else:
            in_dim = cfg.d_code + cfg.d_time + cfg.value_token_dim
        self.post = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, cfg.d_model))

    @torch.no_grad()
    def _value_state_ids(self, batch):
        """
        Compute value_state ids [B,T] from cVAE latent via tokenizer.
        PAD=0 for non-numeric tokens; else 1 + packed base-K id.
        """
        z = self.cvae.encode_mu(
            value_std=batch["value"],
            var_id=batch["code"],
            dt_prev=batch.get("dt_prev"),
            age=batch.get("age"),
            sex=batch.get("sex"),
        )  # [B,T,d_z]
        out = self.tokenizer(z)  # {'indices':[B,T,L], ...}
        vs = pack_rvq_indices(out["indices"], K=self.cfg.codebook_size)  # [B,T] in [0, K^L-1]
        m = batch.get("numeric_mask")
        if m is not None:
            vs = torch.where(m, 1 + vs, torch.zeros_like(vs))
        else:
            vs = 1 + vs
        return vs  # [B,T], 0 = PAD (no numeric)

    def forward(self, batch: dict) -> torch.Tensor:
        e_time = self.time_enc(batch["time"])  # [B,T,d_time]
        vs = self._value_state_ids(batch)      # [B,T]

        if self.cfg.fold_eic:
            # event_id = code * (K^L + 1) + value_state
            event_id = batch["code"].long() * (self.KpowL + 1) + vs
            e_event = self.event_emb(event_id)                 # [B,T,d_code]
            x = torch.cat([e_event, e_time], dim=-1)
        else:
            e_code = self.code_enc(batch["code"])              # [B,T,d_code]
            e_vtok = self.value_token_emb(vs)                  # [B,T,d_value_token_dim]
            x = torch.cat([e_code, e_time, e_vtok], dim=-1)

        return self.post(x)  # [B,T,d_model]

