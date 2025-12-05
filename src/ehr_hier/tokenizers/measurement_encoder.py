from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import math
import torch
import torch.nn as nn

from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig
from src.ehr_hier.tokenizers.value_tokenizer import (
    ValueDiscretizer, TokenizerConfig,
)
from src.ehr_hier.data.token_types import EventToken, TokenCategory


@dataclass
class MeasurementEncoderConfig:
    cvae_ckpt: str
    tokenizer_ckpt: str
    # per-var normalization (same artifact used in training)
    mean_by_var: torch.Tensor
    std_by_var: torch.Tensor
    # frozen mapping (same as training)
    code2id: Dict[str, int]
    # global vocab layout
    value_token_offset: int = 0  # legacy (used as RVQ base if rvq_token_offset not set)
    code_token_offset: int = 900  # rough default from SCHEMA_TOKENS
    rvq_token_offset: int = 100   # rough default from SCHEMA_TOKENS
    rvq_codebook_stride: Optional[int] = None

    # normalization hyperparams (match training)
    value_clip: float = 8.0
    max_dt_prev_hours: float = 24.0 * 7 * 4   # 4 weeks
    max_age_years: float = 100.0
    male_prefix: str = "m"


class MeasurementTokenEncoder(nn.Module):
    category: TokenCategory = TokenCategory.MEASUREMENT

    def __init__(self, cfg: MeasurementEncoderConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.mean_by_var is not None and cfg.std_by_var is not None
        assert isinstance(cfg.code2id, dict) and len(cfg.code2id) > 0

        # register stats as buffers so .to(device) works
        self.register_buffer("mean_by_var", cfg.mean_by_var.float())
        self.register_buffer("std_by_var", cfg.std_by_var.float())

        self.code2id: Dict[str, int] = cfg.code2id

        # ---- load cVAE exactly as trained ----
        cvae_ckpt = torch.load(cfg.cvae_ckpt, map_location="cpu")
        if "cfg" in cvae_ckpt:
            cvae_cfg = ValueCVAEConfig(**cvae_ckpt["cfg"])
        else:
            # last resort; better to always save cfg in ckpt
            raise ValueError("cVAE checkpoint missing cfg; please save training cfg into ckpt")

        # n_vars must match checkpoint var_emb
        n_vars_ckpt = cvae_ckpt["state_dict"]["var_emb.weight"].shape[0]
        self.cvae = ValueCVAE(n_vars=n_vars_ckpt, cfg=cvae_cfg)
        self.cvae.load_state_dict(cvae_ckpt["state_dict"], strict=True)
        self.cvae.eval()
        for p in self.cvae.parameters():
            p.requires_grad = False

        # ---- load tokenizer (RVQ) exactly as trained ----
        tok_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
        if "cfg" in tok_ckpt:
            tok_cfg = TokenizerConfig(**tok_ckpt["cfg"])
        else:
            # fallback if older tokenizer ckpt
            tok_cfg = TokenizerConfig(d_val=cvae_cfg.z_dim)
        self.tokenizer = ValueDiscretizer(tok_cfg)
        self.tokenizer.load_state_dict(tok_ckpt["state_dict"], strict=True)
        self.tokenizer.eval()
        for p in self.tokenizer.parameters():
            p.requires_grad = False

        # per-subject dt_prev state
        self._last_time_by_var: Dict[int, datetime] = {}

        # precompute K^L for range checks (optional)
        self._K = self.tokenizer.cfg.codebook_size
        self._L = self.tokenizer.cfg.num_codebooks
        self._KpowL = self._K ** self._L

    def reset_state(self):
        self._last_time_by_var.clear()

    @torch.no_grad()
    def _compute_dt_prev(self, var_id: int, t: Optional[datetime]) -> float:
        if not isinstance(t, datetime):
            return 0.0
        prev = self._last_time_by_var.get(var_id)
        dt = 0.0 if prev is None else max(0.0, (t - prev).total_seconds() / 3600.0)
        self._last_time_by_var[var_id] = t
        return dt

    @torch.no_grad()
    def _value_z(self, v_raw: float, var_id: int) -> float:
        # safe indexing
        if var_id <= 0 or var_id >= self.mean_by_var.shape[0]:
            return 0.0
        m = float(self.mean_by_var[var_id].item())
        s = float(self.std_by_var[var_id].item())
        if not math.isfinite(s) or s <= 0.0:
            s = 1.0
        z = (float(v_raw) - m) / s
        # clip exactly like training
        z = max(-self.cfg.value_clip, min(self.cfg.value_clip, z))
        return z

    @torch.no_grad()
    def _norm_dt_prev(self, dt_hours: float) -> float:
        h = max(0.0, min(float(dt_hours), self.cfg.max_dt_prev_hours))
        return math.log1p(h)

    @torch.no_grad()
    def _norm_age(self, age_years: Optional[float]) -> float:
        if age_years is None or not math.isfinite(float(age_years)):
            return 0.0
        a = max(0.0, min(float(age_years), self.cfg.max_age_years))
        return a / self.cfg.max_age_years

    @torch.no_grad()
    def _norm_sex(self, sex_attr: Optional[str]) -> float:
        if not isinstance(sex_attr, str):
            return 0.0
        return 1.0 if sex_attr.strip().lower().startswith(self.cfg.male_prefix) else 0.0

    @torch.no_grad()
    def encode_event(self, ev: Any, dt_hours: float) -> List[EventToken]:
        code = getattr(ev, "code", None)
        if code is None:
            return []

        var_id = self.code2id.get(str(code))
        if var_id is None or var_id <= 0:
            # unknown measurement code under strict mapping → skip
            return []

        v_raw = getattr(ev, "numeric_value", None)
        if v_raw is None:
            return []
        try:
            v_raw = float(v_raw)
        except (TypeError, ValueError):
            return []
        if not math.isfinite(v_raw):
            return []

        zval = self._value_z(v_raw, var_id)

        t_ev = getattr(ev, "time", None)
        dt_prev = self._compute_dt_prev(var_id, t_ev)
        dt_prev_n = self._norm_dt_prev(dt_prev)

        age_n = self._norm_age(getattr(ev, "age_years", None))
        sex_n = self._norm_sex(getattr(ev, "sex", None))

        device = next(self.cvae.parameters()).device
        # shape [1, 1] for each field to match encode_mu’s flattening
        value_std = torch.tensor([[zval]], dtype=torch.float32, device=device)
        var_id_t  = torch.tensor([[var_id]], dtype=torch.long, device=device)
        dt_prev_t = torch.tensor([[dt_prev_n]], dtype=torch.float32, device=device)
        age_t     = torch.tensor([[age_n]], dtype=torch.float32, device=device)
        sex_t     = torch.tensor([[sex_n]], dtype=torch.float32, device=device)

        z = self.cvae.encode_mu(
            value_std=value_std,
            var_id=var_id_t,
            dt_prev=dt_prev_t,
            age=age_t,
            sex=sex_t,
        )  # [1,1,D]

        out = self.tokenizer(z)                       # indices: [1,1,L]
        indices = out["indices"][0, 0]                # [L]

        # Token 0: code token for the measurement variable
        code_token_id = self.cfg.code_token_offset + int(var_id)
        tokens: List[EventToken] = [
            EventToken(
                value_id=code_token_id,
                category_id=int(TokenCategory.MEASUREMENT),
                t_from_start_hours=0.0,
                dt_from_prev_hours=float(dt_hours),
                cat_attrs={"var_id": int(var_id)},
                num_attrs={"z": float(zval)},
            )
        ]

        # Tokens 1..L: one per RVQ codebook
        stride = self.cfg.rvq_codebook_stride or self._K
        rvq_base = self.cfg.rvq_token_offset if self.cfg.rvq_token_offset is not None else self.cfg.value_token_offset
        for i, idx in enumerate(indices):
            token_id = rvq_base + i * stride + int(idx.item())
            tokens.append(
                EventToken(
                    value_id=token_id,
                    category_id=int(TokenCategory.MEASUREMENT),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=0.0,
                    cat_attrs={"var_id": int(var_id), "codebook": i},
                    num_attrs={},
                )
            )

        return tokens
