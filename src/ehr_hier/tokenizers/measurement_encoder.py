from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import math

import torch
import torch.nn as nn

from src.ehr_hier.models.value_encoders.cvae import ValueCVAE, ValueCVAEConfig
from src.ehr_hier.tokenizers.value_tokenizer import (
    ValueDiscretizer,
    TokenizerConfig,
    pack_rvq_indices,
)
from src.ehr_hier.data.token_types import TokenTriplet, TokenCategory


@dataclass
class MeasurementEncoderConfig:
    """
    Config for measurement → value token encoding.
    """
    cvae_ckpt: str
    tokenizer_ckpt: str

    # dimensions / codebook
    z_dim: int = 64
    num_codebooks: int = 2     # L
    codebook_size: int = 256   # K

    # var vocabulary size and mapping
    n_vars: int = 200000

    # per-var normalization (same as used for CVAE training)
    mean_by_var: torch.Tensor = None   # shape [max_var_id+1]
    std_by_var: torch.Tensor = None    # shape [max_var_id+1]

    # code → var_id mapping (string MEDS code → integer var_id)
    code2id: Dict[str, int] = None

    # global token vocab offset for measurement value tokens
    # e.g. if you reserve 0–999 for specials & other types,
    # you can start measurement value tokens at 1000.
    value_token_offset: int = 0

    # how to interpret sex
    male_prefix: str = "m"


class MeasurementTokenEncoder(nn.Module):
    """
    Event-level encoder for MEDS measurement events.

    For each measurement event with a numeric_value:
      - standardizes value using per-var mean/std,
      - computes z via frozen cVAE encoder,
      - discretizes z via frozen ValueDiscretizer (RVQ),
      - packs RVQ indices into a single value_state id in [0, K^L-1],
      - maps to a global token id: value_token_offset + (1 + value_state),
      - returns a single TokenTriplet with category=MEASUREMENT.

    dt_hours (time since previous emitted token) is supplied from outside
    (timeline builder); CVAE’s dt_prev (time since previous same-variable
    measurement) is tracked internally per subject.
    """
    category: TokenCategory = TokenCategory.MEASUREMENT

    def __init__(self, cfg: MeasurementEncoderConfig):
        super().__init__()
        self.cfg = cfg

        assert cfg.mean_by_var is not None and cfg.std_by_var is not None, \
            "mean_by_var and std_by_var must be provided"
        assert cfg.code2id is not None, "code2id mapping must be provided"

        # register normalization as buffers so they move with .to(device)
        self.register_buffer("mean_by_var", cfg.mean_by_var.float())
        self.register_buffer("std_by_var", cfg.std_by_var.float())

        # simple Python dict is fine for code2id; we don't register as buffer
        self.code2id: Dict[str, int] = cfg.code2id

        # ---- load frozen cVAE ----
        cvae_cfg = ValueCVAEConfig(
            z_dim=cfg.z_dim,
            var_emb_dim=64,
            hidden=128,
            use_dt_prev=True,
            use_age=True,
            use_sex=True,
        )
        self.cvae = ValueCVAE(n_vars=cfg.n_vars, cfg=cvae_cfg)
        ckpt = torch.load(cfg.cvae_ckpt, map_location="cpu")
        self.cvae.load_state_dict(ckpt["state_dict"], strict=False)
        self.cvae.eval()
        for p in self.cvae.parameters():
            p.requires_grad = False

        # ---- load frozen tokenizer (RVQ) ----
        tok_ckpt = torch.load(cfg.tokenizer_ckpt, map_location="cpu")
        tcfg = TokenizerConfig(
            d_val=cfg.z_dim,
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

        # internal state for dt_prev per var (per subject)
        self._last_time_by_var: Dict[int, datetime] = {}

    def reset_state(self):
        """
        Call this at the start of each subject to reset dt_prev tracking.
        """
        self._last_time_by_var.clear()

    @torch.no_grad()
    def _compute_dt_prev(self, var_id: int, t: Optional[datetime]) -> float:
        if not isinstance(t, datetime):
            return 0.0
        prev = self._last_time_by_var.get(var_id)
        if prev is None:
            dt_prev = 0.0
        else:
            dt_prev = max(0.0, (t - prev).total_seconds() / 3600.0)
        self._last_time_by_var[var_id] = t
        return dt_prev

    @torch.no_grad()
    def _standardize_value(self, value: float, var_id: int) -> float:
        if var_id < 0 or var_id >= self.mean_by_var.shape[0]:
            return 0.0
        m = float(self.mean_by_var[var_id].item())
        s = float(self.std_by_var[var_id].item())
        if not math.isfinite(s) or s <= 0.0:
            s = 1.0
        return (float(value) - m) / s

    @torch.no_grad()
    def encode_event(self, ev: Any, dt_hours: float) -> List[TokenTriplet]:
        """
        Encode a single MEDS measurement event into one TokenTriplet.

        ev   : meds_reader Event (or wrapper) with attributes:
                 - code (str)
                 - numeric_value (float)
                 - time (datetime)
                 - age_years (optional)
                 - sex (optional, 'M'/'F' etc)
        dt_hours : time since previous emitted token (for transformer time embedding).

        Returns:
          [] if this event has no usable numeric_value or code mapping,
          or [TokenTriplet] for a valid measurement.
        """
        code = getattr(ev, "code", None)
        if code is None:
            return []

        code_str = str(code)
        var_id = self.code2id.get(code_str)
        if var_id is None:
            # unknown code → skip
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

        # standardize using precomputed mean/std
        v_std = self._standardize_value(v_raw, var_id)

        # dt_prev for CVAE conditioning (time since previous same-var event)
        t_ev = getattr(ev, "time", None)
        dt_prev = self._compute_dt_prev(var_id, t_ev)

        # optional age/sex for conditioning
        age_attr = getattr(ev, "age_years", None)
        sex_attr = getattr(ev, "sex", None)

        age = float(age_attr) if isinstance(age_attr, (int, float)) else 0.0
        sex = 0.0
        if isinstance(sex_attr, str):
            sex = 1.0 if sex_attr.strip().lower().startswith(self.cfg.male_prefix) else 0.0

        device = next(self.cvae.parameters()).device

        # prepare single-sample tensors [B=1, T=1]
        value_std = torch.tensor([[v_std]], dtype=torch.float32, device=device)
        var_id_t  = torch.tensor([[var_id]], dtype=torch.long, device=device)
        dt_prev_t = torch.tensor([[dt_prev]], dtype=torch.float32, device=device)
        age_t     = torch.tensor([[age]], dtype=torch.float32, device=device)
        sex_t     = torch.tensor([[sex]], dtype=torch.float32, device=device)

        # latent z via frozen cVAE encoder
        z = self.cvae.encode_mu(
            value_std=value_std,
            var_id=var_id_t,
            dt_prev=dt_prev_t,
            age=age_t,
            sex=sex_t,
        )  # [1,1,z_dim]

        # discretize z via tokenizer
        out = self.tokenizer(z)  # {'indices': [1,1,L], ...}
        indices = out["indices"]           # [1,1,L]
        vs = pack_rvq_indices(indices, K=self.cfg.codebook_size)  # [1,1]
        value_state = int(vs[0, 0].item())  # 0 .. K^L-1

        # map to global token id for measurements
        # 0 is reserved for PAD at value_state level, so we use 1+value_state
        value_state_id = 1 + value_state
        global_token_id = self.cfg.value_token_offset + value_state_id

        triplet = TokenTriplet(
            value_id=global_token_id,
            category_id=int(TokenCategory.MEASUREMENT),
            dt_hours=float(dt_hours),
        )
        return [triplet]
