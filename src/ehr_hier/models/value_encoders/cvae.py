import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


@dataclass
class ValueCVAEConfig:
    z_dim: int = 64
    var_emb_dim: int = 64
    hidden: int = 128
    # list of context scalars, currently previous measurement delta, age, sex
    use_dt_prev: bool = True
    use_age: bool = True
    use_sex: bool = True
    # regularization
    beta_kl: float = 0.1
    dropout_p: float = 0.0
    # numeric stability for decoder variance
    min_log_sigma: float = -5.0
    max_log_sigma: float = 2.0


class FiLM(nn.Module):
    """Feature-wise Linear Modulation from variable embedding."""
    def __init__(self, var_emb_dim: int, hidden: int):
        super().__init__()
        self.affine = nn.Linear(var_emb_dim, 2 * hidden)

    def forward(self, h: torch.Tensor, var_emb: torch.Tensor) -> torch.Tensor:
        # h: [N, hidden], var_emb: [N, var_emb_dim]
        gamma, beta = self.affine(var_emb).chunk(2, dim=-1)
        # (1 + tanh(gamma)) keeps modulation bounded around 1
        return h * (1 + torch.tanh(gamma)) + beta


class ValueCVAE(nn.Module):
    """
    Conditional VAE over standardized scalar values.
    Conditions on: variable id (via embedding) + optional dt_prev, age, sex etc.

    Trains with .forward(); use .encode_mu() at inference to get z (mu).

    Can extend heads to fit clinical var better, i.e. Students-t, beta
    """
    def __init__(self, n_vars: int, cfg: ValueCVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.var_emb = nn.Embedding(n_vars, cfg.var_emb_dim)

        in_dim = 1  # standardized numeric value
        cond_dim = int(cfg.use_dt_prev) + int(cfg.use_age) + int(cfg.use_sex)

        # Encoder
        self.enc_in = nn.Linear(in_dim + cond_dim, cfg.hidden)
        self.enc_film = FiLM(cfg.var_emb_dim, cfg.hidden)
        self.enc_h = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout_p),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.mu = nn.Linear(cfg.hidden, cfg.z_dim)
        self.logvar = nn.Linear(cfg.hidden, cfg.z_dim)

        # Decoder (Gaussian over standardized value)
        self.dec_z = nn.Linear(cfg.z_dim + cfg.var_emb_dim + cond_dim, cfg.hidden)
        self.dec_h = nn.Sequential(
            nn.GELU(),
            nn.Dropout(cfg.dropout_p),
            nn.Linear(cfg.hidden, cfg.hidden),
            nn.GELU(),
        )
        self.dec_out = nn.Linear(cfg.hidden, 2)  # (mu_hat, log_sigma)

    def _pack_cond(self,
                   dt_prev: Optional[torch.Tensor],
                   age: Optional[torch.Tensor],
                   sex: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        parts = []
        if self.cfg.use_dt_prev:
            assert dt_prev is not None, "cfg.use_dt_prev=True but dt_prev is None"
            parts.append(dt_prev.unsqueeze(-1))
        if self.cfg.use_age:
            assert age is not None, "cfg.use_age=True but age is None"
            parts.append(age.unsqueeze(-1))
        if self.cfg.use_sex:
            assert sex is not None, "cfg.use_sex=True but sex is None"
            parts.append(sex.unsqueeze(-1))
        return torch.cat(parts, dim=-1) if parts else None

    @torch.no_grad()
    def encode_mu(self,
                  value_std: torch.Tensor,   # [B,T] or [N]
                  var_id: torch.Tensor,      # [B,T] or [N]
                  dt_prev: Optional[torch.Tensor] = None,
                  age: Optional[torch.Tensor] = None,
                  sex: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Return encoder mean μ as the continuous latent z for Triplet."""
        # Flatten to [N,*]
        if value_std.dim() == 2:
            B, T = value_std.shape
            v = value_std.reshape(B*T, 1)
            vid = var_id.reshape(B*T)
            dtp = dt_prev.reshape(B*T) if dt_prev is not None else None
            ag  = age.reshape(B*T) if age is not None else None
            sx  = sex.reshape(B*T) if sex is not None else None
            reshape_back = (B, T)
        else:
            v = value_std.unsqueeze(-1)
            vid = var_id
            dtp, ag, sx = dt_prev, age, sex
            reshape_back = None

        cond = self._pack_cond(dtp, ag, sx)  # [N,C] or None
        var_emb = self.var_emb(vid)          # [N,var_emb_dim]

        h = self.enc_in(v if cond is None else torch.cat([v, cond], dim=-1))
        h = self.enc_film(h, var_emb)
        h = self.enc_h(h)
        mu = self.mu(h)                      # [N,z_dim]

        if reshape_back:
            B, T = reshape_back
            mu = mu.reshape(B, T, -1)
        return mu

    def forward(self,
                value_std: torch.Tensor,     # [B,T] or [N]
                var_id: torch.Tensor,        # [B,T] or [N]
                dt_prev: Optional[torch.Tensor] = None,
                age: Optional[torch.Tensor] = None,
                sex: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """Training step: returns {'loss', 'recon_nll', 'kl', ...}."""
        if value_std.dim() == 2:
            B, T = value_std.shape
            v = value_std.reshape(B*T, 1)
            vid = var_id.reshape(B*T)
            dtp = dt_prev.reshape(B*T) if dt_prev is not None else None
            ag  = age.reshape(B*T) if age is not None else None
            sx  = sex.reshape(B*T) if sex is not None else None
        else:
            v = value_std.unsqueeze(-1)
            vid = var_id
            dtp, ag, sx = dt_prev, age, sex

        cond = self._pack_cond(dtp, ag, sx)
        var_emb = self.var_emb(vid)

        # Encoder
        h = self.enc_in(v if cond is None else torch.cat([v, cond], dim=-1))
        h = self.enc_film(h, var_emb)
        h = self.enc_h(h)
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # Decoder
        din = [z, var_emb]
        if cond is not None:
            din.append(cond)
        d = self.dec_z(torch.cat(din, dim=-1))
        d = self.dec_h(d)
        mu_hat, log_sigma = self.dec_out(d).chunk(2, dim=-1)
        log_sigma = log_sigma.clamp(min=self.cfg.min_log_sigma, max=self.cfg.max_log_sigma)

        # Gaussian NLL on standardized value
        recon_nll = 0.5 * (2*log_sigma + ((v - mu_hat) ** 2) / torch.exp(2*log_sigma))
        recon_nll = recon_nll.mean()

        # KL(q||p) with p=N(0,I)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

        loss = recon_nll + self.cfg.beta_kl * kl
        return {
            "loss": loss,
            "recon_nll": recon_nll.detach(),
            "kl": kl.detach(),
            "mu": mu.detach(),
            "z": z.detach(),
            "mu_hat": mu_hat.detach(),
            "log_sigma": log_sigma.detach(),
        }
