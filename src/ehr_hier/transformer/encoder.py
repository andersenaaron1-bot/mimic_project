import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from src.ehr_hier.data.token_types import TokenCategory


def _alibi_slopes(num_heads: int) -> torch.Tensor:
    """
    Head-specific slope initialization from the ALiBi paper.

    This returns positive slopes; larger slopes = stronger recency bias.
    """

    def _slopes_power_of_2(n: int) -> list[float]:
        start = 2 ** (-2 ** -(math.log2(n) - 3))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    n = int(num_heads)
    if n <= 0:
        raise ValueError(f"num_heads must be > 0, got {num_heads}")

    if math.log2(n).is_integer():
        slopes = _slopes_power_of_2(n)
    else:
        closest = 2 ** math.floor(math.log2(n))
        slopes = _slopes_power_of_2(closest)
        slopes_extra = _slopes_power_of_2(2 * closest)[0::2][: n - closest]
        slopes = slopes + slopes_extra

    return torch.tensor(slopes, dtype=torch.float32)


class WindowAttentionPooler(nn.Module):
    """
    Temperature-scaled attention pooling over a window's token hidden states.

    This produces a single vector summary per window by learning token importance
    scores and taking a softmax-weighted sum. A lower temperature (< 1.0) makes
    the distribution peakier (closer to selecting a few "critical" tokens).
    """

    def __init__(self, d_model: int, *, temperature: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        self.scorer = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor, pool_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden: (N, L, D) token hidden states.
            pool_mask: (N, L) bool/0-1 mask, True/1 indicates eligible tokens.

        Returns:
            summary: (N, D) pooled window summary.
        """
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be (N,L,D), got shape {tuple(hidden.shape)}")
        if pool_mask.ndim != 2:
            raise ValueError(f"pool_mask must be (N,L), got shape {tuple(pool_mask.shape)}")
        if hidden.shape[:2] != pool_mask.shape:
            raise ValueError(
                f"hidden and pool_mask must match on (N,L); got {tuple(hidden.shape[:2])} vs {tuple(pool_mask.shape)}"
            )

        mask = pool_mask.to(dtype=torch.bool, device=hidden.device)
        logits = self.scorer(self.dropout(hidden)).squeeze(-1)  # (N, L)
        temperature = max(1e-4, float(self.temperature))
        logits = logits / temperature

        # If a row is fully masked, fall back to selecting the first token to keep
        # the operation defined (these rows should typically be excluded upstream).
        has_any = mask.any(dim=-1)
        if not has_any.all():
            logits = logits.clone()
            missing = ~has_any
            logits[missing] = float("-inf")
            logits[missing, 0] = 0.0
            mask = mask.clone()
            mask[missing, 0] = True

        logits = logits.masked_fill(~mask, float("-inf"))
        attn = F.softmax(logits, dim=-1)
        summary = torch.einsum("nl,nld->nd", attn, hidden)
        return summary


class WindowMultiQueryPooler(nn.Module):
    """
    Multi-query attention pooling over a window's token hidden states.

    Produces `Q` summary vectors per window by learning `Q` independent scoring
    functions over tokens and taking Q softmax-weighted sums.
    """

    def __init__(self, d_model: int, *, num_queries: int, temperature: float = 1.0, dropout: float = 0.0) -> None:
        super().__init__()
        if int(num_queries) <= 0:
            raise ValueError(f"num_queries must be > 0, got {num_queries}")
        self.num_queries = int(num_queries)
        self.scorer = nn.Linear(d_model, self.num_queries)
        self.dropout = nn.Dropout(dropout)
        self.temperature = float(temperature)

    def forward(self, hidden: torch.Tensor, pool_mask: torch.Tensor, *, query_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Args:
            hidden: (N, L, D) token hidden states.
            pool_mask: (N, L) bool/0-1 mask, True/1 indicates eligible tokens.
            query_mask: Optional (N, Q, L) mask to restrict tokens per query. This is
                        intersected with pool_mask, with a per-query fallback to pool_mask
                        if a query selects no tokens for a given row.

        Returns:
            summaries: (N, Q, D)
        """
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be (N,L,D), got shape {tuple(hidden.shape)}")
        if pool_mask.ndim != 2:
            raise ValueError(f"pool_mask must be (N,L), got shape {tuple(pool_mask.shape)}")
        if hidden.shape[:2] != pool_mask.shape:
            raise ValueError(
                f"hidden and pool_mask must match on (N,L); got {tuple(hidden.shape[:2])} vs {tuple(pool_mask.shape)}"
            )
        if query_mask is not None:
            if query_mask.ndim != 3:
                raise ValueError(f"query_mask must be (N,Q,L), got shape {tuple(query_mask.shape)}")
            if query_mask.shape[0] != hidden.shape[0] or query_mask.shape[2] != hidden.shape[1]:
                raise ValueError(
                    "query_mask must match hidden on (N,L); "
                    f"got {tuple(query_mask.shape)} vs expected (N,Q,L)=({hidden.shape[0]},*,{hidden.shape[1]})"
                )
            if query_mask.shape[1] != self.num_queries:
                raise ValueError(
                    f"query_mask Q must equal num_queries={self.num_queries}; got {int(query_mask.shape[1])}"
                )

        mask = pool_mask.to(dtype=torch.bool, device=hidden.device)  # (N,L)
        logits = self.scorer(self.dropout(hidden))  # (N,L,Q)
        temperature = max(1e-4, float(self.temperature))
        logits = logits / temperature
        logits = logits.transpose(1, 2)  # (N,Q,L)

        # Build per-query mask and fallback for empty queries.
        qmask = mask.unsqueeze(1).expand(hidden.shape[0], self.num_queries, hidden.shape[1])  # (N,Q,L)
        if query_mask is not None:
            qmask = qmask & query_mask.to(dtype=torch.bool, device=hidden.device)

            has_any_q = qmask.any(dim=-1)  # (N,Q)
            if not has_any_q.all():
                # fallback to the generic pool mask for queries that selected nothing
                fallback = mask.unsqueeze(1).expand_as(qmask)
                qmask = torch.where(has_any_q.unsqueeze(-1), qmask, fallback)

        # If a row is fully masked (no tokens at all), fall back to selecting token 0 for all queries.
        has_any_row = mask.any(dim=-1)  # (N,)
        if not has_any_row.all():
            logits = logits.clone()
            missing = ~has_any_row
            logits[missing] = float("-inf")
            logits[missing, :, 0] = 0.0
            qmask = qmask.clone()
            qmask[missing, :, 0] = True

        logits = logits.masked_fill(~qmask, float("-inf"))
        attn = F.softmax(logits, dim=-1)  # (N,Q,L)
        summaries = torch.einsum("nql,nld->nqd", attn, hidden)
        return summaries


class AETCausalAttention(nn.Module):
    """
    Custom Multi-Head Attention that supports Continuous RoPE (cRoPE).
    Enforces Causal Masking (Auto-regressive) + Padding Masking.
    """

    def __init__(
        self,
        d_model,
        num_heads,
        rope_module,
        dropout=0.1,
        *,
        enable_alibi_hours_bias: bool = False,
        alibi_hours_max: float = 28.0 * 24.0,
        alibi_hours_slope_scale: float = 1.0,
    ):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_head = d_model // num_heads
        self.num_heads = num_heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.rope = rope_module  # Instance of ContinuousRotaryPositionalEmbedding
        self.enable_alibi_hours_bias = bool(enable_alibi_hours_bias)
        self.alibi_hours_max = float(alibi_hours_max)

        if self.enable_alibi_hours_bias:
            slopes = _alibi_slopes(self.num_heads) * float(alibi_hours_slope_scale)
            # Store inverse-softplus so F.softplus(param) starts at `slopes`.
            raw = torch.log(torch.expm1(slopes).clamp(min=1e-8))
            self._alibi_slopes_raw = nn.Parameter(raw)
        else:
            self._alibi_slopes_raw = None

        # Projections
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, times, attention_mask=None):
        """
        Args:
            x: (Batch, Seq_Len, d_model)
            times: (Batch, Seq_Len) - Cumulative float times for RoPE
            attention_mask: (Batch, Seq_Len) - 1 for Real, 0 for Pad
        """
        B, S, D = x.shape

        # 1. Project Q, K, V
        q = self.q_proj(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, S, d_head)
        k = self.k_proj(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        # 2. Apply cRoPE per head (avoid mixing heads)
        times_exp = times[:, None, :]  # (B,1,S)
        q_flat = q.reshape(B * self.num_heads, S, self.d_head)
        k_flat = k.reshape(B * self.num_heads, S, self.d_head)
        times_flat = times_exp.expand(B, self.num_heads, S).reshape(B * self.num_heads, S)
        q_rot = self.rope(q_flat, times_flat).reshape(B, self.num_heads, S, self.d_head)
        k_rot = self.rope(k_flat, times_flat).reshape(B, self.num_heads, S, self.d_head)

        # 3. Scaled Dot-Product Attention
        # (B, H, S, S)
        attn_weights = torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale

        # 3b. Optional continuous-time ALiBi-style bias (hours).
        # Bias is negative and grows with time separation, encouraging attention to
        # recent tokens in *time*, not just in sequence index.
        if self._alibi_slopes_raw is not None:
            # (B,S,S) pairwise non-negative time deltas in hours (causal => i>=j).
            t = times.to(dtype=torch.float32)
            dt = (t[:, :, None] - t[:, None, :]).clamp(min=0.0)
            if self.alibi_hours_max > 0:
                dt = dt.clamp(max=self.alibi_hours_max)
            log_dt = torch.log1p(dt)  # (B,S,S)

            slopes = F.softplus(self._alibi_slopes_raw).to(device=attn_weights.device, dtype=attn_weights.dtype)  # (H,)
            bias = -log_dt.to(device=attn_weights.device, dtype=attn_weights.dtype).unsqueeze(1) * slopes.view(1, self.num_heads, 1, 1)
            attn_weights = attn_weights + bias

        # 4. Masking
        # A. Causal Mask (Upper Triangular = -inf)
        causal_mask = torch.triu(torch.ones(S, S, device=x.device), diagonal=1).bool()
        attn_weights.masked_fill_(causal_mask, float('-inf'))

        # B. Padding Mask (if provided)
        if attention_mask is not None:
            extended_mask = attention_mask[:, None, None, :]  # (B, 1, 1, S)
            attn_weights = attn_weights.masked_fill(extended_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        # Fully padded rows can produce all -inf before softmax; sanitize to avoid NaN
        # propagation through the residual stream.
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0, posinf=0.0, neginf=0.0)
        if attention_mask is not None:
            q_mask = attention_mask[:, None, :, None].to(dtype=attn_weights.dtype)  # (B,1,S,1)
            attn_weights = attn_weights * q_mask
        attn_weights = self.dropout(attn_weights)

        # 5. Output
        out = torch.matmul(attn_weights, v)  # (B, H, S, d_h)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        q_mask_bool = None
        if attention_mask is not None:
            q_mask_bool = attention_mask[:, :, None].to(dtype=torch.bool)  # (B,S,1)
            out = torch.where(q_mask_bool, out, torch.zeros_like(out))
        out = self.out_proj(out)
        if q_mask_bool is not None:
            out = torch.where(q_mask_bool, out, torch.zeros_like(out))
        return out


class AETEncoderLayer(nn.Module):
    """
    Standard Transformer Block:
    Input -> LayerNorm -> CausalAttn -> Add -> LayerNorm -> FFN -> Add
    """

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        rope_module,
        dropout=0.1,
        *,
        enable_alibi_hours_bias: bool = False,
        alibi_hours_max: float = 28.0 * 24.0,
        alibi_hours_slope_scale: float = 1.0,
    ):
        super().__init__()
        self.attn = AETCausalAttention(
            d_model,
            num_heads,
            rope_module,
            dropout,
            enable_alibi_hours_bias=enable_alibi_hours_bias,
            alibi_hours_max=alibi_hours_max,
            alibi_hours_slope_scale=alibi_hours_slope_scale,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, times, mask=None):
        # Pre-Norm Architecture (Better stability)
        x_norm = self.norm1(x)
        attn_out = self.attn(x_norm, times, mask)
        x = x + attn_out

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x


class AETLocalEncoder(nn.Module):
    """
    The "Ribs" of the architecture.
    Processes (Batch, Windows, Tokens) by flattening Batch & Windows.
    """

    def __init__(self, config, rope_module):
        super().__init__()
        self.config = config

        pool_temperature = float(getattr(config, "summary_pool_temperature", 1.0))
        pool_dropout = float(getattr(config, "summary_pool_dropout", config.dropout))
        self.exclude_special_from_summary = bool(getattr(config, "exclude_special_from_summary", True))
        self.special_type_id = int(getattr(config, "special_type_id", 0))
        self.summary_num_queries = int(
            getattr(config, "summary_num_queries", getattr(config, "num_summary_queries", 1))
        )
        self.summary_query_use_category_masks = bool(getattr(config, "summary_query_use_category_masks", True))
        self.summary_fuse_terminal = bool(getattr(config, "summary_fuse_terminal", True))
        self.summary_terminal_fuser = (
            nn.Sequential(
                nn.Linear(2 * config.d_model, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, 1),
            )
            if self.summary_fuse_terminal
            else None
        )

        if self.summary_num_queries <= 1:
            self.summary_pooler = WindowAttentionPooler(
                d_model=config.d_model,
                temperature=pool_temperature,
                dropout=pool_dropout,
            )
            self.summary_combine = None
        else:
            self.summary_pooler = WindowMultiQueryPooler(
                d_model=config.d_model,
                num_queries=self.summary_num_queries,
                temperature=pool_temperature,
                dropout=pool_dropout,
            )
            self.summary_combine = nn.Linear(self.summary_num_queries * config.d_model, config.d_model)

        self.enable_window_meta = bool(getattr(config, "enable_window_meta", False))
        if self.enable_window_meta:
            # [log1p(n_tokens), log1p(duration_h), log1p(density), frac_struct, frac_med, frac_meas]
            self.window_meta_proj = nn.Sequential(
                nn.Linear(6, config.d_model),
                nn.GELU(),
                nn.Linear(config.d_model, config.d_model),
            )
            self.window_meta_scale = nn.Parameter(torch.tensor(1.0))
        else:
            self.window_meta_proj = None
            self.window_meta_scale = None

        self.layers = nn.ModuleList([
            AETEncoderLayer(
                d_model=config.d_model,
                num_heads=config.num_heads,
                d_ff=config.d_ff,
                rope_module=rope_module,
                dropout=config.dropout,
                enable_alibi_hours_bias=bool(getattr(config, "enable_alibi_hours_bias", False)),
                alibi_hours_max=float(getattr(config, "alibi_hours_max", 28.0 * 24.0)),
                alibi_hours_slope_scale=float(getattr(config, "alibi_hours_slope_scale", 1.0)),
            ) for _ in range(config.num_local_layers)
        ])

        # Final Norm before passing to Global Aggregator
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, times, attention_mask, token_type_ids=None):
        """
        Args:
            x: (Batch, Num_Windows, Window_Len, Dim) or (Batch, Num_Windows, Num_Chunks, Window_Len, Dim)
            times: matching float tensor without the final Dim
            attention_mask: matching mask tensor without the final Dim
            token_type_ids: Optional matching coarse category tensor.
        Returns:
            x: Contextualized tokens (Same Shape)
            chunk_summaries: (Batch, Num_Windows, Num_Chunks, Dim) or (Batch, Num_Windows, 1, Dim)
        """
        if x.ndim == 4:
            x = x.unsqueeze(2)
            times = times.unsqueeze(2)
            attention_mask = attention_mask.unsqueeze(2)
            if token_type_ids is not None:
                token_type_ids = token_type_ids.unsqueeze(2)
        if x.ndim != 5:
            raise ValueError(f"x must be 4D or 5D, got shape {tuple(x.shape)}")

        B, W, C, L, D = x.shape

        # 1. Flatten Batch, Windows, and local chunks.
        x_flat = x.view(B * W * C, L, D)
        times_flat = times.view(B * W * C, L)
        mask_flat = attention_mask.view(B * W * C, L)

        # 2. Pass through Transformer Layers
        for layer in self.layers:
            x_flat = layer(x_flat, times_flat, mask_flat)

        x_flat = self.norm(x_flat)

        # 3. Unflatten
        x_out = x_flat.view(B, W, C, L, D)

        # 4. Attention-pooled window summaries
        # We pool over eligible (non-pad) tokens, optionally excluding SPECIAL-prefix tokens.
        mask_bool = mask_flat.to(dtype=torch.bool)
        is_real_window = mask_bool.any(dim=-1)  # (B*W*C,)

        window_summaries_flat = torch.zeros((B * W * C, D), device=x.device, dtype=x.dtype)
        if is_real_window.any():
            pool_mask = mask_bool
            types_flat = None
            if token_type_ids is not None and self.exclude_special_from_summary:
                types_flat = token_type_ids.view(B * W * C, L)
                pool_mask = pool_mask & (types_flat != self.special_type_id)

            # For windows where all eligible tokens were excluded (e.g. all SPECIAL),
            # fall back to pooling over non-pad tokens.
            has_any = pool_mask.any(dim=-1)
            if not has_any.all():
                pool_mask = pool_mask.clone()
                pool_mask[~has_any] = mask_bool[~has_any]

            if self.summary_num_queries <= 1:
                pooled = self.summary_pooler(
                    x_flat[is_real_window],
                    pool_mask[is_real_window],
                )
                window_summaries_flat[is_real_window] = pooled.to(dtype=window_summaries_flat.dtype)
            else:
                query_mask = None
                if types_flat is not None and self.summary_query_use_category_masks:
                    # Encourage diversity: dedicate early queries to major token groups.
                    # Remaining queries are unrestricted (learned mixtures).
                    n_real = int(is_real_window.sum().item())
                    types_real = types_flat[is_real_window]  # (n_real, L)
                    qmask = torch.ones((n_real, self.summary_num_queries, L), device=x.device, dtype=torch.bool)

                    # q0: structural
                    qmask[:, 0, :] = types_real == int(TokenCategory.STRUCTURAL)
                    # q1: diag/proc/med
                    if self.summary_num_queries > 1:
                        qmask[:, 1, :] = (types_real == int(TokenCategory.DIAGNOSIS)) | (types_real == int(TokenCategory.PROCEDURE)) | (
                            types_real == int(TokenCategory.MEDICATION)
                        )
                    # q2: measurements
                    if self.summary_num_queries > 2:
                        qmask[:, 2, :] = types_real == int(TokenCategory.MEASUREMENT)

                    query_mask = qmask

                summaries_multi = self.summary_pooler(
                    x_flat[is_real_window],
                    pool_mask[is_real_window],
                    query_mask=query_mask,
                )  # (n_real, Q, D)
                combined = self.summary_combine(
                    summaries_multi.reshape(summaries_multi.shape[0], self.summary_num_queries * D)
                )
                window_summaries_flat[is_real_window] = combined.to(dtype=window_summaries_flat.dtype)

        if self.summary_fuse_terminal and self.summary_terminal_fuser is not None and is_real_window.any():
            lengths = mask_bool.to(dtype=torch.long).sum(dim=-1).clamp(min=1)
            term_idx = (lengths - 1).clamp(min=0, max=max(0, L - 1))
            row_idx = torch.nonzero(is_real_window, as_tuple=False).squeeze(-1)
            terminal = x_flat[row_idx, term_idx[row_idx], :]  # (N_real, D)
            pooled = window_summaries_flat[row_idx]  # (N_real, D)
            alpha = torch.sigmoid(self.summary_terminal_fuser(torch.cat([pooled, terminal], dim=-1)))  # (N_real,1)
            window_summaries_flat[row_idx] = alpha * terminal + (1.0 - alpha) * pooled

        if self.enable_window_meta and self.window_meta_proj is not None and self.window_meta_scale is not None:
            # Window-level features to help global transition modeling (and biasing).
            content_mask = mask_bool
            if token_type_ids is not None:
                types_flat_all = token_type_ids.view(B * W * C, L)
                content_mask = content_mask & (types_flat_all != self.special_type_id)
            else:
                types_flat_all = None

            n_tokens = content_mask.to(dtype=torch.float32).sum(dim=-1)  # (B*W,)
            neg_inf = torch.tensor(float("-inf"), device=times_flat.device, dtype=times_flat.dtype)
            t_masked = torch.where(content_mask, times_flat, neg_inf)
            duration_h = t_masked.max(dim=-1).values
            duration_h = torch.where(torch.isfinite(duration_h), duration_h, torch.zeros_like(duration_h))
            duration_h = duration_h.clamp(min=0.0)
            density = n_tokens / duration_h.clamp(min=0.25)

            if types_flat_all is None:
                n_struct = torch.zeros_like(n_tokens)
                n_med = torch.zeros_like(n_tokens)
                n_meas = torch.zeros_like(n_tokens)
            else:
                n_struct = (content_mask & (types_flat_all == int(TokenCategory.STRUCTURAL))).to(dtype=torch.float32).sum(dim=-1)
                n_med = (
                    content_mask
                    & (
                        (types_flat_all == int(TokenCategory.DIAGNOSIS))
                        | (types_flat_all == int(TokenCategory.PROCEDURE))
                        | (types_flat_all == int(TokenCategory.MEDICATION))
                    )
                ).to(dtype=torch.float32).sum(dim=-1)
                n_meas = (content_mask & (types_flat_all == int(TokenCategory.MEASUREMENT))).to(dtype=torch.float32).sum(dim=-1)

            denom = n_tokens.clamp(min=1.0)
            frac_struct = n_struct / denom
            frac_med = n_med / denom
            frac_meas = n_meas / denom

            meta = torch.stack(
                [torch.log1p(n_tokens), torch.log1p(duration_h), torch.log1p(density), frac_struct, frac_med, frac_meas],
                dim=-1,
            )
            meta_emb = self.window_meta_proj(meta) * self.window_meta_scale
            meta_emb = meta_emb * is_real_window.to(dtype=meta_emb.dtype).unsqueeze(-1)
            window_summaries_flat = window_summaries_flat + meta_emb

        window_summaries = window_summaries_flat.view(B, W, C, D)

        return x_out, window_summaries
