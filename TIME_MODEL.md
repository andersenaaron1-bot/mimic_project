# Time Model (TIME_MODEL)

This repo now models time at three coupled scales:
- **Intra-chunk time**: irregular time within a bounded local attention chunk.
- **Intra-window time**: progress across local chunks inside one semantic care-regime window.
- **Inter-window time**: the macro timing between semantic windows in the global trajectory.

The intent is to preserve clinical semantics: the model reasons about *what happened*
within a care setting, and *how care settings evolve* over longer trajectories.

## 1) Time fields on `EventToken`
Timeline building (`src/ehr_hier/data/subject_timeline_builder.py`) computes:
- `t_from_start_hours`: hours since the start of the subject timeline (absolute for that subject).
- `dt_from_prev_hours`: hours since the previously emitted token (non-negative).

Notes:
- Windowing and attention use `t_from_start_hours` as the canonical timestamp.
- `dt_from_prev_hours` is currently retained for debugging and optional future features.
- Measurement tokenization also uses **per-variable** `dt_prev` internally for cVAE conditioning;
  this is separate from the transformer time model.

## 2) Collation-time tensors (what the model actually receives)
The hierarchical collator (`src/ehr_hier/transformer/collator.py`) produces:
- `time_ids[b,w,c,l]` (float): **chunk-relative** time in hours.
  - For a token at absolute time `t_abs`, `t_rel_chunk = max(0, t_abs - chunk_start_abs)`.
  - Summary/special tokens are forced to `0`.
- `chunk_start_offsets[b,w,c]` (float): hours from semantic-window start to chunk start.
- `window_start_times[b,w]` (float): **absolute** semantic-window start time in hours (macro clock).

This cleanly separates:
- local time inside a bounded chunk,
- semantic-window progress across chunks, and
- macro time across the patient trajectory.

## 3) How AET uses time

### 3.1 Continuous RoPE (cRoPE) on attention
`ContinuousRotaryPositionalEmbedding` rotates Q/K using continuous timestamps instead of
integer positions (see `src/ehr_hier/transformer/embeddings.py`).

- Local encoder: cRoPE is driven by `time_ids` (hours since chunk start).
- Intra-window chunk aggregator: cRoPE is driven by `chunk_start_offsets`.
- Global aggregator: cRoPE is driven by `window_start_times` (hours since timeline start).

### 3.2 Optional ALiBi-style bias in hours
`AETCausalAttention` can add an ALiBi-like recency bias based on *time separation* rather
than token index (see `src/ehr_hier/transformer/encoder.py`):

- Pairwise delta (causal): `dt_ij = clamp(t_i - t_j, min=0, max=alibi_hours_max)`
- Bias: `bias_h(i,j) = -softplus(slope_h) * log1p(dt_ij)`

This encourages attending to recent tokens in **time**, even if the sequence index is dense.

### 3.3 Optional additive time embedding
In addition to cRoPE, AET can add an explicit per-token time feature via `TimeEmbedding`
(see `src/ehr_hier/transformer/embeddings.py` and `src/ehr_hier/transformer/model.py`):

- Normalize: `t' = log1p(clamp(t_hours, 0, max_hours)) / log1p(max_hours)`
- Project: `MLP(1 -> d_model)` and add to token embeddings (with a learned global scale)

Why both?
- cRoPE changes attention *patterns*.
- additive time embeddings make time available to MLPs/heads even without attending.

## 4) Time-aware window transition modeling (signifiers)
AET optionally predicts expected semantic-window length (in tokens and hours) and uses it to bias
the logits of the **semantic transition markers** (`WIN_END` / `WIN_<TYPE>`) on the final chunk.

Separately, the local chunk path can predict chunk length priors and bias the
`WIN_CONTINUE` marker inside dense semantic windows.

Current levels:
- semantic-window priors:
  - `pred_window_len_tokens`, `pred_window_len_hours`
  - `pred_window_dur_mu`, `pred_window_dur_sigma`
- chunk-local priors:
  - `pred_chunk_len_tokens`, `pred_chunk_len_hours`

Both are trained on their own target objects: semantic windows vs bounded local chunks.
(see `src/ehr_hier/transformer/model.py`):

- Predict priors per window: `pred_window_len_tokens`, `pred_window_len_hours`
- Optionally, predict a **distribution** over window duration in hours:
  - `y = log1p(window_duration_hours) ~ Normal(mu, sigma)`
  - outputs: `pred_window_dur_mu`, `pred_window_dur_sigma` (trained via an NLL term in `AETLossModule`)
- Optionally, predict a **per-event** time-to-next-event distribution inside windows:
  - `y = log1p(dt_next_hours) ~ Normal(mu, sigma)`
  - outputs: `pred_dt_next_mu`, `pred_dt_next_sigma` (trained via an NLL term in `AETLossModule`)
- Compute progress per token:
  - token progress from cumulative content tokens
  - time progress from `time_ids`
- Convert progress to a "hazard" logit and add it to marker token logits

This is the mechanism that lets the global trajectory model influence the *local* decision
to end a window and/or choose the next window type.
