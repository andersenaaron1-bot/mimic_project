# Transformer V1 Tracker

Status legend:
- `todo`
- `in_progress`
- `blocked`
- `done`

Last updated: 2026-03-09
Owner: `transformer-v1` branch

## Locked V1 Decisions

1. Tokenization baseline is **v1 frozen contract** (`configs/data/tokenization_v1.yaml` + runtime remapper).
2. Hierarchy is **token -> chunk -> semantic window -> global trajectory**.
3. Intra-window chunking is **mechanical** (bounded context budget), not a semantic regime change.
4. Semantic transitions are represented via **window marker grammar** (`WIN_CONTINUE`, `WIN_END`, `WIN_<NEXT_TYPE>`).
5. Transition supervision must be **boundary-position specific**, not globally broadcast to all tokens.

## Transition Policy Clarification (Current)

- `TRANSFER_TO` is a strong inter-window anchor and should remain a primary boundary seed.
- It is **not sufficient alone** for all clinically meaningful regime changes (for example, transitions driven by procedure or support initiation without transfer).
- Current segmentation already supports richer boundary bundles via:
  - explicit transition action/type metadata (`transition_action_id`, `transition_window_type_id`)
  - fallback `window_hook` boundaries.
- Chunk splits (`max_len`, `max_chunks_per_window`) are local mechanics only and should not be interpreted as semantic transitions.

## Inter-Window Signaling Rule (V1 Locked)

- Primary semantic boundary signal: `TRANSFER_TO` (+ admission/discharge style structural boundaries).
- Secondary semantic boundary signal: selected high-confidence non-transfer transition hooks (used as fallback only).
- Never emit semantic boundary markers from chunk overflow alone.
- `WIN_END` / `WIN_<NEXT_TYPE>` supervision is applied only at true semantic boundaries, not at mechanical chunk cuts.

## LRZ Runtime Profiles (V1 Standard)

Use these profiles to keep iteration practical and avoid VRAM blowups from full-head logits.

### Profile D0: Functional Smoke (fast fail)
- purpose: shape/routing/state-machine sanity
- target device: any GPU (incl. V100)
- recommended caps:
  - `max_subjects=1`
  - `max_windows=6`
  - `max_chunks_per_window=2`
  - `max_len_per_window=64`
  - `d_model=128`, `num_heads=4`, `d_ff=256`, `num_local_layers=1`, `num_global_layers=1`, `num_chunk_layers=1`
- expected: finite forward/backward, non-zero supervised boundary counts, no unrouted tokens

### Profile D1: Stability Smoke (default debug)
- purpose: finite-loss + grad stability with realistic chunk/window chains
- target device: A100-80/H100-94 preferred
- recommended caps:
  - `max_subjects=2`
  - `max_windows=8`
  - `max_chunks_per_window=2`
  - `max_len_per_window=64`
  - same model dims as D0
- expected: stable run in < 30 min; no OOM; reproducible logs

### Profile D2: Pre-Train Budget Check
- purpose: approximate training memory/runtime before long jobs
- target device: A100-80/H100-94
- recommended caps:
  - `max_subjects=4`
  - `max_windows=12`
  - `max_chunks_per_window=3`
  - `max_len_per_window=80`
  - `d_model=128` first; scale only after passing
- expected: pass without OOM twice in a row before launching long run

### Profile T1: 2-Day Feasible Baseline
- purpose: first long run that still allows iteration
- target device: H100-94 preferred, A100-80 acceptable
- policy:
  - keep per-step token budget conservative
  - scale throughput via grad accumulation, not giant local sequence caps
  - checkpoint frequently and resume-friendly
- acceptance gate before launch:
  - D1 and D2 both pass
  - finite metrics for at least two consecutive smoke runs
  - no unknown window leakage, no unrouted tokens

## Ticket Backlog

## P0: Architecture Contract + Stability

- `T0.1` status: `done`
  - step: Freeze generation grammar/state machine (`inside_chunk`, `chunk_end`, `window_end` states + legal tokens).
  - files: `SCHEMA_TOKENS.md`, `src/ehr_hier/transformer/collator.py`, new `src/ehr_hier/transformer/generation.py` (skeleton).
  - accept: State-machine unit tests pass; illegal-token rate = 0 on scripted rollout.

- `T0.2` status: `done`
  - step: Separate transition-control outputs from generic struct-token CE.
  - files: `src/ehr_hier/transformer/model.py`, `src/ehr_hier/transformer/loss.py`.
  - accept: `p_continue`, `p_end`, `p_next_type` losses computed only on valid boundary positions.

- `T0.3` status: `in_progress`
  - step: Numerical stability hardening (padded attention rows, non-finite numeric side-channel values).
  - files: `src/ehr_hier/transformer/encoder.py`, `src/ehr_hier/transformer/collator.py`, `tests/test_alibi_hours_bias.py`.
  - accept: smoke forward/backward finite loss and grad norm on LRZ GPU.

- `T0.4` status: `done`
  - step: Make telemetry strictly valid-mask aware (`id_remap_stats`, overflow counters, routed fractions).
  - files: `src/ehr_hier/transformer/collator.py`, `scripts/smoke_transformer_pipeline.py`.
  - accept: no padding-inflated counters in smoke JSON.

## P1: Local/Chunk/Window/Global Signal Design

- `T1.1` status: `todo`
  - step: Redesign summary composition:
    - chunk summary: pooled + terminal
    - window summary: fuse chunk features + window meta
    - global summary: causal semantic-window chain.
  - files: `src/ehr_hier/transformer/encoder.py`, `src/ehr_hier/transformer/aggregator.py`, `src/ehr_hier/transformer/model.py`.
  - accept: ablation shows improved next-window-type accuracy over last-chunk-only baseline.

- `T1.2` status: `todo`
  - step: Rework transition bias:
    - remove broad broadcast where inappropriate
    - apply local bias only on marker candidate positions.
  - files: `src/ehr_hier/transformer/model.py`.
  - accept: stable training; transition accuracy improves without CE degradation.

- `T1.3` status: `todo`
  - step: Wire chunk-to-chunk and window-to-window progression heads with explicit masks.
  - files: `src/ehr_hier/transformer/model.py`, `src/ehr_hier/transformer/loss.py`.
  - accept: non-zero supervised positions and finite auxiliary losses.

## P2: Time Modeling and Inference

- `T2.1` status: `todo`
  - step: Time ablation matrix:
    - cRoPE only
    - cRoPE + ALiBi local
    - cRoPE + ALiBi global
    - cRoPE + ALiBi both
    - additive time embedding toggle.
  - files: `src/ehr_hier/transformer/{model,encoder,aggregator}.py`, config presets.
  - accept: pick one v1 default by transition metrics + stability.

- `T2.2` status: `todo`
  - step: Implement constrained autoregressive rollout:
    - legal token masks by state
    - deterministic replay mode and stochastic sampling mode.
  - files: new `src/ehr_hier/transformer/generation.py`, tests.
  - accept: no illegal transitions; valid chunk/window chains generated.

- `T2.3` status: `todo`
  - step: Add decode path for human-interpretable timeline reconstruction.
  - files: `src/ehr_hier/tokenizers/decode_tokens.py`, generation integration.
  - accept: sampled sequence -> interpretable timeline report.

## P3: Training Procedure

- `T3.1` status: `todo`
  - step: Define curriculum:
    - Stage A token CE only
    - Stage B + next-window-type
    - Stage C + chunk/window control heads
    - Stage D + optional time NLL losses
    - Stage E enable transition priors.
  - files: trainer entrypoint/config (new transformer-stage trainer).
  - accept: finite metrics across stage boundaries.

- `T3.2` status: `todo`
  - step: Replace legacy `src/ehr_hier/train.py` mock pipeline with real v1 train entrypoint.
  - files: new `scripts/train_transformer_v1.py` (or equivalent), config.
  - accept: reproducible LRZ run with checkpoint + eval logs.

## P4: Evaluation Gates

- `T4.1` status: `todo`
  - step: Build transition/confusion diagnostics:
    - window type confusion
    - boundary timing calibration
    - marker token precision/recall.
  - files: new eval script(s), tests.
  - accept: report generated on representative LRZ subset.

- `T4.2` status: `todo`
  - step: End-to-end smoke gate checklist:
    - finite loss/grad
    - no unrouted tokens
    - no unknown window type leakage unless expected
    - legal generation transitions.
  - files: `scripts/smoke_transformer_pipeline.py`, generation tests.
  - accept: all gate checks green before large training.

## Progress Log

- 2026-03-09:
  - Added runtime vocab remapper + collator overflow telemetry.
  - Added transformer smoke pipeline script.
  - Added backward-compatible timeline builder call in smoke script.
  - Began NaN stabilization on attention and numeric side-channel paths.
- 2026-03-09 [uncommitted]:
  - Completed `T0.1`: added `src/ehr_hier/transformer/generation.py` state machine and tests (`tests/test_generation_grammar.py`).
  - Completed `T0.2`: added `logits_transition_boundary` and `logits_boundary_next_window_type` with boundary-only supervision in `AETLossModule`.
  - Completed `T0.4`: made remapper `block_hits` valid-mask aware (no padding inflation).
  - Extended smoke finite checks (`scripts/smoke_transformer_pipeline.py`) and model/loss nan guards; `T0.3` remains pending LRZ GPU verification.

## How To Update This Tracker

When a ticket changes status:
1. Update `status` (`todo|in_progress|blocked|done`).
2. Add one line under **Progress Log** with date + commit hash.
3. If blocked, add blocker cause directly in ticket line.
