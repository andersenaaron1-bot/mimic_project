# Repository Guidelines

## Project charter (what we're building)
An EHR foundation model that learns patient trajectories from heterogeneous clinical tokens.

Core ideas:
- **Measurements**: numeric measurement events are encoded via a conditional VAE (cVAE) into a latent `z`,
  then discretized with a CODA-inspired residual VQ (RVQ) that uses an attention shortlist per level.
  This yields compact value tokens that can condition downstream dynamics.
- **Clinical semantics (Dx/Proc/Meds)**: diagnoses, procedures, and medications are mapped through MedTok
  vocabularies (ontology/graph-informed codes) with robust canonicalization and UNK fallbacks.
- **Structural signifiers (central contribution)**: explicit tokens that represent patient movement and
  care context (admission, care unit changes, OR phases, discharge, etc.), plus overlay signifiers for
  critical sub-episodes (MV, shock, RRT, sepsis bundle, CPR, etc.).
- **Windows of care**: timelines are segmented into windows using a structural codebook YAML. The model
  is trained to predict **window marker tokens** (end-of-window and optionally next-window type). A global
  trajectory model can bias these marker logits based on window progress/density and prior window history.
- **Time-awareness**: continuous-time RoPE (cRoPE) in attention with an optional ALiBi-hours bias, plus an
  optional additive time embedding. Time is handled both within windows and across windows.

Key contracts (keep these consistent as the dataset evolves):
- `SCHEMA_TOKENS.md` (token families + vocabulary routing)
- `TIME_MODEL.md` (time tensors + how attention uses them)
- `configs/data/structural_codes.yaml` (window boundaries + overlay signifiers)
- `TRANSFORMER_STAGE_HANDOFF.md` (current LRZ state, active artifact paths, cVAE findings, and next-step priorities)

## Project structure & module organization
- `src/ehr_hier/data/`: MEDS routing, demographics helpers, structural codebook loader, and
  `build_subject_timeline()` which emits `EventToken` bundles.
- `src/ehr_hier/tokenizers/`: measurement encoder (cVAE -> RVQ), MedTok encoders, and simple fallbacks.
- `src/ehr_hier/models/value_encoders/`: cVAE + RVQ + attention shortlist (CODA-inspired).
- `src/ehr_hier/transformer/`: AET collator (windowing + markers), local encoder, global aggregator,
  embeddings (cRoPE + time), switched heads, and loss/routing.
- `configs/`: Hydra configs (some are legacy; prefer the AET codepath under `src/ehr_hier/transformer/`).
- `artifacts/`: MedTok files and `vocab_manifest.json` (offsets for token families).
- `tests/`: pytest suite (unit + optional integration tests).

## Build, test, and development commands
- Install deps: `python -m venv .venv; .venv\\Scripts\\activate; pip install -r requirements.txt`
- Run tests: `pytest -q`

Value tokenization artifacts (examples; paths depend on your environment):
- Compute measurement mapping/stats:
  `python scripts/compute_meas_stats.py --meds_reader_db <db> --splits_parquet <splits.parquet> --split train --code2id_pt data/code2id.pt --stats_pt data/stats.pt`
- Train value cVAE:
  `python scripts/train_value_cvae.py --meds_reader_db <db> --code2id_pt data/code2id.pt --splits_parquet <splits.parquet> --out cvae_ckpt.pt`
- Train value tokenizer (RVQ):
  `python scripts/train_value_tokenizer.py --meds_reader_db <db> --code2id_pt data/code2id.pt --cvae_ckpt cvae_ckpt.pt --splits_parquet <splits.parquet> --out value_tokenizer.pt`

Timeline compilation (optional fast-path):
- Use `src/ehr_hier/data/compile_dataset.py` as a library entrypoint to materialize timelines.

## Current LRZ state
- Treat this section as the canonical handoff snapshot for the current LRZ work. Update it whenever the active dataset, DB, or training artifacts move.
- Active DSS base:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2`
- Successful MEDS cohort root:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort`
- Active meds_reader DB:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/meds_reader_db_mimiciv_20260226_033711/mimiciv.db`
- Active measurement artifacts dir:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/pipeline_artifacts_20260226_033711`
- Active runtime deps overlay:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/containers/runtime_pydeps`
- Current retained DSS usage after cleanup: about `22G`.
- Practical DSS quota assumption during this stage: about `40G`. Avoid large duplicate ETL runs or repeated fat image imports.
- `meds_reader_convert` succeeded. The DB is a directory tree, not a single `.db` file; do not search for it with `find -type f -name mimiciv.db`.
- `compute_meas_stats.py` succeeded on the train split and produced:
  - `code2id.pt`
  - `stats.pt`
- Current measurement mapping size:
  - `n_vars_ckpt = 2344`
  - train-kept measurement codes = `2343` plus reserved `0`
- `train_value_cvae.py` produced a valid checkpoint:
  - `cvae_ckpt.pt`
  - config: `z_dim=64`, `hidden=128`, `var_emb_dim=64`, `beta_kl=0.1`
- `scripts/eval_value_cvae.py` was added to evaluate overall and per-variable CVAE behavior on `tuning` and `held_out`.

## LRZ operational notes
- Prefer the lightweight pipeline image for CPU-side inspection/evaluation:
  `docker://ghcr.io#andersenaaron1-bot/ehr-pipeline-cpu:0.3.1-pipeline-cpu-min`
- The NGC PyTorch image import on LRZ (`docker://nvcr.io#nvidia/pytorch:24.10-py3`) can OOM during pyxis/enroot squashfs creation before Python starts. This is an import-memory issue, not a file-quota issue.
- For lightweight evaluation and inspection, use the mounted repo plus `PYTHONPATH=/deps` rather than forcing the NGC image.
- `ehr-pipeline-cpu:0.3.1-pipeline-cpu-min` may still need the mounted `/deps` overlay for `meds_reader`.
- `df -h` on `/dss/dssfs04` reports filesystem capacity, not per-user/project quota. When LRZ reports `Disk quota exceeded`, assume DSS project quota pressure first.
- Long LRZ jobs should be run with `tmux`, `sbatch`, or explicit logging. SSH disconnects are common and not evidence of training failure.

## Measurement/CVAE status
- The current cVAE is numeric-only and should remain the measurement path. It is not a generic replacement for MedTok or structural tokenization.
- Loss interpretation:
  - current objective is Gaussian NLL on standardized values plus `0.1 * KL`
  - negative losses are expected
  - practical floor is near `-5`
  - observed training losses around `-4.7` are already close to the floor for this configuration
- `worst_by_mae` from `eval_value_cvae.py` is dominated by raw scale and is not the main fitness criterion.
- `worst_by_nll_z` is the main signal for variable-level mismatch because it is measured in standardized space.
- Current evaluation takeaway:
  - bulk lab measurement modeling looks acceptable
  - the main problem cases are high-`nll_z` flow/rate/infusion/output style variables and several `UNK`-coded measurements
  - many of the most clinically structural or transitory variables should likely be routed out of the measurement cVAE path anyway
- Implication for the next stage:
  - keep the cVAE path for dense numeric measurement variables
  - expect aggressive pruning or rerouting of sparse, bursty, operational, and structural/transitory numeric codes before final token modality assignment

## Transformer-stage priorities
- The next stage is not "make every event numeric." The next stage is to freeze modality boundaries cleanly:
  - measurement cVAE/RVQ path for dense numeric measurement variables
  - MedTok path for diagnoses/procedures/medications where ontology-aware canonicalization matters
  - structural/signifier path for transitions, unit moves, episode boundaries, overlays, and care-window markers
- The structural/signifier design is the central modeling contribution. Favor explicit, auditable boundary and overlay tokens over overloading the cVAE with transitory operational codes.
- Before changing the transformer, first decide which codes remain in each tokenization family and document the rationale.
- After modality routing is stable, use the remaining high-frequency uncaptured codes to decide whether any additional token families are warranted.

## Fresh-conversation handoff
- Starting a new conversation is optional, not required. If context is getting noisy, use a new thread and begin with:
  - "Use `AGENTS.md` and `TRANSFORMER_STAGE_HANDOFF.md` as the current handoff. The active LRZ MEDS cohort, meds_reader DB, measurement artifacts, and CVAE findings are already fixed there. I want to continue with final token-modality assignment and transformer-stage structural/window design."
- In a new thread, the first files to open should be:
  - `AGENTS.md`
  - `TRANSFORMER_STAGE_HANDOFF.md`
  - `SCHEMA_TOKENS.md`
  - `configs/data/structural_codes.yaml`
  - `src/ehr_hier/data/subject_timeline_builder.py`
  - `src/ehr_hier/transformer/collator.py`

## Coding style & naming conventions
- Python 3.11+, 4-space indents, type hints on public functions, f-strings for logging.
- Keep paths config-driven; do not hardcode dataset locations.
- Do not commit large artifacts (`*.pt`, large `*.parquet`) unless explicitly intended.

## Roles & prompt snippets

### Architect agent
Mission: keep interfaces stable across tokenizers, windowing, and the AET model.

Start-here files:
- `SCHEMA_TOKENS.md`
- `TIME_MODEL.md`
- `configs/data/structural_codes.yaml`
- `src/ehr_hier/data/subject_timeline_builder.py`
- `src/ehr_hier/transformer/collator.py`
- `src/ehr_hier/transformer/model.py`

Definition of done:
- `pytest -q` passes; schema/time docs stay consistent with code.

Prompt snippets:
- "Make window signifiers first-class: freeze label ids and add a manifest-driven window type map."
- "Add a generation-time routine that rolls out windows by predicting WIN_END/WIN_<TYPE> markers."

### Tokenization engineer
Mission: robust per-category encoders and vocab manifests that survive dataset changes.

Start-here files:
- `src/ehr_hier/tokenizers/base_encoder.py`
- `src/ehr_hier/tokenizers/measurement_encoder.py`
- `src/ehr_hier/tokenizers/medtok_*`
- `artifacts/vocab_manifest.json`
- `SCHEMA_TOKENS.md`

Definition of done:
- unknown codes map to UNK (or are explicitly dropped with a documented reason)
- measurement bundles preserve ordering (MEAS_CODE followed by RVQ codes with dt=0)
- manifests/offsets updated in lockstep with encoders

Prompt snippets:
- "Wire MedTok embeddings into the model embedding table (with a projector) and keep UNK stable."
- "Add a single manifest that includes offsets + sizes + hashes and can derive AET routing."

### Data/ETL agent
Mission: deterministic window boundaries and event access (fast + reproducible).

Start-here files:
- `src/ehr_hier/data/event_router.py`
- `src/ehr_hier/data/structural_codes.py`
- `configs/data/structural_codes.yaml`
- `src/ehr_hier/data/compile_dataset.py`

Definition of done:
- segmentation is deterministic; boundary/overlay labels are auditable
- coverage tests can run against a meds_reader DB (`tests/test_*_db_*.py`)

Prompt snippets:
- "Derive structural boundary seeds from ICU stays/transfers and validate the distribution."
- "Implement an audit that counts boundary/overlay hits and top missing codes."

### Modeling agent
Mission: hierarchical transformer (local windows + global trajectory) with time and transition modeling.

Start-here files:
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/encoder.py`
- `src/ehr_hier/transformer/aggregator.py`
- `src/ehr_hier/transformer/embeddings.py`
- `TIME_MODEL.md`

Definition of done:
- tensor shapes covered by tests; ablation flags exist for time/transition features

Prompt snippets:
- "Add a routed embedding layer so sparse global ids from the manifest don't require a giant nn.Embedding."
- "Expose and test a clear separation between intra-window time and inter-window time in attention."

### Evaluation agent
Mission: keep tests and small fixtures aligned with the schema and windowing behavior.

Start-here files:
- `tests/test_collator_*`
- `tests/test_model_*`
- `tests/test_subject_timeline_builder.py`

Definition of done:
- `pytest -q` clean; tests assert deterministic window markers and transition bias behavior

Prompt snippets:
- "Add a synthetic timeline fixture that exercises boundary + overlay + soft signifiers."
- "Test that structural boundaries become window hooks and that window type ids are inferred as expected."

## Testing guidelines
- Framework: pytest. Run from repo root with `pytest -q` (set `PYTHONPATH=.` if needed).
- Prefer small synthetic fixtures; DB-backed tests are marked `@pytest.mark.integration` and require env vars.

## Commit & PR guidelines
- Keep changes focused; document any schema/vocab changes and provide migration notes.
- Flag breaking changes (token id layout, marker semantics, checkpoint formats).
