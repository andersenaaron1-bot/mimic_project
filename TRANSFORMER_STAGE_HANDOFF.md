# Transformer Stage Handoff

This file is the current operational handoff for the post-ETL, post-measurement-bootstrap stage.
It is intentionally concrete and path-specific. Update it whenever the active LRZ artifacts move.

## Current objective
- Freeze the final division of tokenization modalities before deeper transformer work.
- Preserve the current successful MEDS/meds_reader state on LRZ.
- Use the measurement cVAE results as evidence for routing decisions, not as a reason to force all numeric-valued events through a single encoder.

## Active LRZ artifact paths
- DSS base:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2`
- Successful MEDS cohort:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort`
- meds_reader DB:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/meds_reader_db_mimiciv_20260226_033711/mimiciv.db`
- Measurement artifacts:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/pipeline_artifacts_20260226_033711`
- Runtime deps overlay:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/containers/runtime_pydeps`

## Verified LRZ status
- MEDS cohort integrity check passed at the shard/file-manifest level.
- `meds_reader_convert` succeeded and produced a usable `mimiciv.db` directory-backed SubjectDatabase.
- `compute_meas_stats.py` succeeded and produced:
  - `code2id.pt`
  - `stats.pt`
- `train_value_cvae.py` produced a valid `cvae_ckpt.pt`.
- The cVAE checkpoint loads and matches the measurement mapping:
  - `n_vars_ckpt = 2344`
  - measurement codes kept in the train mapping = `2343`
- `scripts/eval_value_cvae.py` exists to evaluate overall and per-variable tuning/held-out behavior.

## Image/runtime notes on LRZ
- Preferred lightweight image for inspection/evaluation:
  `docker://ghcr.io#andersenaaron1-bot/ehr-pipeline-cpu:0.3.1-pipeline-cpu-min`
- That image may still require:
  - mounting `runtime_pydeps` at `/deps`
  - `export PYTHONPATH=/deps${PYTHONPATH:+:$PYTHONPATH}`
- The NGC image:
  `docker://nvcr.io#nvidia/pytorch:24.10-py3`
  can OOM during pyxis import on LRZ before the user command starts. This is a container import-memory issue, not a Python or code bug.
- If a command fails during `Creating squashfs filesystem...` with `error code 137`, treat it as pyxis/enroot OOM.

## Storage/quota notes
- Current retained DSS usage after cleanup is about `22G`.
- Practical quota assumption during this phase is about `40G`.
- Biggest retained items:
  - MEDS cohort output: about `20G`
  - meds_reader DB: about `2.2G`
- Avoid:
  - duplicate MEDS reruns unless necessary
  - repeated large image imports
  - broad pip overlays that reinstall full scientific stacks into `/deps`

## Measurement path status
- The current measurement route is:
  1. build a stable train-split measurement mapping with `compute_meas_stats.py`
  2. train a value cVAE on numeric-only events
  3. later train the residual VQ tokenizer on cVAE latents
- The cVAE is explicitly numeric-only. It is not intended to replace MedTok or structural tokenization.
- Current cVAE config:
  - `z_dim=64`
  - `hidden=128`
  - `var_emb_dim=64`
  - `beta_kl=0.1`
  - conditions: measurement code embedding + `dt_prev` + age + sex

## Interpreting the cVAE results
- The cVAE objective is Gaussian NLL on standardized values plus `0.1 * KL`.
- Negative losses are expected. The practical floor is near `-5` because decoder `log_sigma` is clamped at `-5`.
- Training reaching roughly `-4.7` is already near the floor for this configuration.
- `eval_value_cvae.py` outputs two different failure views:
  - `worst_by_mae`: dominated by raw scale, useful but easy to over-interpret
  - `worst_by_nll_z`: the main signal for real model mismatch in standardized space

## Current cVAE takeaway
- Bulk measurement modeling is adequate enough to continue.
- The major cVAE problem cases are:
  - flow/rate variables
  - infusion start/end numeric quantities
  - fluid-output style codes
  - several `UNK`-coded measurements
- Many of these are not actually desirable "measurement tokens" for the final architecture.
- Several of the problematic high-`nll_z` variables are better conceptual fits for the structural/signifier pathway.

## Strategic implication for token routing
- Do not force all numeric-valued events into the measurement cVAE path.
- The likely final routing should be:
  - dense numeric physiological/lab measurement variables:
    cVAE -> RVQ path
  - diagnosis/procedure/medication semantics:
    MedTok / canonical code path
  - care transitions, episode markers, operational states, overlays, and other transitory signifiers:
    structural/signifier path
- Sparse, bursty, operational, or window-defining variables should be reviewed before being allowed into the measurement tokenizer.

## Immediate next-stage questions
- Which cVAE variables should remain in the measurement path?
- Which high-`nll_z` or operational variables should be blacklisted from the cVAE path and reclassified as structural/signifier tokens?
- Which recurring event families remain outside both MedTok and the cleaned measurement path?
- How should structural windows be seeded and labeled so they reflect meaningful care trajectories rather than arbitrary token density?

## Recommended start files for the next stage
- `SCHEMA_TOKENS.md`
- `configs/data/structural_codes.yaml`
- `src/ehr_hier/data/subject_timeline_builder.py`
- `src/ehr_hier/data/structural_codes.py`
- `src/ehr_hier/tokenizers/measurement_encoder.py`
- `src/ehr_hier/transformer/collator.py`
- `src/ehr_hier/transformer/model.py`

## Suggested opening prompt for a new conversation
Use `AGENTS.md` and `TRANSFORMER_STAGE_HANDOFF.md` as the current handoff. The active LRZ MEDS cohort, meds_reader DB, measurement artifacts, and cVAE findings are already fixed there. Continue with final token-modality assignment, especially separating numeric measurement tokens from structural/transitory signifiers, and prepare the transformer/window-design stage.
