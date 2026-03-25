# Project Tickets V1

This board captures the current end-to-end review state and the next concrete work items for tokenization, windowing, and transformer development.

## Status Snapshot (2026-03-25)

### Completed

- `ARCH-1` Unified generative path
- `ARCH-2` Generation and rollout contract
- `ARCH-4` Autoregressive auxiliary supervision alignment
- `ARCH-5` Numeric side-channel masking
- `TOK-3` Demographic specials policy
- `WIN-1` Causal window contract audit
- `WIN-2` Transfer subtype refinement
- `WIN-3` Window-marker semantics
- `TRAIN-1` Resume-safe trainer
- `EVAL-1A` Rollout implementation
- `EVAL-4` Family-stratified next-token evaluation

### In Progress

- `TOK-1` Medication and procedure semantic coverage
  - Current state: freeze thresholds pass and held-out coverage is acceptable for v1, but unresolved surfaces and held-out drift reporting are not yet systematized.
- `TOK-2` Observation exact-vocab tail policy
  - Current state: exact OBS coverage is strong and hash fallback is gone, but tail reporting and explicit guardrails still need to be tightened.
- `TRAIN-3` Loss balancing review
  - Current state: family-stratified metrics showed severe semantic under-learning relative to measurement/structural lanes; `semantic_boost_v1` materially improved diagnosis/procedure/medication families but was too aggressive on easy lanes. A softer `semantic_boost_v2` preset is now available and still needs a short validation run.
- `TRAIN-2` Throughput and data loading
  - Current state: precompiled-timeline support now has an index/manifest path, a compile CLI, and trainer dataloader wiring with auto worker/prefetch settings. It still needs LRZ validation and throughput benchmarking on a real compiled subset before being treated as the default long-run path.
- `EVAL-1B` End-to-end generation audit
  - Current state: rollout exists and can be audited per subject. The first real rollout already showed a plausible causal transition but overconfident structural generation (`VASO_OFF`) relative to semantic content. Multi-subject audit coverage is still missing.

### Not Started / Still Missing

- `ARCH-3` Hierarchy ablations
- `TOK-4` Production no-hash guardrails
- `EVAL-2` Clinical probe set
- `EVAL-3` Representation sanity checks

### Current Priority Order

1. `TRAIN-2`: compile a real precompiled subset on LRZ, confirm the new index/manifest path, and benchmark throughput versus on-the-fly loading.
2. `TRAIN-3`: validate `semantic_boost_v2` with a short continuation and `EVAL-4`, then decide the long-run recipe.
3. `EVAL-1B`: expand rollout audit from single-subject examples to a small multi-subject review set after the softer weighting pass.
4. `TOK-4`: harden production no-hash guardrails once the long-run input pipeline and weighting recipe are stable.
5. `ARCH-3` and `EVAL-3`: justify the hierarchy with ablations after the training recipe stops moving.

## Architecture

- **ARCH-1 Unified generative path**
  - Goal: keep unified dense-token next-token prediction as the default transformer pretraining path, with auxiliary boundary/window heads only.
  - Why it matters: the compact runtime vocab is small enough that a single calibrated token head is simpler and more stable than routed heads.
  - Suggested files: `src/ehr_hier/transformer/model.py`, `src/ehr_hier/transformer/heads.py`, `src/ehr_hier/transformer/loss.py`, `scripts/train_transformer_v1.py`.
  - Acceptance criteria: next-token loss remains the primary objective; aux heads remain optional; resume training and checkpointing work cleanly; no same-position reconstruction path remains in the default trainer.
  - Dependencies: tokenization freeze, runtime vocab bundle.

- **ARCH-2 Generation and rollout contract**
  - Goal: define how inference rolls out chunk/window sequences using the current marker grammar.
  - Why it matters: the model must generate valid care trajectories, not just minimize training loss.
  - Suggested files: `src/ehr_hier/transformer/generation.py`, `src/ehr_hier/transformer/model.py`, `src/ehr_hier/transformer/collator.py`.
  - Acceptance criteria: a real rollout path exists rather than grammar-only legality helpers; generation obeys `WIN_*`, `WIN_END`, and `WIN_CONTINUE`; window transitions remain causal; rollout summaries can be inspected per subject.
  - Dependencies: ARCH-1, windowing freeze.

- **ARCH-3 Hierarchy ablations**
  - Goal: test whether local chunk encoder, window aggregator, and global aggregator each add value.
  - Why it matters: the current hierarchy is plausible, but should be justified by measurable improvement.
  - Suggested files: `src/ehr_hier/transformer/encoder.py`, `src/ehr_hier/transformer/aggregator.py`, `src/ehr_hier/transformer/embeddings.py`, `scripts/train_transformer_v1.py`.
  - Acceptance criteria: explicit bypass flags exist for local-only and reduced-global runs; zero-layer settings do not silently change behavior; losses and next-window accuracy are compared.
  - Dependencies: ARCH-1.

- **ARCH-4 Autoregressive auxiliary supervision alignment**
  - Goal: align transition and next-window auxiliary supervision with the actual autoregressive marker contract instead of trivial same-position marker reconstruction.
  - Why it matters: current boundary accuracy is near-perfect and not very informative; auxiliary heads should measure something real about future structure.
  - Suggested files: `src/ehr_hier/transformer/loss.py`, `src/ehr_hier/transformer/model.py`, `src/ehr_hier/transformer/collator.py`, `tests/test_model_*`, `tests/test_loss_*`.
  - Acceptance criteria: boundary and next-window supervision are defined on genuinely predictive positions; held-out auxiliary accuracy is interpretable; default end-token mode does not orphan the boundary-next-window-type head.
  - Dependencies: ARCH-1, WIN-3.

- **ARCH-5 Numeric side-channel masking**
  - Goal: ensure numeric-value features are only injected where a token actually carries numeric meaning.
  - Why it matters: unconditional numeric branches can leak constant offsets or family identity in a way that is hard to reason about.
  - Suggested files: `src/ehr_hier/transformer/embeddings.py`, `src/ehr_hier/transformer/model.py`, `src/ehr_hier/transformer/loss.py`.
  - Acceptance criteria: numeric side-channel application is explicitly masked; non-numeric tokens do not receive spurious value projections; tests cover numeric and non-numeric token families.
  - Dependencies: ARCH-1.

## Tokenization/Semantics

- **TOK-1 Medication and procedure semantic coverage**
  - Goal: monitor whether MedTok explicit coverage and exact residual fallback remain acceptable for diagnosis, procedure, and medication.
  - Why it matters: semantic abstraction quality is still uneven, especially for medication and long-tail procedure codes.
  - Suggested files: `scripts/audit_tokenization_flow.py`, `scripts/build_compressed_medtok_vocabs.py`, `src/ehr_hier/tokenizers/medtok_attr_encoder.py`, `src/ehr_hier/tokenizers/medtok_crosswalk.py`.
  - Acceptance criteria: coverage reports remain reproducible; residual hash stays zero; medication and procedure mapped rates remain above current freeze thresholds; held-out reports include explicit vs residual vs UNK counts and top unresolved raw surfaces by family.
  - Dependencies: tokenization freeze, current MedTok crosswalk artifacts.

- **TOK-2 Observation exact-vocab tail policy**
  - Goal: keep OBS as factorized code/value tokens, preserve exact coverage for the critical mass, and audit the remaining drop tail.
  - Why it matters: OBS is now exact for most events, but the tail still needs a durable policy and visible reporting.
  - Suggested files: `scripts/build_observation_vocabs.py`, `src/ehr_hier/data/observation_vocab.py`, `src/ehr_hier/data/subject_timeline_builder.py`, `src/ehr_hier/tokenizers/vocab_contract.py`.
  - Acceptance criteria: OBS exact coverage stays near current levels; dropped tail is explicit and auditable; no hash fallback is reintroduced by default; top uncovered code/value pairs and measurement-like `OTHER` aliases are reported before any bundle-shape change.
  - Dependencies: current OBS vocab artifacts.

- **TOK-3 Demographic specials policy**
  - Goal: keep admission-anchored demographic specials minimal and defensible.
  - Why it matters: global specials must be stable, auditable, and avoid speculative extraction rules.
  - Suggested files: `src/ehr_hier/data/subject_timeline_builder.py`, `SCHEMA_TOKENS.md`, `src/ehr_hier/tokenizers/vocab_contract.py`.
  - Acceptance criteria: sex/age/BMI/height/weight remain explicit; race stays out unless a stable extraction contract is added; special-family ids remain preserved full.
  - Dependencies: current timeline builder and sparse contract.

- **TOK-4 Production no-hash guardrails**
  - Goal: enforce that semantic and observation tails fail loudly or are explicitly dropped rather than silently falling back to hash-like behavior in production.
  - Why it matters: the current contract depends on exact residuals and explicit OBS vocab artifacts; drift back to implicit hashing would undermine the freeze guarantees.
  - Suggested files: `src/ehr_hier/tokenizers/vocab_contract.py`, `src/ehr_hier/data/subject_timeline_builder.py`, `scripts/audit_tokenization_flow.py`, `SCHEMA_TOKENS.md`.
  - Acceptance criteria: missing exact-vocab artifacts or unexpected fallback modes are surfaced in audits/tests; OBS hash paths remain disabled by default; tokenization is documented as frozen with guardrails.
  - Dependencies: TOK-1, TOK-2.

## Structural/Windowing

- **WIN-1 Causal window contract audit**
  - Goal: keep the current `PROLOGUE`, `ED`, `INPATIENT`, `ICU`, `OR`, `POST_DISCHARGE` contract frozen and auditable.
  - Why it matters: the windowing grammar is part of the model input contract and should not drift casually.
  - Suggested files: `src/ehr_hier/data/window_segmentation.py`, `scripts/audit_window_contract_v1.py`, `configs/data/structural_codes.yaml`, `STRUCTURAL_CONTRACT_V1.md`.
  - Acceptance criteria: `window_type_unk_frac` stays low; `TRANSFER_TO` remains authoritative when present; `POST_DISCHARGE` remains causal and clinically populated.
  - Dependencies: current structural codebook and tokenization freeze.

- **WIN-2 Transfer subtype refinement**
  - Goal: maintain conservative opener typing while preserving safe ICU/OR refinements.
  - Why it matters: ICU appears conservative; PACU and unit suffixes may improve subtyping, but only if they are causally defensible.
  - Suggested files: `src/ehr_hier/data/structural_codes.py`, `configs/data/structural_codes.yaml`, `scripts/audit_transfer_to_suffixes.py`.
  - Acceptance criteria: ICU/OR alias coverage is documented; no extra window types are introduced unless opener evidence is reliable.
  - Dependencies: WIN-1.

- **WIN-3 Window-marker semantics**
  - Goal: ensure collator-inserted marker tokens are the sole mechanism for window continuation/end signaling.
  - Why it matters: this keeps the model-input grammar aligned with generation and avoids hidden boundary logic.
  - Suggested files: `src/ehr_hier/transformer/collator.py`, `src/ehr_hier/transformer/generation.py`, `src/ehr_hier/transformer/loss.py`.
  - Acceptance criteria: marker tokens are inserted deterministically; no boundary morphing happens after timeline building; training and generation share the same grammar assumptions; default end-token mode is explicitly documented.
  - Dependencies: WIN-1.

## Training/Optimization

- **TRAIN-1 Resume-safe trainer**
  - Goal: keep `scripts/train_transformer_v1.py` resume-safe and walltime-safe.
  - Why it matters: the first long runs exposed resume semantics and finalization bugs.
  - Suggested files: `scripts/train_transformer_v1.py`, `tests/test_train_transformer_v1.py`.
  - Acceptance criteria: resume from checkpoint continues for the requested number of epochs; `epoch_end` is written; final checkpoints are saved on max-step stop.
  - Dependencies: ARCH-1.

- **TRAIN-2 Throughput and data loading**
  - Goal: reduce walltime overhead from on-the-fly timeline building and single-worker loading.
  - Why it matters: current throughput is good for debugging but too slow for full-cohort training.
  - Suggested files: `scripts/train_transformer_v1.py`, `src/ehr_hier/data/compile_dataset.py`.
  - Acceptance criteria: precompiled timelines become the default long-run path; a manifest/index exists for precompiled shards; benchmarked throughput improves materially over on-the-fly `num_workers=0` loading without changing the contract.
  - Dependencies: windowing freeze, trainer baseline.

- **TRAIN-3 Loss balancing review**
  - Goal: decide whether auxiliary losses should be downweighted or kept fixed now that boundary loss is nearly saturated.
  - Why it matters: the token loss is learning well; auxiliaries should not dominate future runs.
  - Suggested files: `src/ehr_hier/transformer/loss.py`, `scripts/train_transformer_v1.py`.
  - Acceptance criteria: token loss remains primary; boundary/next-window losses are reviewed on a held-out run; value loss remains disabled unless explicitly needed.
  - Dependencies: TRAIN-1, ARCH-4.

## Evaluation/Clinical Goals

- **EVAL-1A Rollout implementation**
  - Goal: implement a concrete inference-time rollout path from `best.pt` using the current marker grammar and compact runtime vocab.
  - Why it matters: token-level metrics are necessary but not sufficient for trajectory modeling.
  - Suggested files: `src/ehr_hier/transformer/generation.py`, `scripts/smoke_transformer_pipeline.py`, `scripts/train_transformer_v1.py`.
  - Acceptance criteria: rollout can step token-by-token across chunks and windows; marker legality is enforced; outputs are serializable for review.
  - Dependencies: ARCH-2, WIN-3.

- **EVAL-1B End-to-end generation audit**
  - Goal: inspect generated trajectories from `best.pt` and verify grammatical and clinical realism.
  - Why it matters: token-level metrics are necessary but not sufficient for trajectory modeling.
  - Suggested files: `src/ehr_hier/transformer/generation.py`, `scripts/smoke_transformer_pipeline.py`, `scripts/train_transformer_v1.py`.
  - Acceptance criteria: generated sequences respect window grammar, marker placement, and causal ordering; obvious illegal transitions are rare; rollout artifacts can be audited per subject.
  - Dependencies: EVAL-1A.

- **EVAL-2 Clinical probe set**
  - Goal: define and run downstream probes for clinically important tasks.
  - Why it matters: the model should be judged by trajectory utility, not only pretraining loss.
  - Suggested files: `src/ehr_hier/transformer/model.py`, `src/ehr_hier/transformer/heads.py`, `src/ehr_hier/transformer/loss.py`.
  - Acceptance criteria: at least a small probe suite exists for mortality, readmission, ICU transfer, and length-of-stay/discharge horizon.
  - Dependencies: ARCH-1, TRAIN-1.

- **EVAL-3 Representation sanity checks**
  - Goal: inspect whether the hierarchy learns meaningful states and whether auxiliary signals are informative.
  - Why it matters: the architecture should improve sample efficiency, not just memorization.
  - Suggested files: `scripts/train_transformer_v1.py`, `src/ehr_hier/transformer/encoder.py`, `src/ehr_hier/transformer/aggregator.py`.
  - Acceptance criteria: layer-wise or pooled state inspection is possible; token accuracy and next-window accuracy improve over smoke runs; ablations are comparable; transition auxiliaries are interpreted only after ARCH-4 lands.
  - Dependencies: TRAIN-1, ARCH-3, ARCH-4.

- **EVAL-4 Family-stratified next-token evaluation**
  - Goal: break next-token performance out by token family and semantic stage instead of relying on a single aggregate accuracy.
  - Why it matters: current overall metrics look good, but semantic quality is uneven across explicit MedTok, residual, structural, and observation families.
  - Suggested files: `scripts/train_transformer_v1.py`, `src/ehr_hier/transformer/loss.py`, `src/ehr_hier/tokenizers/vocab_contract.py`, `SCHEMA_TOKENS.md`.
  - Acceptance criteria: evaluation reports next-token loss and accuracy for diagnosis/procedure/medication explicit tokens, residual tokens, UNKs, measurement code/value, observation code/value, structural, and special markers.
  - Dependencies: ARCH-1, TOK-1, TOK-2.
