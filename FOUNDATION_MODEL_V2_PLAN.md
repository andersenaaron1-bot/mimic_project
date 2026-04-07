# Foundation Model V2 Plan

Status: proposed research and implementation roadmap
Last updated: 2026-04-07
Scope: post-v1 architecture and tokenization refactor built on the current repo state

## 1. Purpose

The current repo already has three strong assets:

1. explicit structural windowing and transition metadata
2. a hierarchical local/chunk/window/global transformer path
3. a novel numeric measurement codec based on cVAE plus RVQ

However, the current form is still closer to a strong v1 research baseline than to a clear state-of-the-art foundation model for long-horizon EHR trajectory modeling. The main gaps are:

- token bundles are still more explicit than fused event objects
- the global path summarizes windows but is not yet a persistent latent health state
- there is no exact episodic memory to preserve rare critical facts over long trajectories
- time modeling is present, but the primary objective is still next-token style rather than a full marked time-to-event formulation

This document defines the target v2 system and the concrete implementation order.

## 2. Target Research Thesis

The v2 model should be framed as a modality-aware semi-Markov EHR foundation model with:

- typed fused event representations
- continuous measurement payload codecs
- explicit structural regime transitions
- a persistent cross-admission latent health state
- a sparse exact memory reservoir for rare critical facts
- marked time-to-event pretraining as the primary objective

The paper claim should not be "one tokenizer for everything." The stronger and more defensible claim is:

"All events share one event algebra and one fused model interface, while different event families use different payload codecs because their information geometry differs."

## 3. What V2 Keeps vs Replaces

### Keep

- structural/signifier design and transition metadata
- semantic windows and chunked local attention
- cVAE plus RVQ measurement path for dense numeric observations
- MedTok where it already provides good semantic support
- runtime vocab compaction and family-aware routing discipline
- current synthetic and unit-test-heavy development style

### Replace or Refactor

- bundle-first modeling as the primary local interface
- purely token-level local representation for measurements
- global transformer summary chain as the only long-range state
- auxiliary time heads as the only time-structured objective
- weak fallback semantics for unsupported discrete codes

## 4. V2 Tokenization Design

## 4.1 Unified event algebra

Introduce a typed fused event representation for model-facing collation. Suggested conceptual schema:

- `family_id`: measurement_numeric, measurement_qual, diagnosis, procedure, medication, structural, demographic, other
- `concept_id`: canonical symbolic identity for the event
- `payload_kind`: none, rvq_latent, categorical_value, numeric_scalar, modifier_set
- `payload_ids`: discrete payload slots when applicable
- `payload_num`: numeric side payload when applicable
- `modifier_ids`: route, dose unit, form, status, site, severity, etc.
- `structural_role`: opener, closer, overlay, none
- `window_type_id`: regime label if known
- `t_abs_hours`: absolute time from subject start
- `dt_prev_hours`: delta from previous fused event

Current `EventToken` should remain available during migration for backward compatibility, but v2 collation should consume fused events rather than raw token bundles whenever possible.

## 4.2 Family-specific payload codecs

### Numeric measurements

- retain the current cVAE plus RVQ latent codec
- stop exposing raw measurement bundles as the main local-model input
- instead, fuse `variable id + RVQ codes + selected metadata` into one event embedding through an event composer
- keep factorized decoding for analysis and loss routing

Why: the measurement codec remains useful, but recent tokenization evidence favors joint event encoding over forcing the model to re-bind split attributes during pretraining.

### Qualitative observations

- keep OBS as a separate family
- canonicalize exact high-frequency value surfaces
- represent as `concept + categorical payload value`
- move toward one fused event embedding at the model interface

### Diagnoses, procedures, medications

- use canonical symbolic concept ids plus structured modifiers
- use MedTok embeddings where supported
- where MedTok is not supported, do not pretend ontology coverage exists
- instead, preserve exact canonicalized source concepts with stable family-specific vocabularies and optional parent metadata

### Structural events

- keep them first-class
- treat them as explicit regime and overlay events, not as generic categories
- preserve action, target type, site, and overlay role as modifiers

### Demographics and static context

- keep the minimal global-special policy
- expose these as fused special events or window-prefix context, not as repeated timeline noise

## 4.3 Ontology strategy for discrete variables

Do not build a new ontology from scratch.

Recommended practical stack:

1. canonical source code systems first
   - diagnoses: ICD-9-CM, ICD-10-CM, SNOMED if available
   - procedures: ICD-9-PCS, ICD-10-PCS, CPT, HCPCS when available
   - medications: RxNorm, NDC, ingredient or product identifiers when available
   - labs and observations: LOINC when available, otherwise stable MIMIC/MEDS itemid surfaces
2. MedTok embeddings as the primary semantic embedding source where coverage exists
3. exact family-specific residual vocabularies for unsupported but frequent canonicalized surfaces
4. optional parent metadata from source hierarchies or OMOP standard concepts if the upstream ETL already exposes them
5. explicit UNK or explicit drop for the long tail
6. no hashing in production

This gives ontology support without requiring the project to build a new universal terminology resource.

## 4.4 Tokenization design rule

The model should operate on fused event embeddings, not on raw decomposed bundles.

A good v2 rule is:

- keep decomposition at the codec level for interpretability and decoding
- fuse at the event-composer level before the local sequence model

That resolves the consistency concern between a rich measurement path and simpler symbolic families.

## 5. V2 Architecture Design

## 5.1 Event composer

Add a shared event composer that maps each typed fused event into one `d_model` vector.

Inputs to the composer:

- family embedding
- concept embedding
- payload embedding or payload projection
- modifier embeddings
- structural role embedding
- time features

Outputs:

- one fused event embedding per event
- optional family-specific decoder side metadata

## 5.2 Local regime encoder

Keep local attention inside bounded chunks within semantic windows.

Target behavior:

- local attention handles dense within-window interactions
- chunking remains a compute device, not a semantic transition
- structural events and global specials remain visible inside each chunk

## 5.3 Window summarization

Each semantic window should emit two outputs:

1. `state_update_summary`
   - compressed evidence used to update the persistent global health state
2. `memory_candidate_set`
   - a small scored set of exact events worth preserving beyond the compressive state

This replaces the current single-summary mentality with an explicit split between compressive memory and exact memory.

## 5.4 Persistent global latent health state

Replace the current transformer-only global summary path with a persistent latent state module.

Default target:

- a Mamba-class or equivalent selective state-space model over semantic windows

Inputs per window:

- window summary
- window type
- gap since previous window
- optional admission boundary flags

Outputs:

- updated latent health state
- context vector returned to the local model for the next window or chunk

Design rule:

- the global state represents slowly evolving health and trajectory context
- it is not required to preserve every rare exact detail

## 5.5 Exact episodic memory reservoir

Add a sparse exact memory path to offset fixed-state channel limits.

Memory contents should be limited to high-salience events such as:

- first occurrence of major chronic diagnoses
- major procedures
- code-status changes
- ventilation, shock, RRT, CPR, sepsis-bundle transitions
- extreme or clinically critical measurement events
- discharge and readmission anchors

Retrieval should be conditioned on:

- current local query
- current global latent state
- current window type

This memory should be tiny and selective, not a second full history encoder.

## 5.6 Global-to-local conditioning

The local model for a chunk should receive:

- the current persistent latent health state
- retrieved exact memory items
- static demographic specials

This should replace the current simpler shifted global-context fusion with a clearer mechanism:

- compressive state for broad context
- exact memory for rare precise facts

## 5.7 Output heads and objectives

Primary objective:

- marked time-to-event prediction

Suggested factorization:

1. next event time within window
2. next event family
3. next event concept or mark
4. measurement payload distribution when the next event is numeric
5. next window transition time and next window type at semantic boundaries

Secondary objectives:

- next-event autoregressive loss for training stability
- measurement latent reconstruction or consistency loss
- memory selection supervision or retrieval-usefulness regularization

The key change is that time is no longer only an auxiliary prediction head. It becomes part of the central generative objective.

## 6. Concrete Implementation Work Packages

## WP0. Freeze the v1 baseline and define comparison points

Goal:

- lock a fair v1 baseline before v2 refactors begin

Files:

- `SCHEMA_TOKENS.md`
- `TIME_MODEL.md`
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/loss.py`
- `scripts/train_transformer_v1.py`

Acceptance:

- one reproducible v1 baseline config
- family-stratified evaluation
- at least one long-horizon validation split

## WP1. Add a fused event interface without breaking current tokenization

Goal:

- create a migration-safe typed event layer

Detailed implementation plan:

- `FOUNDATION_MODEL_V2_WP1_PLAN.md`

Suggested files:

- `src/ehr_hier/data/token_types.py`
- new `src/ehr_hier/data/event_frames.py`
- `src/ehr_hier/data/subject_timeline_builder.py`
- `src/ehr_hier/tokenizers/interfaces.py`

Acceptance:

- timeline builder can emit current `EventToken` bundles and new fused event objects
- collator accepts fused events in a compatibility mode
- tests cover parity on timing and window metadata

## WP2. Refactor tokenization into codec modules

Goal:

- separate codec logic from model-facing event composition

Suggested files:

- new `src/ehr_hier/tokenizers/codecs/base.py`
- new `src/ehr_hier/tokenizers/codecs/measurement_numeric.py`
- new `src/ehr_hier/tokenizers/codecs/measurement_qual.py`
- new `src/ehr_hier/tokenizers/codecs/semantic_symbolic.py`
- new `src/ehr_hier/tokenizers/codecs/structural.py`
- existing `src/ehr_hier/tokenizers/measurement_encoder.py`
- existing `src/ehr_hier/tokenizers/medtok_*`

Acceptance:

- every family returns a fused event payload object
- measurement codec still uses cVAE plus RVQ internally
- no production hash fallback remains

## WP3. Harden discrete semantic coverage without building a new ontology

Goal:

- make unsupported symbolic surfaces still clinically coherent

Suggested files:

- `src/ehr_hier/tokenizers/medtok_crosswalk.py`
- `src/ehr_hier/tokenizers/medtok_canonicalize.py`
- `scripts/audit_tokenization_flow.py`
- `MEDTOK_CODE_SYSTEM_CONTRACT.md`

Acceptance:

- MedTok remains preferred where available
- unsupported frequent canonical surfaces enter exact residual vocabularies by family
- optional parent metadata is recorded when cheaply derivable
- held-out coverage reports are explicit and reproducible

## WP4. Add the event composer and fused local input path

Goal:

- make the local model consume one fused event embedding per event

Suggested files:

- new `src/ehr_hier/transformer/event_composer.py`
- `src/ehr_hier/transformer/embeddings.py`
- `src/ehr_hier/transformer/collator.py`
- `src/ehr_hier/transformer/model.py`

Acceptance:

- local sequence length is measured in fused events, not bundle tokens
- measurement bundles are internally composed before attention
- ablation compares bundle input vs fused-event input

## WP5. Replace the global summary chain with latent state plus exact memory

Goal:

- implement long-range modeling that scales across admissions and years

Suggested files:

- new `src/ehr_hier/transformer/global_state.py`
- new `src/ehr_hier/transformer/episodic_memory.py`
- `src/ehr_hier/transformer/aggregator.py`
- `src/ehr_hier/transformer/model.py`

Acceptance:

- one persistent state update per semantic window
- memory proposal and retrieval path exist
- local chunks consume both latent state and retrieved memory
- ablations compare:
  - transformer global path only
  - latent state only
  - latent state plus exact memory

## WP6. Promote marked time-to-event to the primary objective

Goal:

- move beyond token CE as the defining pretraining loss

Suggested files:

- new `src/ehr_hier/objectives/marked_tte.py`
- `src/ehr_hier/transformer/loss.py`
- `src/ehr_hier/transformer/model.py`
- tests under `tests/test_*time*`

Acceptance:

- next-event time and mark are trained jointly
- next-window time and type are trained at semantic boundaries
- numeric measurements use a family-appropriate payload loss
- token CE can be retained as an auxiliary stabilization term

## WP7. Training curriculum and evaluation

Goal:

- train v2 in stages that minimize instability

Recommended curriculum:

1. fused event path plus autoregressive token or family loss
2. add marked intra-window time-to-event loss
3. add latent global state
4. add exact memory
5. add semantic boundary time and next-window losses

Evaluation must include:

- standard classification probes
- regression tasks
- explicit time-to-event tasks
- long-history and multi-admission subsets
- external or temporal-shift evaluation
- ablations for:
  - fused event composition
  - measurement latent codec
  - latent global state
  - exact memory
  - marked TTE objective

## 7. Minimum Publication Bar

A publishable v2 paper should beat strong recent baselines, not only older BEHRT-style models.

At minimum compare against:

- current repo v1 baseline
- a strong transformer baseline
- a long-context or state-space baseline when feasible

The key evidence should show:

- better long-horizon performance on patients with multiple admissions
- better time-to-event and regression capability, not only classification
- better robustness under temporal or external shift
- clear gains from the measurement codec only when integrated into the fused-event plus marked-TTE system

## 8. Recommended First Implementation Order

Start with this order:

1. WP0 baseline freeze
2. WP1 fused event interface
3. WP2 codec refactor
4. WP4 event composer
5. WP6 marked TTE objective
6. WP5 latent state plus exact memory
7. WP7 ablations and probes

Do not start with the Mamba swap alone.

The correct dependency order is:

- event interface first
- objective second
- long-range state and memory third

## 9. Non-goals

The following are explicitly out of scope for v2:

- building a new universal medical ontology
- forcing all event families through one shared discrete codebook
- pretending unsupported semantic surfaces have ontology grounding when they do not
- replacing explicit structural signifiers with purely learned latent boundaries

## 10. Files to Open First for V2 Work

- `AGENTS.md`
- `TRANSFORMER_STAGE_HANDOFF.md`
- this file
- `SCHEMA_TOKENS.md`
- `TIME_MODEL.md`
- `configs/data/structural_codes.yaml`
- `src/ehr_hier/data/subject_timeline_builder.py`
- `src/ehr_hier/transformer/collator.py`
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/loss.py`
