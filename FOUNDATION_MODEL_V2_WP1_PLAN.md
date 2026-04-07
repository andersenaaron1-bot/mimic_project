# Foundation Model V2 WP1 Plan

Status: proposed implementation plan
Last updated: 2026-04-07
Depends on: WP0 baseline freeze
Current WP0 note: the user is running the intended v1 production baseline now; treat that run as the baseline candidate unless it fails validation

## 1. WP1 Objective

WP1 introduces a migration-safe fused event interface without breaking the current v1 token path.

Concretely, WP1 must achieve all of the following:

- preserve the current `build_subject_timeline()` default contract
- add a new typed fused-event representation for post-v1 work
- allow the timeline builder to emit both legacy tokens and fused events from the same source pass
- allow the collator to accept fused events in a compatibility mode by expanding them back to legacy tokens
- establish parity tests so later WP2 and WP4 refactors have a stable foundation

WP1 is intentionally not the full tokenization refactor. It is the interface layer that makes later refactors safe.

## 2. Current Repo Constraints

The current repo has several hard constraints that WP1 must respect:

- `build_subject_timeline()` currently returns `List[EventToken]`
- `EventTokenEncoder` currently only exposes `encode_event(...)->List[EventToken]`
- many tests construct `EventToken` directly and assume token-level ordering
- the collator expects token lists, not event objects
- measurement semantics currently live in token bundles:
  - `MEAS_CODE + RVQ_*`
  - `OBS_CODE + OBS_VALUE`
- some raw events emit multiple semantic emissions at the same timestamp
  - for example structural plus original procedure token

Because of this, WP1 must not replace token-level behavior in place. It must layer a new representation on top of the current one.

## 3. Design Principles

### 3.1 No breaking changes to v1 by default

The following default behaviors must remain unchanged:

- `build_subject_timeline()` returns `List[EventToken]`
- all current token-based tests continue to pass unchanged
- current trainer and collator continue to consume token timelines without awareness of fused events

### 3.2 No union return type for the main builder

Do not make `build_subject_timeline()` return "sometimes tokens, sometimes frames." That would spread migration complexity everywhere.

Instead:

- keep the current function intact
- add a new explicit builder entrypoint for dual-emission

### 3.3 Fused events must be lossless relative to current tokenization

The new fused event object must preserve enough information to reconstruct the current token bundle exactly in compatibility mode.

That means a fused event must retain at least:

- ordered legacy token ids
- timing fields
- window and structural metadata
- categorical and numeric attributes
- the canonical source code surface or equivalent identity key

### 3.4 Cardinality is semantic emission, not raw DB row

One raw MEDS event may produce more than one fused event if the current pipeline already treats it as more than one semantic emission.

Examples:

- measurement numeric event -> one fused event with multiple legacy measurement tokens
- qualitative observation event -> one fused event with two legacy observation tokens
- structural-plus-original event -> two fused events if both semantics are retained

This mirrors current behavior and avoids burying important semantics in opaque subfields.

## 4. Proposed Data Model

## 4.1 New file

Add:

- `src/ehr_hier/data/event_frames.py`

## 4.2 Proposed enums and dataclasses

### `EventPayloadKind`

Purpose:

- identify how a fused event carries its payload

Suggested values:

- `NONE`
- `LEGACY_TOKEN_BUNDLE`
- `RVQ_LATENT`
- `CATEGORICAL_VALUE`
- `NUMERIC_SCALAR`
- `MODIFIER_SET`

For WP1, most events will use `LEGACY_TOKEN_BUNDLE`. More precise payload kinds become active in WP2.

### `EventFrame`

Purpose:

- model-facing fused event object for migration and later composition

Suggested required fields:

- `family_id: int`
- `concept_id: int`
- `concept_key: str | None`
- `payload_kind: int`
- `legacy_token_ids: list[int]`
- `legacy_token_categories: list[int]`
- `t_from_start_hours: float`
- `dt_from_prev_hours: float`
- `cat_attrs: dict[str, int]`
- `num_attrs: dict[str, float | None]`
- `raw_time: datetime | None`
- `window_hook: str | None`

Suggested optional migration fields:

- `source_code: str | None`
- `source_event_index: int | None`
- `emission_index_within_event: int | None`
- `primary_token_idx: int`

Key rule:

- `legacy_token_ids` must preserve exact current token order for compatibility expansion

### `TimelineBuildOutput`

Purpose:

- explicit dual-emission return object for migration-safe timeline building

Suggested fields:

- `tokens: list[EventToken]`
- `frames: list[EventFrame]`

Optional later additions can include audit metadata, but WP1 should stay lean.

## 5. Builder API Plan

## 5.1 Keep the current function untouched

Retain:

- `build_subject_timeline(...)->List[EventToken]`

This stays the default v1 path.

## 5.2 Add a new dual-emission builder

Add:

- `build_subject_timeline_output(...)->TimelineBuildOutput`

Suggested placement:

- `src/ehr_hier/data/subject_timeline_builder.py`

This new entrypoint should:

- execute the same routing and timing logic as the existing builder
- produce the exact token list currently used by v1
- additionally emit aligned `EventFrame` objects

Implementation rule:

- refactor shared logic into an internal private function instead of copy-pasting the full builder

## 5.3 Internal emission helper

Add a private helper in `subject_timeline_builder.py`:

- `_frame_from_emitted_tokens(...) -> EventFrame`

This helper should:

- consume the already-emitted token bundle for one semantic emission
- infer `family_id`, `concept_id`, and `payload_kind`
- attach legacy token ids and metadata

WP1 should build frames from emitted tokens, not by re-implementing every encoder.

That keeps the migration low-risk and guarantees parity with current tokenization.

## 6. Encoder Interface Plan

## 6.1 Keep `EventTokenEncoder` intact

Do not require existing encoders to implement fused-event emission yet.

Current protocol in [interfaces.py](C:/Users/King%20Kong/Desktop/EHRPRED/mimic_project/src/ehr_hier/tokenizers/interfaces.py) remains valid for v1 and WP1.

## 6.2 Add optional future-facing protocol

Extend:

- `src/ehr_hier/tokenizers/interfaces.py`

with an optional protocol such as:

- `EventFrameEncoder`

Suggested method:

- `encode_event_frame(ev: Any, dt_hours: float) -> list[EventFrame]`

However:

- no existing encoder needs to implement this in WP1
- this protocol exists only to reserve the migration target for WP2

## 6.3 Adapter stance

WP1 should not create a complex adapter stack yet.

The correct sequence is:

1. token encoders remain the source of truth
2. builder derives frames from emitted token bundles
3. later WP2 moves semantic families to frame-native codecs

## 7. Collator Compatibility Plan

## 7.1 Preserve token-first default

The current collator behavior remains unchanged for token input.

## 7.2 Add explicit compatibility mode

Extend:

- `src/ehr_hier/transformer/collator.py`

with an explicit input mode such as:

- `input_mode="tokens" | "frames"`

Default:

- `tokens`

When `frames` is selected:

- the collator expands each `EventFrame` into legacy `EventToken` objects internally
- it then follows the existing code path

This keeps all segmentation and marker behavior unchanged in WP1.

## 7.3 Frame-to-token expansion helper

Add helper:

- `_frames_to_event_tokens(frames: list[EventFrame]) -> list[EventToken]`

Rules:

- preserve order exactly
- preserve `dt_from_prev_hours` only on the first legacy token of each frame
- preserve `t_from_start_hours`, `cat_attrs`, `num_attrs`, `raw_time`, `window_hook`
- preserve category ids from `legacy_token_categories`

Measurement and observation frames must expand back into the current bundle shapes without loss.

## 8. File-by-File Work Plan

## Phase A. Add the new data model

Files:

- new `src/ehr_hier/data/event_frames.py`

Tasks:

- add `EventPayloadKind`
- add `EventFrame`
- add `TimelineBuildOutput`

Acceptance:

- dataclasses instantiate cleanly
- unit tests cover defaults and field round-tripping

## Phase B. Add builder dual-emission

Files:

- `src/ehr_hier/data/subject_timeline_builder.py`

Tasks:

- add internal shared builder core
- add `build_subject_timeline_output()`
- keep `build_subject_timeline()` delegating to the core and returning only `.tokens`
- add `_frame_from_emitted_tokens()`

Acceptance:

- token output from `build_subject_timeline()` is unchanged
- dual-emission output produces token list identical to the legacy builder
- emitted frames align one-to-one with semantic emissions

## Phase C. Reserve future encoder protocol

Files:

- `src/ehr_hier/tokenizers/interfaces.py`

Tasks:

- add optional `EventFrameEncoder` protocol
- document that it is not yet required by existing encoders

Acceptance:

- no current encoder breaks
- typing remains readable

## Phase D. Add collator frame compatibility mode

Files:

- `src/ehr_hier/transformer/collator.py`

Tasks:

- add explicit frame input mode
- add frame-to-token expansion helper
- keep token path as the default

Acceptance:

- collating tokens directly matches current behavior
- collating frames in compatibility mode yields identical tensors on parity fixtures

## Phase E. Add tests

Files:

- new `tests/test_event_frames.py`
- new `tests/test_collator_frame_compat.py`
- extend `tests/test_subject_timeline_builder.py`

Suggested test cases:

1. single-token semantic event becomes one frame with one legacy token
2. measurement numeric bundle becomes one frame with multiple legacy token ids
3. OBS code/value bundle becomes one frame with two legacy token ids
4. structural plus original semantic emission produces two frames when current builder emits two semantic outputs
5. `build_subject_timeline_output().tokens` matches `build_subject_timeline()`
6. collator on token input matches collator on frame input in compatibility mode

## 9. Proposed Interface Details

## 9.1 `concept_id` in WP1

For WP1, `concept_id` should be pragmatic rather than philosophically perfect.

Recommended rule:

- use the first legacy token id as `concept_id`

Why:

- it is stable
- it avoids inventing a new concept mapping during WP1
- it is enough to support parity and later composer work

WP2 can replace this with cleaner family-specific concept ids where needed.

## 9.2 `payload_kind` in WP1

Recommended initial rules:

- single-token symbolic event -> `NONE`
- measurement numeric bundle -> `LEGACY_TOKEN_BUNDLE`
- qualitative observation bundle -> `LEGACY_TOKEN_BUNDLE`
- structural single token -> `NONE`
- demographic special -> `NONE`

WP1 should not overfit payload semantics yet.

## 9.3 Timing semantics

Frame timing must follow current emission semantics:

- frame `dt_from_prev_hours` equals the dt carried by the first token in that semantic emission
- subsequent legacy tokens inside the same frame carry zero dt when expanded

This preserves current training and segmentation assumptions.

## 10. Risks and Mitigations

### Risk 1: hidden return-type breakage

Mitigation:

- do not change `build_subject_timeline()` return type
- add a new builder entrypoint instead

### Risk 2: parity bugs in measurement bundles

Mitigation:

- build frames from emitted token bundles rather than re-deriving semantics in WP1
- add explicit measurement and OBS parity tests

### Risk 3: structural timing drift

Mitigation:

- use emitted token objects as the source for frame construction
- verify structural-plus-original-event cases in tests

### Risk 4: collator auto-detection ambiguity

Mitigation:

- do not auto-detect by default
- require explicit `input_mode="frames"` for compatibility path

## 11. Acceptance Criteria

WP1 is complete when all of the following hold:

- current token-based training path is unchanged by default
- a new `EventFrame` data model exists
- a new dual-emission builder exists
- `build_subject_timeline_output().tokens` exactly matches the legacy token builder
- the collator accepts frames in explicit compatibility mode
- parity tests pass for token input versus frame input

WP1 is not required to:

- make encoders frame-native
- change model inputs to fused events
- implement an event composer
- change training objectives

Those belong to later work packages.

## 12. Recommended Implementation Order

1. add `event_frames.py`
2. add builder dual-emission path
3. add tests for builder parity
4. add collator frame compatibility mode
5. add collator parity tests
6. add optional protocol in `interfaces.py`

This order minimizes breakage and keeps every intermediate step testable.
