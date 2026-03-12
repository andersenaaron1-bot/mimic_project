# Token Schema (SCHEMA_TOKENS)

This document is the contract between tokenization (`EventToken`) and the hierarchical
transformer (`AdaptiveEpisodicTransformer`, a.k.a. AET).

Final vocab size and categories will be set with the full dataset available. What must remain
stable is:
- how events become `EventToken` bundles,
- how windows-of-care are delimited and labeled,
- how global token ids are laid out and routed into model heads.

## 1) Intermediate representation: `EventToken`

All tokenizers emit `EventToken` objects (see `src/ehr_hier/data/token_types.py`).
This is the only object that timeline building/windowing logic should depend on.

Required fields (conceptual):
- `value_id` (int): the global token id to feed into the model.
- `category_id` (int): coarse `TokenCategory` (type embedding, pooling masks, routing).
- `t_from_start_hours` (float): absolute time from subject timeline start (hours).
- `dt_from_prev_hours` (float): delta from previously emitted token (hours).
- `cat_attrs` (dict[str,int]): optional categorical metadata already mapped to global ids.
- `num_attrs` (dict[str,float|None]): optional numeric metadata (may be projected later).
- `window_hook` (str|None): optional boundary marker used for segmentation into windows.

Notes:
- `t_from_start_hours` is the canonical time signal for windowing and cRoPE. `dt_from_prev_hours`
  is retained mainly for debugging and optional future features.
- `category_id` is not the same as the "head lane". For example, structural tokens may be
  `TokenCategory.STRUCTURAL` while still living in the "STRUCT/SPECIAL" head lane.

## 2) Windowing: "windows of care" + signifier tokens

### 2.1 Data-driven boundary seeds (structural codebook)
Boundary seeds and overlay signifiers are defined in `configs/data/structural_codes.yaml`
and loaded via `src/ehr_hier/data/structural_codes.py`.

Concepts:
- **Structural tokens**: tokens that represent care setting changes (admission, ICU transfer,
  OR, discharge, etc.) and critical overlays (mechanical ventilation, RRT, shock, etc.).
- **Boundary labels**: a subset of structural labels that *split* the timeline into new
  windows by setting `EventToken.window_hook`.
- **Overlay labels**: structural labels that do *not* split windows; they are in-window
  conditioning signals (e.g., MV on/off) that can start/end within a higher-level window.
- **Soft signifiers**: convenience list of codes that should emit a structural marker token
  without necessarily being a boundary (defaults to label `SOFT::<code>`).

Emitted structural `EventToken` fields (by convention):
- `category_id = TokenCategory.STRUCTURAL`
- `cat_attrs["struct_label_id"] = <0-based label id>`
- `cat_attrs["transition_action_id"] = <int>` when the codebook marks the token as a
  regime-transition candidate (`open_next`, `close_current`, `close_open`, `suppress`)
- `cat_attrs["transition_window_type_id"] = <int>` when the token implies a target care-regime type
- `window_hook = <non-None>` iff the label (or code) is configured as a boundary

### 2.2 Model-driven delimiters (window marker tokens)
The collator (`src/ehr_hier/transformer/collator.py`) now has a 3-level hierarchy:
- **semantic windows**: care-regime segments produced by transition-bundle segmentation
- **local chunks**: bounded local attention units inside a semantic window
- **tokens**: ordered event-token bundles inside each chunk

Semantic windows are segmented using **transition bundles**:
- explicit transition metadata takes precedence
- nearby transition candidates are grouped into bundles
- sparse administrative transition chains can be merged
- legacy `window_hook` boundaries remain as a fallback

It then inserts **window marker tokens** inside each local chunk sequence:

    [special_tokens...] [WIN_TYPE] [chunk_tokens...] [WIN_CONTINUE | WIN_END | WIN_<NEXT_TYPE>]

Marker semantics:
- `WIN_CONTINUE`: continue within the current semantic window using another local chunk
- `WIN_END`: end the current semantic window
- `WIN_<NEXT_TYPE>`: end the current semantic window and indicate the next semantic-window type

This keeps the global chain at semantic-window granularity while still giving the local
model a bounded sequence budget.

### 2.3 Generation-time grammar (state machine)
For constrained autoregressive rollout, marker tokens are handled by an explicit state
machine (`src/ehr_hier/transformer/generation.py`):

- `inside_chunk`: only non-marker content tokens are legal
- `chunk_end`: `WIN_CONTINUE` is always legal; `WIN_END`/`WIN_<TYPE>` are legal only when the
  current position is a semantic boundary candidate
- `window_end`: only `WIN_<TYPE>` is legal to open the next semantic window

This prevents invalid chains such as emitting `WIN_END` mid-chunk or using chunk overflow as a
semantic transition signal.

Configuration: `WindowMarkerConfig` (and `vocab_config["window_markers"]` for the model)
- `type_token_offset`: global token id where `WIN_<TYPE>` starts.
- `num_types`: number of supported window types.
- `end_mode`:
  - `"end_token"`: append a dedicated `WIN_END` token id.
  - `"next_type"`: append the *next* window's `WIN_<TYPE>` id (last window still uses `WIN_END`).

Window type id inference:
- If segmentation already assigned a typed regime window, use that type id.
- Else if the first token in a window carries `cat_attrs["window_type_id"]`, use it.
- Else if an early token in the window carries `cat_attrs["transition_window_type_id"]`, use it.
- Else fall back to `unk_type_id`.

## 3) Token families (what we model)

### 3.1 Measurements: cVAE -> attention-shortlisted RVQ (CODA-inspired)
Measurements are encoded as a small *bundle* of tokens:
1) a measurement identity token (variable id), then
2) one token per RVQ codebook (the discretized value latent).

See: `src/ehr_hier/tokenizers/measurement_encoder.py` and `src/ehr_hier/tokenizers/value_tokenizer.py`.

Ordering (per MEDS measurement event):
- `MEAS_CODE(var_id)` with `dt_from_prev_hours = dt_event`
- `RVQ_L0(idx0)`, `RVQ_L1(idx1)`, ... with `dt_from_prev_hours = 0`

This gives the model:
- explicit *what* (the measurement variable),
- discretized *value semantics* (RVQ codes) that can condition downstream dynamics,
- access to raw value normalization and conditioning (age/sex/dt_prev) inside the cVAE.

### 3.1b Qualitative Measurement Surface (`OBS_QUAL`)
Non-numeric measurement/charted events (for example `LAB//<itemid>//UNK|N/A`) are not
sent through cVAE/RVQ. They are emitted as a 2-token categorical bundle:
- `OBS_CODE(itemid_or_code_hash)`
- `OBS_VALUE(value_text_or_code_tail_hash)`

Ordering:
- `OBS_CODE` carries `dt_from_prev_hours = dt_event`
- `OBS_VALUE` carries `dt_from_prev_hours = 0`

These remain `TokenCategory.MEASUREMENT` but use dedicated global-id ranges from
the sparse vocab contract (`observation_code`, `observation_value`).
The pragmatic v1 routing also sends high-volume chart aliases like `Blood Pressure`
through this OBS path when they do not map to the numeric cVAE variable map.
For v1 this remains a two-token bundle rather than a fused event-value token:
the split keeps event identity and value identity factorized while keeping the
vocabulary modest. Exact OBS vocabularies can be added later without changing
the bundle shape.

### 3.2 Medications / Diagnoses / Procedures: MedTok-backed codes
Medications, diagnoses, and procedures are mapped to MedTok vocab ids (from ontology-graph
pretraining) and emitted as `EventToken(value_id=<global_id>)`.

See: `src/ehr_hier/tokenizers/medtok_*` and `artifacts/medtok/*`.
The pinned source-to-key contract for v1 lives in `MEDTOK_CODE_SYSTEM_CONTRACT.md`.

Metadata:
- `cat_attrs`: route/form/frequency/unit ids (optional, if configured)
- `num_attrs`: normalized numeric attributes (optional; currently not projected by the collator
  except for `num_attrs["numeric_value"]`)

For the minimal v1 structural contract, synthetic process action/entity tokens are disabled by
default. Structural semantics should come from the structural codebook path, not from extra
generated token families.

Fallback policy for unresolved semantic surfaces:
- first try explicit MedTok resolution (`exact`, `canonicalized`, `parent_lookup`,
  `crosswalk_lookup`, `lexical_bridge`)
- then try an exact residual vocabulary for common unresolved surfaces
- only the rare tail should hash or drop, depending on experiment policy

### 3.3 Numeric side-channel (`numeric_values`)
The collator extracts `EventToken.num_attrs["numeric_value"]` into a dense tensor
`numeric_values` (shape `(B,W,C,L,1)` in the chunked hierarchy) plus a `numeric_mask`.

Current intended uses:
- medication dose / rate / duration when available
- any future scalar attributes that should be modeled as "continuous alongside discrete"

Measurements typically do *not* rely on this side-channel (values are discretized via RVQ).

### 3.4 Global demographic specials + rare-critical OTHER
To keep timeline length controlled while retaining static context:
- repeated anthropometric admin events (BMI/weight/height aliases) are handled via
  **global demographic SPECIAL tokens** (sex, age bucket, BMI bucket) emitted once per subject;
- collator behavior prefixes SPECIAL tokens to each window/chunk, making them globally attendable;
- a lightweight clinically informed rare-critical keyword list is routed from `OTHER`
  into `STRUCTURAL` to avoid dropping high-acuity events.

## 4) Vocabulary layout: global ids and head routing

### 4.1 Source of truth: sparse vocab contract
`artifacts/token_vocab_sparse_v1.json` is the sparse/base token contract. It is the
single source of truth for global token families before dense runtime remapping. Conceptually:

    global_id = offset[family] + local_id

The sparse contract contains:
- family offsets
- source sizes
- exact residual fallback family sizes (when fallback vocab JSONs are present) or legacy hash ranges
- special/window marker ids
- runtime head routing for model-visible families

Legacy note:
- `artifacts/vocab_manifest.json` remains only as a compatibility fallback for older scripts.
- New runtime vocab generation should derive from the sparse contract, not from the legacy manifest.

For v1 experiments:
- `configs/data/tokenization_v1.yaml` remains the human-edited policy/config input
- `artifacts/token_vocab_sparse_v1.json` is the generated sparse/base contract
- the dense/runtime vocab bundle is derived only from that sparse contract plus observed-id compaction

### 4.2 AET head lanes (recommended)
AET uses multiple output heads to avoid a monolithic softmax:
- `logits_struct`: special + window markers + (often) structural signifiers
- `logits_rvq`: RVQ value tokens
- `logits_meas`: measurement identity tokens + qualitative observation bundles
- `logits_medtok`: MedTok semantic tokens (diagnosis/procedure/medication, etc.)

The loss routes targets to heads by **token id ranges** (see `src/ehr_hier/transformer/loss.py`).
Therefore, for a given experiment you must provide a `vocab_config` that includes:
- `offsets` and `size_*` keys, or an explicit `routing` table, and
- `window_markers` offsets for the synthetic marker tokens.

Compatibility note:
- Today the repo contains both (a) sparse, high-offset vocab manifests (good for multi-source
  integration) and (b) a compact, lane-based `vocab_config` expected by AET embeddings/heads.
  If you keep sparse offsets, you will need an embedding/routing layer that does *not* require
  `nn.Embedding(max_global_id+1)`; if you keep compact offsets, ensure the manifest and encoders
  use the same layout.
