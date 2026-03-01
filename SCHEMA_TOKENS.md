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
- `window_hook = <non-None>` iff the label (or code) is configured as a boundary

### 2.2 Model-driven delimiters (window marker tokens)
The collator (`src/ehr_hier/transformer/collator.py`) segments timelines on `window_hook`
and inserts **window marker tokens** inside each window sequence:

    [special_tokens...] [WIN_TYPE] [window_tokens...] [WIN_END or WIN_<NEXT_TYPE>]

These marker tokens are the learnable "signifiers" that the model can predict to end a
window and (optionally) specify the type of the next window.

Configuration: `WindowMarkerConfig` (and `vocab_config["window_markers"]` for the model)
- `type_token_offset`: global token id where `WIN_<TYPE>` starts.
- `num_types`: number of supported window types.
- `end_mode`:
  - `"end_token"`: append a dedicated `WIN_END` token id.
  - `"next_type"`: append the *next* window's `WIN_<TYPE>` id (last window still uses `WIN_END`).

Window type id inference:
- If the first token in a window carries `cat_attrs["window_type_id"]`, use it.
- Else if it carries `cat_attrs["struct_label_id"]`, use `struct_label_id + 1` (reserve `0` for UNK).
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

### 3.2 Medications / Diagnoses / Procedures: MedTok-backed codes
Medications, diagnoses, and procedures are mapped to MedTok vocab ids (from ontology-graph
pretraining) and emitted as `EventToken(value_id=<global_id>)`.

See: `src/ehr_hier/tokenizers/medtok_*` and `artifacts/medtok/*`.

Metadata:
- `cat_attrs`: route/form/frequency/unit ids (optional, if configured)
- `num_attrs`: normalized numeric attributes (optional; currently not projected by the collator
  except for `num_attrs["numeric_value"]`)

### 3.3 Numeric side-channel (`numeric_values`)
The collator extracts `EventToken.num_attrs["numeric_value"]` into a dense tensor
`numeric_values` (shape `(B,W,L,1)`) plus a `numeric_mask`.

Current intended uses:
- medication dose / rate / duration when available
- any future scalar attributes that should be modeled as "continuous alongside discrete"

Measurements typically do *not* rely on this side-channel (values are discretized via RVQ).

## 4) Vocabulary layout: global ids and head routing

### 4.1 Source of truth: vocab manifest
`artifacts/vocab_manifest.json` defines offsets for token families. Conceptually:

    global_id = offset[family] + local_id

This is enough for tokenizers and debug tooling. For training, the transformer additionally
needs **sizes** and **routing rules** (which ids are predicted by which head).

### 4.2 AET head lanes (recommended)
AET uses multiple output heads to avoid a monolithic softmax:
- `logits_struct`: special + window markers + (often) structural signifiers
- `logits_rvq`: RVQ value tokens
- `logits_meas`: measurement identity tokens
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
