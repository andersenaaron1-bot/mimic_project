# Repository Charter

## Mission

This repo is the v2 line for a **dual-memory patient world model** over
heterogeneous longitudinal EHR.

The goal is not a better flat sequence encoder. The goal is a generative model
that separates:

- **local attention** for within-window event binding
- **global latent state** for compressive current condition and transition hazard
- **patient-internal memory** for exact personalized facts that must not be compressed away
- **cohort-external precedent memory** for analogous prior states and their observed futures

The intended outcome is a model that can do more than next-event prediction:

- generate plausible future event streams across semantic care windows
- adapt online to new patient-specific evidence through memory without weight updates
- retrieve analogous clinical states rather than only similar token prefixes
- support intervention-conditioned prospective rollout

This is a **world-model** framing. It is not a claim of unrestricted causal
counterfactual reasoning. Any counterfactual language in the paper must stay
conditional on explicit causal assumptions.

## Canonical Metadata

The only canonical metadata files in this repo are:

- `AGENTS.md`
- `paper/foundation_model_v2_outline.tex`
- `paper/foundation_model_v2_refs.bib`

All other prior planning or contract markdown files have been retired to avoid
split design centers.

## Core Representation Contract

- `EventFrame` is the public semantic timeline unit.
- Family-specific codecs stay behind one shared event algebra.
- Numeric measurements use cVAE plus RVQ.
- Diagnoses, procedures, medications, and structural/process events remain
  typed symbolic events with canonical routing.
- Structural transitions and overlays define explicit care-regime windows.

The model objective remains explicitly marked and typed:

- discrete marks use family-, payload-, and family-conditioned concept CE
- numeric measurement marks use continuous likelihood on the event lane
- event timing and inter-window gap timing use continuous-time NLL terms
- dense token CE remains only as an auxiliary stabilization path

## Target Architecture

### 1. Local event model

Local attention operates over typed fused events inside semantic care windows.
Its job is:

- content binding
- short-range physiological and intervention reasoning
- extraction of compact state evidence
- proposal of exact memory writes

### 2. Global latent health state

The global latent is a **belief state**, not a patient file.

Its job is to represent:

- current disease burden
- support intensity and instability
- unresolved burden carried across windows
- transition hazard across care regimes
- tempo across long irregular gaps

It should not be relied on to preserve exact rare facts.

The latent contract is now frozen at the semantic/interface level:

- it is a **window-boundary belief state**
- it updates primarily by **jump updates at semantic-window boundaries**
- it must be explicitly **gap-sensitive** across irregular time
- it is the compressive state used for:
  - precedent-query construction
  - global-to-local conditioning
  - transition and timing heads
- it is not the primary store of exact history

The target latent family is also now frozen at the family-class level:

- a **multi-slot hybrid jump-plus-drift latent**

The intended slot semantics are:

- `g_regime`:
  care setting, transition readiness, discharge/readmission hazard
- `g_acute`:
  current instability, support intensity, short-horizon deterioration
- `g_burden`:
  unresolved disease burden carried across windows
- `g_long`:
  slower residual risk and chronic consequence state

This freezes the final architectural goal without forcing the exact mechanism
today. The current implementation in `global_state.py` remains an adapter
baseline until the final latent mechanism is chosen.

### 3. Patient-internal memory

This is an exact, personalized memory bank. Its job is to retain facts that
should survive compression:

- static baseline facts available at timeline start
- durable chronic or historical facts
- sparse acute high-value events

Conceptually the internal bank is split into:

- `static`
- `persistent`
- `episodic`

### 4. Cohort-external precedent memory

This is the second memory and the main research direction.

The model should retrieve **similar prior patient states with observed futures**,
not merely similar token prefixes. Retrieval is meant to capture predictive
state similarity:

- different histories may map to similar current condition
- similar current condition should imply similar short- and medium-horizon
  future distributions under comparable intervention context

This memory is what turns the model from a long-context predictor into a
patient simulator with analogical future support.

### 5. Predictive-state principle

The theoretical framing for precedent memory is closest to a
**predictive state representation**:

- a boundary state should be judged by what it predicts about the future
- two histories may look very different in token space and still be retrieval
  neighbors if they imply similar future window dynamics
- similarity should therefore be shaped by future agreement rather than lexical
  overlap or raw token-prefix similarity

For this repo, the practical implication is:

- precedent retrieval should happen at semantic-window boundaries
- the retrieval key should encode current predictive clinical state
- the retrieved value should encode what happened next after similar states

## Frozen Latent Contract

The repo should now treat the latent as **frozen in role and interface**, even
though the exact update mechanism is still provisional.

What is frozen now:

- the latent is a **window-stepped belief state**
- the latent is updated from the canonical `WindowStatePacket`
- the latent is explicitly **gap-aware**
- the latent should support retrieval and conditioning through dedicated
  readouts rather than by serving as raw memory
- the latent should stay focused on **current condition and transition hazard**

What is not frozen yet:

- the exact recurrent/selective/continuous-time mechanism
- the exact internal slot parameterization
- whether the final implementation uses selective SSM, a learned jump-plus-drift
  module, or another mechanism inside the same family

The current working interpretation is:

```text
G_w^- = Drift(G_{w-1}^+, gap_w, meta_w)
G_w^+ = Jump(G_w^-, packet_w, type_w)
```

where:

- `packet_w` is the canonical `WindowStatePacket`
- `G_w` is the multi-slot latent state
- `Drift` models evolution across irregular gaps
- `Jump` models discrete boundary updates at semantic-window boundaries

The latent should expose three readouts:

- `z_query`:
  used for precedent retrieval together with packet and persistent-memory digest
- `z_local`:
  used for global-to-local conditioning, primarily as modulation and optionally
  compact context tokens
- `z_heads`:
  used for transition, timing, and other global predictive heads

The default information flow remains:

- patient memory influences local attention directly
- local attention emits the packet
- the latent reads the packet
- precedent retrieval uses packet + latent + persistent-memory digest
- any future direct memory-to-latent path must be gated and compressed

## Governing Refactor Decisions

The current choices for chunk/window summaries, latent-state form, and
patient-memory layout are all subordinate to the dual-memory world-model goal.

Until the precedent-memory path is in place, treat these current components as
useful precursors, not final interfaces:

- the current single-vector semantic summary is a migration bridge, not the
  final boundary-state object
- the current latent state in `global_state.py` is an adapter baseline for the
  now-frozen multi-slot jump-plus-drift latent family, not the final mechanism
- the current exact-memory module in `episodic_memory.py` is the core of
  patient-internal memory, but it should evolve into a banked design rather than
  remain one undifferentiated reservoir

The architecture should converge toward one canonical boundary object:

- a **window state packet** emitted at each semantic-window boundary

That packet is the substrate that should drive:

- latent-state updates
- patient-memory writes
- precedent-memory queries
- global-to-local conditioning for the next window

## Holistic Implementation Approach

### 1. Introduce a boundary-state packet

The local path should stop handing the global model one opaque summary vector.
Instead, it should emit a structured packet that separates:

- end-of-window state
- whole-window burden
- early-to-late change
- volatility or instability
- intervention and support summary
- physiology and measurement summary
- exact memory-write candidates

This packet should become the canonical interface between local attention and
all long-range modules.

### 2. Keep the local path event-native

The event path already exists and should remain the canonical source of:

- within-window content binding
- typed marked generation
- exact patient-memory write candidates
- window-packet emission

Chunking should remain a compute device, not the semantic object. The semantic
object is the boundary packet produced from chunk states and event states.

### 3. Refactor patient memory into explicit banks

The current exact-memory path should be retained, but formalized into three
banks:

- `static`
- `persistent`
- `episodic`

The present reservoir implementation should be treated as the initial
`episodic` core. Static and persistent paths should become explicit rather than
implicit conventions.

### 4. Add cohort-external precedent memory

The external memory should index **predictive clinical states**, not token
prefixes. Each item in the index should represent:

- a pre-window state key
- current care-regime and intervention context
- a compact future continuation summary
- optionally an exact continuation snippet

The point is to retrieve what happened after similar states, not only which
histories looked similar.

### 5. Make similarity future-aware

The precedent query space should be shaped by future agreement. The retrieval
metric should prefer states that imply similar:

- next window type
- next window timing
- short-horizon support evolution
- near-future event mixture
- medium-horizon disposition or readmission behavior

This is the key difference between precedent memory and nearest-neighbor token
matching.

### 6. Phase 3 design principles

Phase 3 should be implemented as a **full offline precedent system**, not as a
minimal placeholder:

- one index item per semantic-window boundary
- dense exact retrieval over stored boundary-state keys, with ANN acceleration as
  a later optimization
- compact future summaries plus exact continuation references on the value side
- retrieval over the full cohort, not a hand-pruned subset
- retrieval at semantic-window boundaries rather than token-level matching

Chunk-level precedent retrieval can be explored later, but it is not the
primary Phase 3 object. Token-level precedent retrieval is explicitly not the
target.

### 7. Finalize the latent only after the packet and dual memory exist

Once the packet, patient memory, and precedent memory are fixed, the latent can
be chosen cleanly as the best mechanism for compressive current-condition
dynamics. The likely final choice is still one of:

- gated recurrent update
- selective SSM / Mamba-like state
- hybrid jump-plus-drift continuous-time latent

That choice should come after the dual-memory interfaces are fixed, not before.

## Current Repo State

Already implemented:

- `build_subject_timeline()` emits `EventFrame`
- frame-native codec routing is in place
- packed precompiled storage serializes frame timelines
- the collator accepts frame timelines directly
- the collator emits token-to-event alignment, event metadata, and memory-rule metadata
- the local model composes bundle tokens into event states before local attention
- the local path now emits a canonical `WindowStatePacket` from chunk states and
  window metadata, with a temporary summary adapter into the current global path
- the patient-internal memory path is now banked as:
  - `static`
  - `persistent`
  - `episodic`
  while preserving the current exact event-write and cross-segment carry logic
- the `static` bank is now seeded from admission-anchored global demographic
  special events on the canonical event lane, with an explicit allowlist for
  admission-safe features
- the model emits event-family, event-payload, family-conditioned event-concept,
  code-conditioned numeric event-value, next-event-gap, next-window-type, and
  next-window-gap lanes
- the model supports a persistent semantic-window latent state via
  `global_context_mode=latent_state`
- the model supports patient-internal exact memory with:
  - event-exact writes
  - explicit bank routing for persistent vs episodic carry
  - auditable rule priors
  - causal retrieval
  - per-bank retrieval accounting
  - diversity-aware retention
  - retrieval-conditioned local fusion
  - ordered cross-segment carry
- the model now exports per-window boundary-state information required for
  precedent indexing:
  - `window_global_states`
  - per-bank memory retrieval context
  - per-bank boundary memory digests after writes
- Phase 3 precedent infrastructure is now in code:
  - `PrecedentIndexItem` / `PrecedentIndexStore`
  - shard-backed `FutureSnippetRef`
  - multi-horizon `FutureSummary` targets for `h1` / `h2` / `h3`
  - offline precedent-index builder in `scripts/build_precedent_index.py`
  - dense exact top-k precedent retrieval in `precedent_memory.py`
- Phase 4 predictive-state retrieval is now in code:
  - learned `PrecedentQueryHead`, `PrecedentKeyProjector`, and
    `FutureSummaryProjector`
  - online query construction from packet + latent + persistent-memory digest
  - future-summary target export on the training path
  - future-summary agreement loss
  - contrastive predictive-state retrieval loss over retrieved candidates
  - anchor-recovery auxiliary loss against stored boundary items
- Phase 5 core dual-memory decoding path is now in code:
  - `NextWindowHeader` and prompt-bearing precedent contracts in
    `world_model_contract.py`
  - stored `future_prefix_prompt` values in the offline precedent index
  - two-stage precedent retrieval:
    - boundary prior
    - header-conditioned generation prompt
  - rollout-time next-window header prediction for:
    - next-window type
    - next-window gap
    - next-window duration
    - coarse support profile
    with teacher forcing when observed next-window metadata is available
  - prompt-conditioned local fusion for the next window
- Phase 5.5 core training/objective substrate is now in code:
  - explicit `world_model_mttee` objective preset in
    `scripts/train_transformer_v1.py`
  - delayed precedent-loss curriculum in the trainer
  - loss-module support for marked-primary supervision with dense token CE as
    auxiliary rather than the preferred path
  - patient-memory aging aligned to boundary hours instead of semantic-window
    step count
- the event-native marked loss path is implemented
- the default symbolic contract prefers exact residual vocabularies or explicit
  `UNK`; production hash fallback is no longer implicit

Not yet implemented:

- full `WindowStatePacket` usage across all long-range interfaces
- implementation of the full Phase 5.5 family-level MTTE contract, especially:
  - explicit family-by-family marked-event roles
  - categorical/numeric attribute heads for medication and qualitative
    observation payloads
  - retirement of legacy RVQ / marker-token supervision from the primary
    objective path wherever they only exist for dense-token compatibility
- ANN-accelerated precedent lookup beyond the current dense exact store
- chunk-refresh precedent queries during long generated windows
- retrieval-conditioned latent updates
- final latent mechanism implementation under the frozen multi-slot
  jump-plus-drift family contract
- intervention-conditioned rollout path
- multi-step generative evaluation centered on patient simulation

## Architectural Defaults

Until evidence suggests otherwise, the working defaults are:

- local attention reads:
  - previous latent state
  - retrieved patient memory
  - current window type
- local attention emits:
  - a compressive state summary
  - sparse memory-write candidates
- patient memory influences local attention directly
- the latent reads memory only indirectly through the local summary in the
  default design
- any future direct memory-to-latent path should be gated and compressed
- the first latent state should be a static prior from baseline context, not a
  zero vector and not a history-informed state
- precedent retrieval should read:
  - `WindowStatePacket.query_token`
  - latent query readout
  - compact persistent-memory digest
  - window type and timing metadata
- the final latent family target is:
  - multi-slot hybrid jump-plus-drift

## Research Thesis

The central research question is now:

> Can a typed, marked EHR generator learn a notion of **predictive clinical
> state** that supports both exact patient-specific memory and analogical
> precedent retrieval, thereby enabling stronger generative forecasting than
> flat token autoregression or compressed-state modeling alone?

The publishable thesis is not "a better recurrent block." It is:

- one event algebra over heterogeneous EHR events
- one marked generative objective over what, when, and value
- one compressive latent for current condition
- one exact internal memory for patient-specific facts
- one external precedent memory for similar states and futures

## Phase 3 Literature Anchors

The precedent-memory design should stay grounded in the following references
and their roles:

- **Predictive Representations of State**:
  precedent similarity should be defined by future behavior rather than past
  token overlap
- **REMed**:
  retrieval over long clinical history is useful, but the precedent path here
  is generative and boundary-state based rather than discriminative
- **RAFT**:
  retrieval of similar historical futures is the closest direct precedent for
  storing compact future continuations as precedent values
- **PromptTPP**:
  event-sequence retrieval can support streaming adaptation without immediate
  weight updates
- **Memorizing Transformers**, **RETRO**, and **Titans**:
  external memory should be treated as a first-class generative conditioning
  path rather than an analysis-only add-on
- **G-Transformer**:
  rollout should be intervention-conditioned, while causal claims remain
  conservative
- **NEXTPP** and **ORA**:
  the precedent path should support typed marked generation rather than
  reducing retrieval to plain token prediction

## Immediate Priorities

1. Complete the migration from the legacy single-vector window summary to full
   `WindowStatePacket` usage across latent updates, memory interfaces, and
   retrieval.
2. Implement Phase 5.5 objective/tokenization/time-substrate harmonization
   before spending substantial LRZ budget on architecture comparisons.
3. Run short shaping experiments to validate the revised primary loss geometry
   before broad ablations:
   - core marked objective only
   - core marked objective + patient memory
   - resumed dual-memory run with delayed precedent losses
4. Evaluate the current dual-memory decoder path and compare:
   - no memory
   - patient memory only
   - precedent memory only
   - dual memory
5. Add multi-step rollout evaluation on top of the current header-conditioned
   dual-memory path.
6. Finalize the latent mechanism only after Phase 5.5 and Phase 5 clarify what
   retrieval and rollout actually demand from the frozen latent family.

## Concrete Implementation Path

### Phase 1. Canonical boundary packet

Primary files:

- `src/ehr_hier/transformer/encoder.py`
- `src/ehr_hier/transformer/aggregator.py`
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/world_model_contract.py`

Implementation:

- add a `WindowStatePacket` with fixed slot semantics
- upgrade local/chunk aggregation so it emits packet slots rather than only a
  single semantic summary
- keep a compatibility projection from packet to one summary vector so the
  current latent path keeps working during migration

### Phase 2. Banked patient memory

Primary files:

- `src/ehr_hier/transformer/episodic_memory.py`
- `src/ehr_hier/transformer/memory_rules.py`
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/collator.py`

Implementation:

- split the current exact-memory contract into static, persistent, and episodic
  banks
- keep the current learned-salience and rule-prior write path as the initial
  episodic write policy
- add bank-aware retrieval quotas and bank-aware carry across segments

### Phase 3. Cohort precedent memory

Primary files:

- `src/ehr_hier/transformer/precedent_memory.py`
- `src/ehr_hier/transformer/world_model_contract.py`
- `src/ehr_hier/transformer/aggregator.py`
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/data/precompiled_format.py`
- `src/ehr_hier/data/dataset.py`
- `scripts/build_precedent_index.py`
- `scripts/train_transformer_v1.py`

Implementation:

- define one precedent item per semantic-window boundary, anchored after window
  `w` and before generation of window `w+1`
- export canonical boundary packets from the local path so index building uses
  the same state object as model training
- define the precedent item with:
  - identity fields:
    `item_id`, `subject_id`, `trajectory_ord`, `boundary_ord`,
    `anchor_window_ord`
  - anchor metadata:
    `current_window_type_id`, `current_window_start_h`,
    `current_window_duration_h`, `gap_prev_h`
  - coarse context:
    `support_flags`, `anchor_mask_flags`
  - key tensors:
    `key_state`, `key_packet`, `key_memory`
  - value tensors:
    `future_summary_h1`, `future_summary_h2`, `future_summary_h3`
  - exact continuation reference:
    `future_snippet_ref`
- build `key_state` from:
  - `WindowStatePacket.query_token`
  - current latent state
  - compact persistent-memory digest
  - current window-type embedding
  - support/intervention flags
  - log-gap and log-duration metadata
- store values as:
  - compact multi-horizon future summaries
  - exact continuation references back into packed shards
- keep the index offline and periodically rebuilt from a frozen or EMA model
  checkpoint; online full-corpus re-encoding is not required for the paper
- use a dense key store for `key_state` and a separate value store for future
  summaries and snippet references; exact top-k is an acceptable first
  implementation, with ANN as a later optimization

#### Phase 3 future-summary targets

The precedent value side should be multi-horizon and structured:

- `h1`:
  next semantic window
- `h2`:
  next 2 windows or next 24 hours, whichever is smaller
- `h3`:
  next 4 windows or the remainder of admission capped at 7 days

Each horizon should summarize the future with targets that are learnable in
MIMIC:

- next window type
- next window gap
- next window duration
- event-family histogram
- payload histogram
- support/intervention flags
- transition flags
- event count and measurement count
- extreme-measurement count
- numeric severity summaries
- medium-horizon disposition flags
- optional readmission targets at lower priority

Exact token futures should not be the main precedent value representation.
Exact continuation snippets should be optional references, not the primary
target.

#### Phase 3 retrieval architecture

The storage layout should be:

- dense key matrix:
  one `key_state` vector per boundary item
- metadata table:
  boundary ids, regime context, support flags, and snippet references
- compact future-summary store:
  `h1`, `h2`, and `h3`
- optional exact-snippet reference store:
  shard path id, subject idx, start boundary, and stop boundary

This is preferred over:

- token-prefix nearest-neighbor retrieval
- topic-style retrieval
- fully online full-corpus latent recomputation

Phase 3 retrieval should operate at semantic-window boundaries only.

#### Phase 3 execution order

Implement Phase 3 in the following order:

1. Extend `world_model_contract.py` with:
   - `PrecedentIndexItem`
   - `FutureSummaryH1`
   - `FutureSummaryH2`
   - `FutureSummaryH3`
   - `FutureSnippetRef`
2. Extend `aggregator.py` and `model.py` so the canonical `WindowStatePacket`
   can be exported together with:
   - the current latent state
   - a compact persistent-memory digest
   - support/intervention flags
3. Add `scripts/build_precedent_index.py` to:
   - stream packed precompiled trajectories
   - replay the current model over semantic windows
   - emit one index item per boundary
   - write ANN keys, metadata, future summaries, and snippet references
4. Extend `precedent_memory.py` so it can:
   - load the offline index
   - run ANN lookup on `key_state`
   - return compact precedent readouts and optional snippet references
5. Extend `dataset.py` and `precompiled_format.py` only as needed to support
   stable snippet references and shard path ids; avoid introducing a second
   heavyweight storage family
6. Keep Phase 3 retrieval offline and non-differentiable; the learned
   future-aware metric belongs to Phase 4, not to the initial index build

### Phase 4. Predictive-state retrieval objective

Primary files:

- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/loss.py`
- `src/ehr_hier/transformer/heads.py`
- `src/ehr_hier/transformer/precedent_memory.py`
- `src/ehr_hier/transformer/world_model_contract.py`
- `scripts/train_transformer_v1.py`

Implementation:

Phase 4 should be implemented against the frozen latent contract above.

#### Phase 4 query/key contract

The retrieval query must be built from the current predictive state, not from
token history. The default query substrate is:

- `WindowStatePacket.query_token`
- latent query readout `z_query`
- compact persistent-memory digest
- current window type / care-regime metadata
- support flags
- log-gap and log-duration metadata

The first concrete implementation should add:

- a `PrecedentQueryHead`
- a `PrecedentKeyProjector`
- a `FutureSummaryProjector`

with the following roles:

- `PrecedentQueryHead`:
  maps current packet + latent + persistent-memory digest into the retrieval
  query embedding
- `PrecedentKeyProjector`:
  projects offline `key_state` vectors into the learned retrieval space
- `FutureSummaryProjector`:
  maps `h1` / `h2` / `h3` summaries into a comparable supervision space

The stored Phase 3 `key_state` remains the canonical export. Phase 4 adds a
learned retrieval space on top of that export rather than replacing the export
format itself.

#### Phase 4 target notion of similarity

Two states should be close if, under comparable current regime/intervention
context, they imply similar:

- next window type
- next window gap
- next window duration
- short-horizon support evolution
- near-future event-family mixture
- payload/modality mixture
- medium-horizon disposition behavior

This is explicitly different from:

- token-prefix similarity
- diagnosis-set overlap
- topic-style semantic similarity

#### Phase 4 losses

Phase 4 should combine four loss families:

1. **Future-summary agreement**
   - retrieved neighbors should agree on `h1`, `h2`, and `h3`
   - this is the primary shaping signal

2. **Contrastive predictive-state loss**
   - positive pairs:
     states with similar future summaries
   - hard negatives:
     states from similar current regime/support buckets but divergent futures

3. **Self-consistency / anchor recovery**
   - the current state should still recover its own or nearest stored precedent
     item under the learned query/key map
   - lower weight than future-summary agreement

4. **Optional snippet consistency**
   - exact continuation snippets should remain secondary supervision only
   - they are useful for diagnostics, not the primary metric-learning signal

#### Phase 4 negative sampling policy

Hard negatives should be chosen from states that match coarse present context
but diverge in future behavior. The default negative buckets should match on:

- current window type
- support flag profile
- coarse gap bucket

and then prefer negatives with different:

- next window type
- next-window timing
- support escalation or de-escalation
- medium-horizon disposition summary

This prevents the retrieval space from solving the task by trivial regime
partitioning alone.

#### Phase 4 training flow

The recommended first implementation path is:

1. keep the Phase 3 precedent store fixed and offline
2. build learned query embeddings online in the model
3. project stored keys into the learned retrieval space
4. compute in-batch and retrieved-candidate future-summary losses
5. keep retrieval itself non-differentiable at the store level

This preserves the offline precedent-index design while still learning a
future-aware predictive-state metric.

#### Phase 4 ablations

At minimum, compare:

- raw dense `key_state` retrieval
- learned future-aware retrieval
- learned retrieval without persistent-memory digest
- learned retrieval without latent query readout

These ablations are necessary to tell whether the latent is materially helping
retrieval or whether the packet alone is carrying the query semantics.

#### Phase 4 deliverable

Phase 4 is complete when:

- the model produces learned precedent queries online
- the query/key space is shaped by future agreement
- precedent retrieval is no longer plain nearest-neighbor over raw stored keys
- retrieval quality can be measured by future-summary agreement, not just by
  neighbor identity or token overlap

### Phase 5. Dual-memory decoding and rollout

Primary files:

- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/loss.py`
- `src/ehr_hier/transformer/precedent_memory.py`
- `src/ehr_hier/transformer/world_model_contract.py`
- `scripts/build_precedent_index.py`
- `scripts/train_transformer_v1.py`

Implementation:

Phase 5 should now be implemented around one explicit boundary-to-generation
loop rather than a generic "add retrieval to the decoder" idea.

#### Phase 5 boundary contract

After semantic window `w` ends, the canonical post-window state is:

- updated `WindowStatePacket`
- updated patient memory banks
- updated latent belief state

Generation of window `w+1` should be mediated through one explicit object:

- `NextWindowHeader`

The next-window header should carry:

- predicted next window type
- predicted gap to next window
- predicted next-window duration
- coarse support/intervention profile
- optional coarse event-family mixture prior

This header is the bridge between boundary-state reasoning and local event
generation.

#### Phase 5 end-to-next-window flow

The default causal flow should be:

1. local path ends window `w` and emits:
   - `P_w = WindowStatePacket`
   - exact memory write candidates
2. patient memory writes are applied immediately
3. latent state is updated from the packet:
   - `G_w^- = Drift(G_{w-1}^+, gap_w, meta_w)`
   - `G_w^+ = Jump(G_w^-, P_w, type_w)`
4. a first precedent query runs from the updated state to produce a
   boundary-level future prior
5. the model predicts or samples `NextWindowHeader`
6. a second precedent query runs conditioned on that chosen next-window header
7. patient memory is queried again for the next-window regime
8. the local generator for `w+1` is initialized from:
   - latent readout
   - patient-memory readout
   - precedent-memory generation readout
   - `NextWindowHeader`
9. local event generation runs until the next semantic boundary

This means precedent retrieval is not one monolithic read. It is a two-stage
process:

- boundary prior retrieval
- header-conditioned generation retrieval

#### Phase 5 division of labor

The three long-range sources should not enter the decoder identically.

Latent state:

- enters primarily as modulation
- may also expose 1-2 compact latent tokens
- is the compressive current-condition prior
- should not behave like a long token memory

Patient memory:

- enters as exact retrieval tokens or exact bank summaries
- should be queried separately across:
  - `static`
  - `persistent`
  - `episodic`
- should carry high-trust exact patient facts

Precedent memory:

- enters as analogical future guidance
- should provide both:
  - future-summary priors for boundary/global heads
  - compact future-prompt tokens for local generation
- should be treated as a softer source than patient memory

#### Phase 5 precedent readout shape

The precedent value used for generation should be multi-view, not only one
dense vector and not only raw continuation tokens.

Add a canonical precedent-generation readout:

- `summary_prior`
- `prompt_tokens`
- `candidate_weights`
- `matched_item_ids`
- `snippet_refs`

with the following roles:

- `summary_prior`:
  aggregated `h1` / `h2` / `h3` future summaries used for:
  - next-window type prediction
  - gap prediction
  - duration prediction
  - support/intervention priors
  - coarse family-mixture priors
- `prompt_tokens`:
  compact future-prefix prompt tokens used by the local generator
- `candidate_weights`:
  explicit mixture weights across retrieved precedents
- `matched_item_ids`:
  diagnostics and analysis
- `snippet_refs`:
  optional exact backreferences for later analysis or refinement

#### Phase 5 precedent query points

Precedent retrieval should not be token-level by default.

The preferred query points are:

1. boundary query:
   after window `w` ends and after latent/memory update
2. header-conditioned initialization query:
   after `NextWindowHeader` is chosen and before local generation starts
3. optional chunk refresh query:
   only for long or uncertain generated windows

The first implementation should support:

- boundary query
- header-conditioned initialization query

Chunk refresh is a later refinement, not the initial Phase 5 requirement.

#### Phase 5 index extension

Phase 3/4 precedent items already store:

- `key_state`
- `future_summary_h1`
- `future_summary_h2`
- `future_summary_h3`
- `future_snippet_ref`

Phase 5 should extend the precedent value side with:

- `future_prefix_prompt`

This should be a compact offline-computed prompt extracted from the first
future window or first future chunk after the anchor boundary. It is preferred
over raw future token splicing as the default precedent-conditioning surface.

#### Phase 5 world-model contracts

Add the following canonical objects in `world_model_contract.py`:

- `NextWindowHeader`
- `PrecedentGenerationReadout`
- `WindowGenerationConditioning`

with the intended meanings:

- `NextWindowHeader`:
  the predicted coarse plan for the next semantic window
- `PrecedentGenerationReadout`:
  the precedent-memory output used during generation
- `WindowGenerationConditioning`:
  the combined long-range conditioning package for the next local generator

#### Phase 5 model wiring

`model.py` should be refactored so that:

- precedent summary priors affect boundary/global heads directly
- header-conditioned precedent prompt retrieval happens before next-window local
  generation
- local generation receives:
  - latent modulation
  - patient-memory exact retrieval
  - precedent prompt tokens
  - `NextWindowHeader`

The local generator should therefore not receive precedent memory only as one
pooled dense vector.

#### Phase 5 training and evaluation

Training should keep the marked generative objective primary.
Precedent-conditioning losses should be secondary and usefulness-oriented.

At minimum, evaluate:

- no memory
- patient memory only
- precedent memory only
- dual memory

and compare:

- next-window header quality
- next-window event generation quality
- multi-step rollout quality

under the same intervention/context framing used elsewhere in the project.

#### Phase 5 deliverable

Phase 5 is complete when:

- the next window is generated from an explicit `NextWindowHeader`
- precedent retrieval is used in two stages:
  - boundary prior
  - header-conditioned generation prompt
- precedent memory contributes compact future-prompt tokens, not only dense
  future-summary vectors
- local generation can be run under:
  - latent only
  - latent + patient memory
  - latent + precedent memory
  - full dual memory

### Phase 5.5. Objective, tokenization, and temporal-substrate harmonization

Primary files:

- `src/ehr_hier/transformer/loss.py`
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/heads.py`
- `src/ehr_hier/transformer/event_composer.py`
- `src/ehr_hier/transformer/collator.py`
- `src/ehr_hier/data/event_frames.py`
- `src/ehr_hier/data/subject_timeline_builder.py`
- `src/ehr_hier/transformer/global_state.py`
- `src/ehr_hier/transformer/precedent_memory.py`
- `src/ehr_hier/transformer/episodic_memory.py`
- `scripts/train_transformer_v1.py`

Implementation:

This phase freezes the substrate on which all later architecture comparisons
depend. The main questions are now:

- what exactly counts as one marked event
- which losses are primary versus auxiliary
- how numeric value is attached to an event
- which time variables are canonical and which are redundant views

Phase 5.5 must be completed before broad LRZ ablations or final latent-family
selection.

#### Phase 5.5 primary objective

The default world-model objective should be a **marked time-to-event objective**
over the event-native lane, not dense token CE with marked heads treated as
secondary add-ons.

The primary loss stack should be:

- event-family CE
- event-payload CE
- family-conditioned event-concept CE
- event inter-arrival / next-event time NLL
- numeric event-value NLL for measurement-like payloads
- next-window type head
- next-window gap NLL
- next-window duration NLL
- next-window support-profile BCE

The auxiliary loss stack should be:

- dense unified token CE only as stabilization, with low weight or disabled by
  default once the marked path is stable
- legacy switched token heads only for compatibility debugging
- precedent future/contrast/anchor losses as delayed curriculum terms, not as
  cold-start primary objectives

The repo must no longer default to a regime where dense token CE is the
preferred supervisory path whenever `logits_token` exists.

#### Phase 5.5 event/value factorization contract

Every clinically meaningful emitted object must map cleanly onto one marked
event with one timestamp and one family assignment.

The intended factorization is:

- `family`:
  coarse clinical event family
- `payload`:
  measurement-like versus symbolic subtype
- `concept`:
  family-conditioned discrete identity
- `dt`:
  time to the next event on the event lane
- `value`:
  conditional numeric payload only when the payload kind is numeric

This means:

- measurement code and measurement value belong to one marked event, not two
  unrelated competing events
- diagnoses, procedures, medications, and structural/process events are marked
  events without numeric value supervision
- exact-memory writes operate over event objects, not over detached value tokens

Tokenization must therefore be re-audited so no family is forced into a
single-dense-token formulation that contradicts the marked objective.

#### Phase 5.5 tokenization audit requirements

Re-audit all event families against the marked objective:

- numeric measurements:
  confirm that code, timestamp, and value are emitted as one event object whose
  value head is conditional on the code-conditioned event representation, and
  that legacy RVQ bundle tokens are not treated as the primary semantic target
- qualitative/non-numeric measurements:
  confirm that non-numeric observations enter as explicit observation event
  frames rather than degenerate measurement-token fallbacks
- symbolic diagnoses and procedures:
  confirm that residual-vocab handling remains explicit at the event-concept
  level and does not leak objective semantics back into dense token CE
- medications:
  confirm what constitutes one medication event and which modifiers should be
  modeled as categorical or numeric attributes under that event rather than as
  separate autoregressive token targets
- structural/process events:
  confirm that boundaries and overlays remain true marked events, not only
  control tokens
- special / demographic markers:
  confirm which are event-lane objects, which are header/static-bank seeds, and
  which should never be treated as ordinary autoregressive targets

#### Phase 5.5 family-level MTTE contract

The tokenization audit should freeze the following family-by-family target
contract for the world-model objective.

- `numeric_measurement` payload:
  one marked event frame with:
  - `family = measurement`
  - `concept = measurement code`
  - `dt = next event delta on the event lane`
  - `value = one continuous standardized scalar`
  The current cVAE/RVQ bundle may remain as a compatibility encoding path, but
  it is not the primary semantic object. The primary training target is the
  event-level continuous value. At generation time, the intended long-term path
  is to generate the continuous value first and only optionally re-discretize
  it for backward-compatibility exports.

- `qualitative_observation` payload:
  one marked event frame with:
  - `family = observation`
  - `concept = observation code`
  - `dt = next event delta on the event lane`
  - `categorical attributes = observation value / interpretation / status`
  Qualitative observation fallback for non-numeric measurement must remain an
  explicit observation event frame. It should not be reduced to an opaque
  dense-token fallback. Observation value categories belong to explicit
  attribute targets under the same event, not to detached continuation tokens.

- `symbolic_code` payload for diagnoses:
  one marked event frame with:
  - `family = diagnosis`
  - `concept = resolved diagnosis concept`
  - `dt = next event delta on the event lane`
  - `value = none`
  Exact residual concepts are allowed as explicit concept targets under the
  diagnosis family head. Hash-only residual fallback is not an acceptable
  primary semantic target for the world-model regime.

- `symbolic_code` payload for procedures:
  one marked event frame with:
  - `family = procedure`
  - `concept = resolved procedure concept`
  - `dt = next event delta on the event lane`
  - `value = none`
  The same residual policy as diagnoses applies: explicit residual concepts may
  participate in the family-conditioned concept head; opaque hash fallback
  should not be a primary marked target.

- `symbolic_code` payload for medications:
  one medication event frame with:
  - `family = medication`
  - `concept = resolved medication concept`
  - `dt = next event delta on the event lane`
  - `categorical attributes = route / formulation / administration-action-like context`
  - `numeric attributes = dose / rate / duration when available`
  Medication metadata should not be collapsed into one generic scalar `value`.
  Dose, rate, and duration require typed conditional attribute heads if they
  are supervised generatively. Start/end/stop semantics should not remain
  semantically dependent on a second marker token in the primary objective;
  they should be folded into medication action attributes or promoted to an
  explicit process/structural event representation.

- `structural` / `process` payloads:
  one marked event frame with:
  - `family = structural/process`
  - `concept = explicit boundary / overlay / transition concept`
  - `dt = next event delta on the event lane`
  - `value = none`
  These remain first-class marked events because they define semantic care
  windows and intervention regime changes.

- `demographic` / static-header payloads:
  these are not ordinary autoregressive marked events for the main objective.
  They should seed static memory, admission/header context, or control tokens.
  They may remain present on the timeline for alignment/debugging, but they are
  not the target object of ordinary concept/value generation.

#### Phase 5.5 fallback and residual policy

All canonical and fallback paths must still enter the model as sensible event
frames rather than as semantically detached token leftovers.

- non-numeric measurement fallback must become
  `payload = qualitative_observation`
  with an explicit observation concept and categorical observation-value
  attributes
- residual-exact diagnosis / procedure / medication mappings must remain
  explicit concept targets within their own family heads
- residual-hash fallbacks are allowed only as a last-resort compatibility path
  and should be disabled for primary world-model shaping runs whenever possible
- explicit `UNK` concepts are preferable to opaque hash buckets when the model
  would otherwise be asked to learn a semantically meaningless marked target
- legacy secondary bundle tokens that exist only for old dense-token decoding
  must not define the primary marked-event semantics

#### Phase 5.5 inference-alignment requirement

The audit must also state how each family should be generated once the model is
fully aligned to the marked objective.

- numeric measurements:
  generate `concept`, `dt`, and continuous `value`, then optionally
  discretize/export through the old cVAE/RVQ path only if compatibility with
  legacy token views is required
- qualitative observations:
  generate observation concept plus categorical value/status attributes within
  the same event frame
- diagnoses / procedures / structural/process:
  generate only concept and timing
- medications:
  generate concept and timing first, then conditional categorical/numeric
  attributes; do not treat medication attribute generation as ordinary dense
  token continuation
- static/demographic/header objects:
  seed context and memory, not free-running event generation

The tokenization audit should end with one explicit repo-level statement:

- which families participate in marked concept prediction
- which families participate in continuous value prediction
- which families participate in categorical or typed attribute prediction
- which families are header/static metadata only
- which legacy dense-token targets remain auxiliary only

#### Phase 5.5 temporal substrate contract

Time must be made consistent across local MTTE modeling, latent drift,
patient-memory aging, precedent retrieval, and window-header prediction.

Adopt the following canonical clocks:

- `event_dt`:
  local inter-event delta on the event lane inside semantic windows
- `window_start_h`:
  absolute semantic-window start time on the patient timeline
- `window_duration_h`:
  duration of the current semantic window
- `gap_prev_h`:
  elapsed time between the previous semantic boundary and the current one
- `memory_age_h`:
  elapsed time since a patient-memory item was written, derived from boundary
  times rather than an independent clock

Use them as follows:

- local marked generation supervises only `event_dt` on the event lane
- `WindowStatePacket`, `NextWindowHeader`, and precedent keys use
  `window_start_h`, `window_duration_h`, and `gap_prev_h`
- latent drift uses boundary-level gaps, not raw token timestamps
- patient-memory age decay uses `memory_age_h`
- precedent retrieval uses boundary timing metadata only, not local token-time
  sequences directly

This phase should explicitly remove redundant time pathways where the same time
quantity is injected twice under different names without a clear role.

#### Phase 5.5 training curriculum

The default training schedule should become phased:

1. **Marked-objective warmup**
   - train the event-family / payload / concept / dt / value path
   - train next-window type / gap / duration / support heads
   - keep dense token CE low-weight or off
   - keep precedent metric losses off
2. **Patient-memory integration**
   - enable exact-memory retrieval/write with the same primary marked objective
   - verify that memory improves generation without changing the basic loss
     geometry
3. **Precedent-metric curriculum**
   - enable precedent future/contrast/anchor losses only after the marked path
     and boundary heads are stable
   - treat precedent losses as shaping terms on top of a functioning generative
     model, not as early representation-learning substitutes

This curriculum is the default answer to the identifiability problem:
retrieval similarity should be learned after the model has a sensible future
geometry, not before.

#### Phase 5.5 concrete code changes

Implement the phase in the following order:

1. In `loss.py`:
   - demote `prefer_unified_token_loss` from the default training path
   - make the marked event losses the preferred primary path
   - group losses into:
     - primary marked
     - primary boundary/header
     - auxiliary token
     - delayed precedent
2. In `train_transformer_v1.py`:
   - expose an explicit objective preset for the world-model regime
   - add curriculum controls for when precedent losses turn on and at what
     weight
   - make dense token CE opt-in or clearly low-weight under that preset
3. In `event_frames.py`, `subject_timeline_builder.py`, `collator.py`, and
   `event_composer.py`:
   - audit every family against the marked-event factorization above
   - remove or document any tokenization choices that exist only for legacy
     dense-token CE compatibility
   - make explicit which fallback paths produce:
     - numeric-measurement events
     - qualitative-observation events
     - symbolic concept events with explicit residual concepts
   - identify which medication/observation attributes require dedicated heads
     instead of being collapsed into one scalar or left implicit in token
     bundles
4. In `global_state.py`, `episodic_memory.py`, and `precedent_memory.py`:
   - align every time-dependent operation to the canonical clocks above
   - document which clock each module consumes
5. In tests:
   - add coverage proving that primary marked losses can run with dense token CE
     disabled
   - add coverage proving that numeric value supervision stays attached to
     measurement-like marked events
   - add coverage proving that boundary/header timing and memory-age timing are
     derived consistently from the same timeline substrate

#### Phase 5.5 deliverable

Phase 5.5 is complete when:

- the repo has one explicit world-model objective preset centered on marked
  event generation
- dense token CE is auxiliary rather than the default preferred path
- every event family has an auditable marked-event/tokenization role
- every canonical and fallback path enters the world-model objective as a
  sensible event frame rather than as an opaque token-only artifact
- time enters the model through one coherent substrate with no unresolved
  duplication between local generation, latent drift, memory aging, and
  precedent retrieval
- short shaping runs can be launched without ambiguity about which losses are
  meant to converge first

### Phase 6. Final latent mechanism selection

Primary files:

- `src/ehr_hier/transformer/global_state.py`
- `src/ehr_hier/transformer/model.py`

Implementation:

- keep the frozen latent family contract:
  - multi-slot hybrid jump-plus-drift
- compare candidate mechanisms for that family:
  - current gated baseline
  - selective SSM / Mamba-like update
  - explicit jump-plus-drift continuous-time mechanism
- make the decision based on dual-memory ablations, predictive-state retrieval,
  and rollout behavior, not on isolated next-token-style metrics

## Build and Test

- install deps:
  `python -m venv .venv; .venv\\Scripts\\activate; pip install -r requirements.txt`
- run tests:
  `pytest -q`

## Active LRZ Paths

- DSS base:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2`
- MEDS cohort:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort`
- meds_reader DB:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/meds_reader_db_mimiciv_20260226_033711/mimiciv.db`
- measurement artifacts:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/etl/pipeline_artifacts_20260226_033711`
- runtime deps overlay:
  `/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2/containers/runtime_pydeps`

## Fresh Start

If starting fresh, begin with:

- `AGENTS.md`
- `paper/foundation_model_v2_outline.tex`
- `paper/foundation_model_v2_refs.bib`
- `src/ehr_hier/transformer/world_model_contract.py`
- `src/ehr_hier/transformer/precedent_memory.py`
- `src/ehr_hier/data/event_frames.py`
- `src/ehr_hier/data/subject_timeline_builder.py`
- `src/ehr_hier/transformer/collator.py`
- `src/ehr_hier/transformer/model.py`
- `src/ehr_hier/transformer/global_state.py`
- `src/ehr_hier/transformer/episodic_memory.py`
