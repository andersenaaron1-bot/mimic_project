# Structural Contract V1

This note records the cleanup of structural tokenization/window sources for the
tokenization-v1 contract.

## Side-By-Side

| Source | Role | Active in live builder/runtime? | Notes |
| --- | --- | --- | --- |
| `configs/data/structural_codes.yaml` | Human-edited structural source of truth | Yes | Defines `structural_map`, `window_boundary_labels`, `transition_map`, `window_types`, `window_type_map`, `soft_signifiers`, and `keep_original`. |
| `configs/data/transition_event_candidates.yaml` | Descriptive registry from an earlier audit phase | No | Removed from the live contract. It did not drive tokenization or runtime vocab generation. |
| `src/ehr_hier/data/subject_timeline_builder.py` | Applies the structural contract while building `EventToken` timelines | Yes | With a structural codebook present, transition/window behavior now comes from the codebook only. Legacy boundary-prefix fallback remains only for no-codebook compatibility. |
| `artifacts/token_vocab_sparse_v1.json` | Generated sparse/base token contract | Yes | Now embeds a serialized `structural_contract` summary so sparse/dense vocab generation can inspect the live structural policy from one generated artifact. |

## Final v1 Rule

The structural semantics used by tokenization are:

1. `configs/data/structural_codes.yaml` is the only human-edited structural source.
2. `build_sparse_vocab_contract.py` snapshots that source into `structural_contract`
   inside the generated sparse vocab artifact.
3. Dense/runtime vocab generation derives from the sparse artifact, not from separate
   legacy manifests.

The causal window-typing rule for v1 is:

- window type comes from the opening transition only
- if `TRANSFER_TO` is present in an opening bundle, it is the authoritative opener
- the leading pre-transition segment can default to `PROLOGUE`
- post-discharge follow-up tokens belong to a causal `POST_DISCHARGE` window until the next opener
- no `TERMINAL` or `INTER_ADMISSION` window type is synthesized in the live v1 path
- discharge and death are closing events only; they end the current window but do not type a new one
- post-discharge windows are created only when tokens are actually observed after the closer

## Builder-Specific Policy

When a structural codebook is present:

- transition boundary behavior is driven by `transition_map`
- structural codebook hits use `is_window_boundary()`
- routed structural raw prefixes such as `TRANSFER_TO` still receive transition metadata
  via the codebook prefix lookup
- legacy hard-coded boundary prefixes are *not* consulted

The legacy prefix fallback is retained only for code paths that do not pass a
structural codebook at all.
