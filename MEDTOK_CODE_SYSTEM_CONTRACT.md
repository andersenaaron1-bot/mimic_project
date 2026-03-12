# MedTok Code-System Contract

This file pins the v1 contract for how MEDS/MIMIC semantic events are expected to
map into MedTok-backed diagnosis, procedure, and medication vocabularies.

This is a contract file, not a loose note. If code-system support changes, update:
- `src/ehr_hier/tokenizers/medtok_canonicalize.py`
- `src/ehr_hier/tokenizers/medtok_attr_encoder.py`
- `scripts/audit_tokenization_flow.py`
- this file

## Pinned Resolution Cascade

The active MedTok resolution order is:

1. `exact`
2. `canonicalized`
3. `parent_lookup`
4. `crosswalk_lookup`
5. `lexical_bridge`
6. `residual_exact`
7. `residual_hash`
8. `unk`

`drop` is only possible when an encoder is explicitly configured with `drop_unknowns=True`.
For v1 compressed vocabs, the preferred policy is `residual_exact` plus `drop` on the far tail;
`residual_hash` is now a legacy-compatible fallback rather than the preferred default.

## Canonical Key Formats

Current MedTok-supported systems in local artifacts:
- diagnosis: `ICD10CM//...`, `ICD9CM//...`
- procedure: `ICD10PCS//...`, `ICD9PROC//...`, `CPT//...`
- medication: `RXNORM//...`, `NDC//...`
- additional systems present in MedTok artifacts but not active in v1 semantic tokenization:
  - `SNOMED`
  - `ATC`

## Source-To-Key Matrix

| Family | Expected MEDS / MIMIC source surface | Canonical system | Target MedTok key format | v1 status | Notes |
| --- | --- | --- | --- | --- | --- |
| Diagnosis | `DIAGNOSIS//ICD//10//<code>` | ICD-10-CM | `ICD10CM//<code>` | supported | Direct canonicalization path. |
| Diagnosis | `DIAGNOSIS//ICD//9//<code>` | ICD-9-CM | `ICD9CM//<code>` | supported | Direct canonicalization path. |
| Diagnosis | raw `diagnoses_icd.icd_code` + `icd_version` | ICD-10-CM / ICD-9-CM | `ICD10CM//<code>` or `ICD9CM//<code>` | supported | Version must be retained or reconstructed upstream. |
| Diagnosis | SNOMED diagnosis surfaces | SNOMED | none in v1 | out-of-scope | Local MedTok artifacts include SNOMED globally, but v1 diagnosis tokenization does not map diagnosis SNOMED. |
| Procedure | `PROCEDURE//ICD//10//<code>` | ICD-10-PCS | `ICD10PCS//<code>` | supported | Direct canonicalization path. |
| Procedure | `PROCEDURE//ICD//9//<code>` | ICD-9-Proc | `ICD9PROC//<code>` | supported | Direct canonicalization path. |
| Procedure | `PROCEDURE//CPT//<code>` | CPT | `CPT//<code>` | supported | Direct canonicalization path. |
| Procedure | raw `procedures_icd.icd_code` + `icd_version` | ICD-10-PCS / ICD-9-Proc | `ICD10PCS//<code>` or `ICD9PROC//<code>` | supported | Version must be retained or reconstructed upstream. |
| Procedure | HCPCS surfaces | HCPCS | none in v1 | out-of-scope | Current local MedTok summary does not advertise HCPCS as an active system; do not silently claim support. |
| Procedure | ED / ICU local procedure labels with MIMIC concept-map SNOMED parents | SNOMED | bare concept code preferred, `SNOMED//<code>` accepted | supported with crosswalk | Uses `proc_itemid.csv`, `proc_datetimeevents.csv`, or parent-coded metadata to map local labels into SNOMED procedure concepts. |
| Medication | explicit `RXNORM//<code>` | RxNorm | `RXNORM//<code>` | supported | Exact or canonicalized path. |
| Medication | explicit `NDC//<code>` | NDC | `NDC//<code>` | supported | Exact or canonicalized path. |
| Medication | `MEDICATION//...` or `INFUSION...` surfaces with embedded RxNorm/NDC | RxNorm / NDC | `RXNORM//<code>` or `NDC//<code>` | supported | Extracted from the MEDS surface string. |
| Medication | event or metadata `parent_codes` that carry RxNorm/NDC | RxNorm / NDC | `RXNORM//<code>` or `NDC//<code>` | supported | Uses `parent_lookup` stage. |
| Medication | ICU medication item labels / local medication names with MIMIC concept-map RxNorm parents | RxNorm | bare concept code preferred, `RXNORM//<code>` accepted | supported with crosswalk | Uses `inputevents_to_rxnorm.csv` and optional `codes.parquet` parent metadata. |
| Medication | formulary / GSN / local medication names without RxNorm/NDC | local / formulary | none yet | crosswalk-required | Needs an explicit crosswalk artifact before claiming MedTok support. |
| Medication | lexical medication names that uniquely alias a MedTok medication token | lexical alias | vocab-specific | supported with caution | Uses `lexical_bridge`; only one-to-one aliases survive. |

## Interpretation Rules

- `supported` means the current encoder is expected to resolve the event into an explicit
  MedTok token without residual fallback in normal operation.
- `crosswalk-required` means the source is clinically in-scope, but the repo needs an
  explicit mapping artifact rather than more regex.
- `out-of-scope` means v1 should not implicitly promise support.

## Pinned Tokenization Cleanup Order

1. Freeze this MedTok code-system contract.
2. Audit exact MedTok resolution stages from the live encoder path.
3. Refactor the structural family to use real emitted structural events.
4. Keep window markers fully independent from structural-token OOVs.
5. Collapse sparse/base vocab sources into one generated spec.
6. Derive dense/runtime vocab only from the sparse spec plus observed-id compaction.
7. Remove smoke-vocab fallthrough from production tokenization.
8. Add one tokenization freeze audit gate before the first transformer run.

## External References

- MIMIC-IV diagnoses ICD docs: https://mimic.mit.edu/docs/iv/modules/hosp/diagnoses_icd/
- MIMIC-IV procedures ICD docs: https://mimic.mit.edu/docs/iv/modules/hosp/procedures_icd/
- MIMIC medication background: https://mimic.mit.edu/fhir/medication-background.html
- MIMIC medication formulary code system: https://mimic.mit.edu/fhir/CodeSystem-mimic-medication-formulary-drug-cd.html
- MIMIC medication formulary value set: https://mimic.mit.edu/fhir/ValueSet-mimic-medication-formulary-drug-cd.html
- MIMIC ED procedure types value set: https://mimic.mit.edu/fhir/ValueSet-mimic-procedure-types-ed.html
