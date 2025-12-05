from src.ehr_hier.tokenizers.medtok_canonicalize import (
    canonicalize_diagnosis_code,
    canonicalize_procedure_code,
)


def test_icd9_meds_prefix_diagnosis():
    cands = canonicalize_diagnosis_code("DIAGNOSIS//ICD//9//496")
    assert "ICD9CM//496" in cands
    assert "496" in cands


def test_icd10_meds_prefix_diagnosis():
    cands = canonicalize_diagnosis_code("DIAGNOSIS//ICD//10//E11.9")
    assert "ICD10CM//E11.9" in cands
    assert "E11.9" in cands


def test_procedure_icd9_prefix():
    cands = canonicalize_procedure_code("PROCEDURE//ICD//9//5491")
    assert "ICD9PROC//5491" in cands


def test_procedure_icd10_prefix():
    cands = canonicalize_procedure_code("PROCEDURE//ICD//10//0BH17EZ")
    assert "ICD10PCS//0BH17EZ" in cands


def test_procedure_cpt_prefix():
    cands = canonicalize_procedure_code("PROCEDURE//CPT//00100")
    assert "CPT//00100" in cands
