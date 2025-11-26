# src/ehr_hier/tokenizers/medtok_diag_encoder.py
import re
from dataclasses import dataclass
from typing import Optional, Dict, List
from transformers import AutoTokenizer

from src.ehr_hier.data.token_types import TokenTriplet, TokenCategory

_ICD10_RE = re.compile(r'\b([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)\b')   # ICD-10-CM (no 'U')
_ICD9_RE  = re.compile(r'\b([0-9]{3}(?:\.[0-9A-Z]{1,2})?)\b')               # ICD-9-CM

def extract_icd_from_meds_code(code_str: str) -> Optional[str]:
    """
    Try to recover a clean ICD-10/ICD-9 code from a MEDS code string.
    Returns e.g. 'E11.9' or '250.00' if found, else None.
    """
    if not isinstance(code_str, str):
        return None
    s = code_str.upper().replace('ICD-10', 'ICD10').replace('ICD-9', 'ICD9')
    # Prefer explicit markers if present
    m = re.search(r'ICD10\w*[:/\\|-]*([A-TV-Z][0-9][0-9A-Z](?:\.[0-9A-Z]{1,4})?)', s)
    if m: return m.group(1)
    m = re.search(r'ICD9\w*[:/\\|-]*([0-9]{3}(?:\.[0-9A-Z]{1,2})?)', s)
    if m: return m.group(1)
    # Generic fallbacks
    m = _ICD10_RE.search(s)
    if m: return m.group(1)
    m = _ICD9_RE.search(s)
    if m: return m.group(1)
    return None

@dataclass
class MedTokDiagnosisConfig:
    offset: int = 1_000_000       # put DIAG tokens far from measurement range
    unk_global_id: Optional[int] = None  # if you want a stable UNK; else drop unknowns
    trust_remote_code: bool = True
    hf_id: str = "mims-harvard/MedTok"

class MedTokDiagnosisEncoder:
    category = TokenCategory.DIAGNOSIS

    def __init__(self, cfg: MedTokDiagnosisConfig):
        self.cfg = cfg
        self.tok = AutoTokenizer.from_pretrained(cfg.hf_id, trust_remote_code=cfg.trust_remote_code)
        self.cache: Dict[str, Optional[int]] = {}

    def reset_state(self) -> None:
        # stateless across subjects
        return None

    def _lookup_id(self, icd: str) -> Optional[int]:
        if icd in self.cache:
            return self.cache[icd]
        ids = self.tok.encode(icd)
        if isinstance(ids, list) and len(ids) > 0:
            gid = self.cfg.offset + int(ids[0])
        else:
            gid = self.cfg.unk_global_id  # could be None
        self.cache[icd] = gid
        return gid

    def encode_event(self, ev, dt_hours: float):
        code = getattr(ev, "code", None)
        if not code:
            return []
        icd = extract_icd_from_meds_code(str(code))
        if not icd:
            return [] if self.cfg.unk_global_id is None else [TokenTriplet(value_id=self.cfg.unk_global_id,
                                                                           category_id=int(self.category),
                                                                           dt_hours=float(dt_hours))]
        gid = self._lookup_id(icd)
        if gid is None:
            return []
        return [TokenTriplet(value_id=gid, category_id=int(self.category), dt_hours=float(dt_hours))]
