from __future__ import annotations
import ast
import re
import zlib
from typing import Dict, List, Any, Optional, Iterable, Callable

from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig
from src.ehr_hier.tokenizers.medtok_canonicalize import ensure_list

_LEX_NORM_RE = re.compile(r"[^A-Z0-9]+")
_MED_ACTION_SUFFIXES = {
    "ADMINISTERED",
    "NOT ADMINISTERED",
    "NOT GIVEN",
    "NOT GIVEN PER SLIDING SCALE",
    "CONFIRMED",
    "FLUSHED",
    "NOT FLUSHED",
    "STARTED",
    "STOPPED",
    "HELD",
    "REFUSED",
    "PAUSED",
    "RESUMED",
}


def _normalize_lexical_key(raw: object) -> str:
    s = _LEX_NORM_RE.sub("", str(raw).upper())
    # Avoid unstable tiny/mostly-numeric aliases.
    if len(s) < 4:
        return ""
    if not any(ch.isalpha() for ch in s):
        return ""
    return s


def _strip_medication_action_tail(raw: object) -> Optional[str]:
    parts = [p.strip() for p in str(raw).split("//")]
    if not parts:
        return None
    if parts[0].upper() != "MEDICATION":
        return None
    if len(parts) >= 4 and parts[1].upper() in {"START", "END", "STOP"}:
        return f"MEDICATION//{'//'.join(parts[2:])}"
    if len(parts) >= 3 and parts[-1].upper() in _MED_ACTION_SUFFIXES:
        return f"MEDICATION//{'//'.join(parts[1:-1])}"
    return None


def load_parent_lookup_from_codes_parquet(codes_parquet_fp: str) -> Dict[str, List[str]]:
    """
    Load code -> parent_codes from MEDS metadata/codes.parquet.
    Returns uppercase raw-code keys to match event.code normalization.
    """
    if not codes_parquet_fp:
        return {}
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise ImportError("pyarrow is required to read codes.parquet parent_codes lookup") from exc

    tbl = pq.read_table(str(codes_parquet_fp), columns=["code", "parent_codes"])
    codes = tbl.column("code").to_pylist()
    parents = tbl.column("parent_codes").to_pylist()
    out: Dict[str, List[str]] = {}

    def _parse_parent_cell(v: object) -> List[str]:
        vals: List[str] = []

        def _append(x: object) -> None:
            if x is None:
                return
            s = str(x).strip()
            if s:
                vals.append(s)

        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            for x in v:
                _append(x)
        elif isinstance(v, str):
            s = v.strip()
            if not s:
                return []
            parsed = None
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = None
            if isinstance(parsed, (list, tuple, set)):
                for x in parsed:
                    _append(x)
            else:
                _append(s)
        else:
            _append(v)

        # de-dupe preserving order
        seen = set()
        uniq: List[str] = []
        for x in vals:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    for code, parent_cell in zip(codes, parents):
        if code is None:
            continue
        key = str(code).upper()
        pcs = _parse_parent_cell(parent_cell)
        if pcs:
            out[key] = pcs
    return out


class MedTokenWithAttrsEncoder:
    """
    Emits a single EventToken with MedTok base id and bundled metadata.
    - base token id comes from MedTok vocab (with canonicalization fallback)
    - categorical metadata stored under EventToken.cat_attrs
    - numeric metadata stored under EventToken.num_attrs (normalized to [0,1])
    """

    def __init__(
        self,
        category: TokenCategory,
        base_vocab: CategoryVocab,
        *,
        categorical_attrs: Dict[str, CategoryVocab] | None = None,
        numeric_attrs: Dict[str, NumericBinConfig] | None = None,
        canonicalize_fn: Optional[Callable[[Any], Iterable[str]]] = None,
        parent_lookup: Optional[Dict[str, List[str]]] = None,
        drop_unknowns: bool = False,
        fallback_to_raw: bool = True,
        residual_fallback_offset: Optional[int] = None,
        residual_fallback_buckets: int = 40_000,
    ):
        self.category = category
        self.base_vocab = base_vocab
        self.categorical_attrs = categorical_attrs or {}
        self.numeric_attrs = numeric_attrs or {}
        self.canonicalize_fn = canonicalize_fn
        self.parent_lookup = {
            str(k).upper(): list(v)
            for k, v in (parent_lookup or {}).items()
            if v
        }
        self.drop_unknowns = drop_unknowns
        self.fallback_to_raw = fallback_to_raw
        self.residual_fallback_offset = (
            int(residual_fallback_offset)
            if residual_fallback_offset is not None
            else None
        )
        self.residual_fallback_buckets = max(1, int(residual_fallback_buckets))
        self.unk_gid = self.base_vocab.offset + self.base_vocab.unk_id
        self._cache: Dict[str, Optional[int]] = {}
        self._lexical_bridge: Dict[str, str] = {}
        # START/END/STOP markers from MEDS-style medication/infusion/procedure events
        self._marker_attr = "event_marker"
        self._marker_to_id = {"START": 1, "END": 2, "STOP": 3}
        if self.category == TokenCategory.MEDICATION:
            self._lexical_bridge = self._build_medication_lexical_bridge()

    def reset_state(self) -> None:
        return None  # stateless

    def _strip_marker(self, raw_code: Optional[str]) -> tuple[Optional[str], Optional[str]]:
        """
        Remove MEDS-style start/stop markers while remembering the marker type.
        Examples:
            MEDICATION//START//FUROSEMIDE   -> (MEDICATION//FUROSEMIDE, START)
            INFUSION_END//225158            -> (INFUSION//225158, END)
            PROCEDURE//STOP//XYZ            -> (PROCEDURE//XYZ, STOP)
        """
        if raw_code is None:
            return None, None
        s = str(raw_code)
        upper = s.upper()

        # Prefix forms
        if upper.startswith("INFUSION_START//"):
            suffix = s[len("INFUSION_START//") :]
            return f"INFUSION//{suffix}", "START"
        if upper.startswith("INFUSION_END//"):
            suffix = s[len("INFUSION_END//") :]
            return f"INFUSION//{suffix}", "END"

        # Infix markers
        for marker in ("//START//", "//END//", "//STOP//"):
            if marker in upper:
                # strip marker in a case-insensitive way
                pattern = re.compile(re.escape(marker), re.IGNORECASE)
                base = pattern.sub("//", s, count=1)
                label = marker.strip("/").upper()
                return base, label
        return s, None

    def _iter_parent_codes(self, ev: Any) -> List[str]:
        """
        Extract parent code candidates from event metadata when available.
        Accepts:
          - ev.parent_code: scalar/string
          - ev.parent_codes: iterable or serialized iterable string
        """
        out: List[str] = []

        def _append(v: Any) -> None:
            if v is None:
                return
            s = str(v).strip()
            if s:
                out.append(s)

        parent_code = getattr(ev, "parent_code", None)
        _append(parent_code)

        parent_codes = getattr(ev, "parent_codes", None)
        if parent_codes is None:
            return list(dict.fromkeys(out))

        if isinstance(parent_codes, (list, tuple, set)):
            for v in parent_codes:
                _append(v)
            return list(dict.fromkeys(out))

        if isinstance(parent_codes, str):
            s = parent_codes.strip()
            if not s:
                return list(dict.fromkeys(out))
            parsed = None
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = ast.literal_eval(s)
                except Exception:
                    parsed = None
            if isinstance(parsed, (list, tuple, set)):
                for v in parsed:
                    _append(v)
            else:
                _append(s)
            return list(dict.fromkeys(out))

        _append(parent_codes)
        return list(dict.fromkeys(out))

    def _lookup_parent_codes(
        self,
        *,
        raw_code: Optional[str],
        base_code: Optional[str],
    ) -> List[str]:
        out: List[str] = []
        keys: List[Optional[str]] = [raw_code, base_code]
        if raw_code is not None:
            keys.append(_strip_medication_action_tail(raw_code))
        for key in keys:
            if key is None:
                continue
            out.extend(self.parent_lookup.get(str(key).upper(), []))
        # de-dupe preserving order
        seen = set()
        uniq: List[str] = []
        for x in out:
            if x in seen:
                continue
            seen.add(x)
            uniq.append(x)
        return uniq

    def _build_medication_lexical_bridge(self) -> Dict[str, str]:
        """
        Build a conservative one-to-one normalized alias -> vocab code map.
        Ambiguous aliases are discarded.
        """
        alias_hits: Dict[str, set[str]] = {}
        for code in self.base_vocab.code2id.keys():
            if code == self.base_vocab.unk_token:
                continue
            raw = str(code)
            parts = [p.strip() for p in raw.split("//") if p.strip()]
            aliases = [raw]
            if parts:
                aliases.append(parts[-1])
                head = parts[0].upper()
                if head == "MEDICATION":
                    stripped = _strip_medication_action_tail(raw)
                    if stripped:
                        aliases.append(stripped)
                    if len(parts) >= 2:
                        aliases.append(parts[1])
                elif head == "INFUSION" and len(parts) >= 2:
                    aliases.append(parts[1])

            for alias in aliases:
                key = _normalize_lexical_key(alias)
                if not key:
                    continue
                alias_hits.setdefault(key, set()).add(raw)

        out: Dict[str, str] = {}
        for key, targets in alias_hits.items():
            if len(targets) == 1:
                out[key] = next(iter(targets))
        return out

    @staticmethod
    def _stable_bucket(raw: str, buckets: int) -> int:
        data = str(raw).encode("utf-8", errors="ignore")
        return 1 + (zlib.crc32(data) % max(1, int(buckets)))

    def _is_residual_gid(self, gid: int) -> bool:
        if self.residual_fallback_offset is None:
            return False
        lo = int(self.residual_fallback_offset)
        hi = lo + int(self.residual_fallback_buckets)
        return lo <= int(gid) <= hi

    def _candidate_codes(
        self,
        base_code: Optional[str],
        raw_code: Optional[str],
        *,
        parent_codes: Optional[Iterable[str]] = None,
    ) -> List[str]:
        if base_code is None and raw_code is None:
            return []
        candidates: List[str] = []

        def _extend_for(src: Optional[str]) -> None:
            if src is None:
                return
            if self.canonicalize_fn:
                canonicalized = self.canonicalize_fn(src)
                candidates.extend(ensure_list(canonicalized))
            if self.fallback_to_raw:
                candidates.append(str(src))

        # Prefer parent ontology link if present, then regular code path.
        for pc in list(parent_codes or []):
            _extend_for(pc)
        canon_input = base_code if base_code is not None else raw_code
        _extend_for(canon_input)
        if self.fallback_to_raw and raw_code is not None and raw_code != canon_input:
            candidates.append(str(raw_code))

        # dedupe while preserving order
        seen = set()
        uniq = []
        for c in candidates:
            if c in seen:
                continue
            uniq.append(c)
            seen.add(c)

        if self._lexical_bridge:
            for c in list(uniq):
                key = _normalize_lexical_key(c)
                if not key:
                    continue
                bridged = self._lexical_bridge.get(key)
                if bridged and bridged not in seen:
                    uniq.append(bridged)
                    seen.add(bridged)
        return uniq

    def _encode_base(
        self,
        base_code: Optional[str],
        raw_code: Optional[str],
        *,
        parent_codes: Optional[Iterable[str]] = None,
    ) -> Optional[int]:
        if base_code is None and raw_code is None:
            return None if self.drop_unknowns else self.unk_gid

        parent_key = "|".join(sorted(str(x) for x in (parent_codes or [])))
        cache_key = f"{raw_code}||{parent_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        for cand in self._candidate_codes(base_code, raw_code, parent_codes=parent_codes):
            gid = self.base_vocab.maybe_encode(cand)
            if gid is not None:
                self._cache[cache_key] = gid
                return gid

        if self.drop_unknowns:
            self._cache[cache_key] = None
            return None

        if self.residual_fallback_offset is not None:
            seed = str(base_code if base_code is not None else raw_code)
            fallback_gid = int(self.residual_fallback_offset) + self._stable_bucket(
                seed,
                int(self.residual_fallback_buckets),
            )
            self._cache[cache_key] = fallback_gid
            return fallback_gid

        self._cache[cache_key] = self.unk_gid
        return self.unk_gid

    def _encode_categorical_attrs(self, ev: Any) -> Dict[str, int]:
        cat_attrs: Dict[str, int] = {}
        for attr_name, vocab in self.categorical_attrs.items():
            val = getattr(ev, attr_name, None)
            cat_attrs[attr_name] = vocab.encode(val)
        return cat_attrs

    def _encode_numeric_attrs(self, ev: Any) -> Dict[str, float]:
        num_attrs: Dict[str, float] = {}
        for attr_name, cfg in self.numeric_attrs.items():
            val = getattr(ev, attr_name, None)
            num_attrs[attr_name] = cfg.normalize(val)
        return num_attrs

    def encode_event(self, ev: Any, dt_hours: float) -> List[EventToken]:
        raw_code = getattr(ev, "code", None)
        base_code, marker = self._strip_marker(raw_code)
        parent_codes = self._iter_parent_codes(ev)
        parent_codes.extend(self._lookup_parent_codes(raw_code=raw_code, base_code=base_code))
        # de-dupe preserving order
        parent_codes = list(dict.fromkeys(parent_codes))
        base_gid = self._encode_base(base_code, raw_code, parent_codes=parent_codes)
        if base_gid is None:
            return []
        cat_attrs = self._encode_categorical_attrs(ev)
        if self._is_residual_gid(base_gid):
            cat_attrs["residual_fallback"] = 1
        num_attrs = self._encode_numeric_attrs(ev)

        tokens = [
            EventToken(
                value_id=base_gid,
                category_id=int(self.category),
                t_from_start_hours=0.0,
                dt_from_prev_hours=float(dt_hours),
                cat_attrs=cat_attrs,
                num_attrs=num_attrs,
            )
        ]

        if marker in self._marker_to_id:
            marker_id = self._marker_to_id[marker]
            tokens.append(
                EventToken(
                    value_id=self.unk_gid,  # generic marker token in the same category band
                    category_id=int(self.category),
                    t_from_start_hours=0.0,
                    dt_from_prev_hours=0.0,  # immediately after base token
                    cat_attrs={self._marker_attr: marker_id},
                    num_attrs={},
                )
            )

        return tokens
