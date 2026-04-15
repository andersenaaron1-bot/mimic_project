from __future__ import annotations
import ast
import math
import re
import zlib
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Iterable, Callable

from src.ehr_hier.data.token_types import EventToken, TokenCategory
from src.ehr_hier.tokenizers.medtok_loader import CategoryVocab
from src.ehr_hier.tokenizers.attr_bins import NumericBinConfig
from src.ehr_hier.tokenizers.medtok_canonicalize import ensure_list
from src.ehr_hier.tokenizers.medication_ontology import (
    DEFAULT_MED_GROUP_OFFSET,
    MEDICATION_CODE_SYSTEM_TO_ID,
    build_medication_group_vocab,
    build_medication_semantic_descriptor,
)
from src.ehr_hier.tokenizers.medtok_crosswalk import (
    crosswalk_candidate_keys,
)

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

EXPLICIT_MEDTOK_RESOLUTION_STAGES = (
    "exact",
    "canonicalized",
    "parent_lookup",
    "crosswalk_lookup",
    "lexical_bridge",
)
NON_MEDTOK_FALLBACK_STAGES = (
    "residual_exact",
    "residual_hash",
)


@dataclass(frozen=True)
class MedTokResolution:
    base_gid: Optional[int]
    stage: str
    matched_code: Optional[str] = None
    source_code: Optional[str] = None
    group_code: Optional[str] = None
    semantic_label: Optional[str] = None
    code_system: Optional[str] = None


def _category_name(category: TokenCategory | str | object) -> str:
    if isinstance(category, TokenCategory):
        return str(category.name).upper()
    return str(category).strip().upper()


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


def normalize_residual_surface_key(raw: object) -> str:
    return str(raw).strip().upper()


def residual_fallback_candidate_codes(
    category: TokenCategory | str | object,
    *,
    base_code: Optional[str],
    raw_code: Optional[str],
) -> List[str]:
    out: List[str] = []
    seen = set()

    def _append(value: Optional[str]) -> None:
        if value is None:
            return
        norm = normalize_residual_surface_key(value)
        if not norm or norm in seen:
            return
        seen.add(norm)
        out.append(norm)

    category_name = _category_name(category)
    if category_name == "MEDICATION":
        _append(_strip_medication_action_tail(raw_code))
    _append(base_code)
    _append(raw_code)
    return out


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
        crosswalk_lookup: Optional[Dict[str, str]] = None,
        drop_unknowns: bool = False,
        fallback_to_raw: bool = True,
        residual_exact_vocab: Optional[CategoryVocab] = None,
        med_group_vocab: Optional[CategoryVocab] = None,
        med_group_offset: int = DEFAULT_MED_GROUP_OFFSET,
        residual_fallback_offset: Optional[int] = None,
        residual_fallback_buckets: int = 40_000,
        residual_tail_policy: Optional[str] = None,
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
        self.crosswalk_lookup = {
            str(k).strip().upper(): str(v)
            for k, v in (crosswalk_lookup or {}).items()
            if str(k).strip() and str(v).strip()
        }
        self.drop_unknowns = drop_unknowns
        self.fallback_to_raw = fallback_to_raw
        self.residual_exact_vocab = residual_exact_vocab
        self.med_group_offset = int(med_group_offset)
        self.residual_fallback_offset = (
            int(residual_fallback_offset)
            if residual_fallback_offset is not None
            else None
        )
        self.residual_fallback_buckets = max(1, int(residual_fallback_buckets))
        tail_policy = (
            str(residual_tail_policy).strip().lower()
            if residual_tail_policy is not None
            else "drop"
        )
        if tail_policy not in {"hash", "drop"}:
            raise ValueError(f"Unsupported residual_tail_policy={residual_tail_policy!r}")
        self.residual_tail_policy = tail_policy
        self.unk_gid = self.base_vocab.offset + self.base_vocab.unk_id
        self._cache: Dict[str, MedTokResolution] = {}
        self._lexical_bridge: Dict[str, str] = {}
        # START/END/STOP markers from MEDS-style medication/infusion/procedure events
        self._marker_attr = "event_marker"
        self._marker_to_id = {"START": 1, "END": 2, "STOP": 3}
        self.med_group_vocab: Optional[CategoryVocab] = None
        if self.category == TokenCategory.MEDICATION:
            self._lexical_bridge = self._build_medication_lexical_bridge()
            self.med_group_vocab = med_group_vocab or build_medication_group_vocab(
                med_vocab=self.base_vocab,
                residual_vocab=self.residual_exact_vocab,
                offset=int(self.med_group_offset),
            )

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
    def _dedupe_strs(values: Iterable[object]) -> List[str]:
        seen = set()
        out: List[str] = []
        for value in values:
            if value is None:
                continue
            s = str(value).strip()
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
        return out

    @staticmethod
    def _stable_bucket(raw: str, buckets: int) -> int:
        data = str(raw).encode("utf-8", errors="ignore")
        return 1 + (zlib.crc32(data) % max(1, int(buckets)))

    def _residual_exact_size(self) -> int:
        if self.residual_exact_vocab is None:
            return 0
        vals = self.residual_exact_vocab.code2id.values()
        return (max(int(v) for v in vals) + 1) if vals else 0

    def _residual_hash_offset(self) -> Optional[int]:
        if self.residual_fallback_offset is None:
            return None
        return int(self.residual_fallback_offset) + int(self._residual_exact_size())

    def _is_residual_gid(self, gid: int) -> bool:
        if self.residual_exact_vocab is not None:
            lo = int(self.residual_exact_vocab.offset)
            hi = lo + max(0, int(self._residual_exact_size()) - 1)
            if lo <= int(gid) <= hi:
                return True
        if self.residual_fallback_offset is None or self.residual_tail_policy != "hash":
            return False
        lo = int(self._residual_hash_offset() or 0)
        hi = lo + int(self.residual_fallback_buckets) - 1
        return lo <= int(gid) <= hi

    def _exact_candidates(
        self,
        base_code: Optional[str],
        raw_code: Optional[str],
    ) -> List[str]:
        return self._dedupe_strs([base_code, raw_code])

    def _canonical_candidates(
        self,
        base_code: Optional[str],
        raw_code: Optional[str],
    ) -> List[str]:
        if self.canonicalize_fn is None:
            return []
        out: List[str] = []
        seen = set()
        for src in self._dedupe_strs([base_code, raw_code]):
            for cand in ensure_list(self.canonicalize_fn(src)):
                cand_str = str(cand).strip()
                if not cand_str or cand_str in seen:
                    continue
                seen.add(cand_str)
                out.append(cand_str)
        return out

    def _parent_candidates(
        self,
        parent_codes: Optional[Iterable[str]] = None,
    ) -> List[str]:
        if not parent_codes:
            return []
        out: List[str] = []
        seen = set()
        for src in self._dedupe_strs(parent_codes):
            if src not in seen:
                seen.add(src)
                out.append(src)
            if self.canonicalize_fn is None:
                continue
            for cand in ensure_list(self.canonicalize_fn(src)):
                cand_str = str(cand).strip()
                if not cand_str or cand_str in seen:
                    continue
                seen.add(cand_str)
                out.append(cand_str)
        return out

    def _resolve_bridge(
        self,
        candidates: Iterable[str],
        *,
        seen_candidates: Optional[set[str]] = None,
    ) -> Optional[MedTokResolution]:
        if not self._lexical_bridge:
            return None
        for cand in self._dedupe_strs(candidates):
            key = _normalize_lexical_key(cand)
            if not key:
                continue
            bridged = self._lexical_bridge.get(key)
            if not bridged:
                continue
            if seen_candidates is not None and bridged in seen_candidates:
                continue
            gid = self.base_vocab.maybe_encode(bridged)
            if gid is None:
                continue
            return MedTokResolution(
                base_gid=gid,
                stage="lexical_bridge",
                matched_code=bridged,
                source_code=cand,
            )
        return None

    def _resolve_residual_exact(
        self,
        *,
        base_code: Optional[str],
        raw_code: Optional[str],
    ) -> Optional[MedTokResolution]:
        if self.residual_exact_vocab is None:
            return None
        for cand in residual_fallback_candidate_codes(
            self.category,
            base_code=base_code,
            raw_code=raw_code,
        ):
            gid = self.residual_exact_vocab.maybe_encode(cand)
            if gid is None:
                continue
            return MedTokResolution(
                base_gid=gid,
                stage="residual_exact",
                matched_code=cand,
                source_code=cand,
            )
        return None

    def _resolve_crosswalk(
        self,
        *,
        base_code: Optional[str],
        raw_code: Optional[str],
        parent_codes: Optional[Iterable[str]] = None,
        seen_candidates: Optional[set[str]] = None,
    ) -> Optional[MedTokResolution]:
        if not self.crosswalk_lookup:
            return None
        values: List[object] = []
        values.extend(self._dedupe_strs([base_code, raw_code]))
        if parent_codes:
            values.extend(self._dedupe_strs(parent_codes))
        for key in crosswalk_candidate_keys(self.category, *values):
            target = self.crosswalk_lookup.get(key)
            if not target:
                continue
            if seen_candidates is not None and target in seen_candidates:
                continue
            gid = self.base_vocab.maybe_encode(target)
            if gid is None:
                continue
            return MedTokResolution(
                base_gid=gid,
                stage="crosswalk_lookup",
                matched_code=target,
                source_code=key,
            )
        return None

    def resolve_code(
        self,
        base_code: Optional[str],
        raw_code: Optional[str],
        *,
        parent_codes: Optional[Iterable[str]] = None,
    ) -> MedTokResolution:
        if base_code is None and raw_code is None:
            if self.drop_unknowns:
                return MedTokResolution(base_gid=None, stage="drop")
            return MedTokResolution(base_gid=self.unk_gid, stage="unk")

        parent_key = "|".join(sorted(str(x) for x in (parent_codes or [])))
        cache_key = f"{base_code}||{raw_code}||{parent_key}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        seen_candidates: set[str] = set()

        def _resolve_stage(stage: str, candidates: Iterable[str]) -> Optional[MedTokResolution]:
            for cand in self._dedupe_strs(candidates):
                seen_candidates.add(cand)
                gid = self.base_vocab.maybe_encode(cand)
                if gid is None:
                    continue
                return MedTokResolution(
                    base_gid=gid,
                    stage=stage,
                    matched_code=cand,
                    source_code=cand,
                )
            return None

        exact = _resolve_stage("exact", self._exact_candidates(base_code, raw_code))
        if exact is not None:
            self._cache[cache_key] = exact
            return exact

        canonical_seen = set(seen_candidates)
        canonical = _resolve_stage(
            "canonicalized",
            [cand for cand in self._canonical_candidates(base_code, raw_code) if cand not in canonical_seen],
        )
        if canonical is not None:
            self._cache[cache_key] = canonical
            return canonical

        parent_seen = set(seen_candidates)
        parent = _resolve_stage(
            "parent_lookup",
            [cand for cand in self._parent_candidates(parent_codes) if cand not in parent_seen],
        )
        if parent is not None:
            self._cache[cache_key] = parent
            return parent

        crosswalk = self._resolve_crosswalk(
            base_code=base_code,
            raw_code=raw_code,
            parent_codes=parent_codes,
            seen_candidates=seen_candidates,
        )
        if crosswalk is not None:
            self._cache[cache_key] = crosswalk
            return crosswalk

        bridge_inputs = self._dedupe_strs(
            [
                *(self._exact_candidates(base_code, raw_code)),
                *(self._canonical_candidates(base_code, raw_code)),
                *(self._parent_candidates(parent_codes)),
            ]
        )
        bridge = self._resolve_bridge(bridge_inputs, seen_candidates=seen_candidates)
        if bridge is not None:
            self._cache[cache_key] = bridge
            return bridge

        residual_exact = self._resolve_residual_exact(
            base_code=base_code,
            raw_code=raw_code,
        )
        if residual_exact is not None:
            self._cache[cache_key] = residual_exact
            return residual_exact

        if self.drop_unknowns:
            dropped = MedTokResolution(base_gid=None, stage="drop")
            self._cache[cache_key] = dropped
            return dropped

        if self.residual_fallback_offset is not None and self.residual_tail_policy == "hash":
            seed = str(base_code if base_code is not None else raw_code)
            fallback_gid = int(self._residual_hash_offset() or 0) + self._stable_bucket(
                seed,
                int(self.residual_fallback_buckets),
            ) - 1
            residual = MedTokResolution(
                base_gid=fallback_gid,
                stage="residual_hash",
                matched_code=None,
                source_code=seed,
            )
            self._cache[cache_key] = residual
            return residual

        unk = MedTokResolution(
            base_gid=self.unk_gid,
            stage="unk",
            matched_code=self.base_vocab.unk_token,
            source_code=str(base_code if base_code is not None else raw_code),
        )
        self._cache[cache_key] = unk
        return unk

    def _resolve_event_core(self, ev: Any) -> MedTokResolution:
        raw_code = getattr(ev, "code", None)
        base_code, _ = self._strip_marker(raw_code)
        parent_codes = self._iter_parent_codes(ev)
        parent_codes.extend(self._lookup_parent_codes(raw_code=raw_code, base_code=base_code))
        parent_codes = list(dict.fromkeys(parent_codes))
        return self.resolve_code(base_code, raw_code, parent_codes=parent_codes)

    def resolve_event(self, ev: Any) -> MedTokResolution:
        resolution = self._resolve_event_core(ev)
        if self.category != TokenCategory.MEDICATION:
            return resolution
        raw_code = getattr(ev, "code", None)
        _, marker = self._strip_marker(raw_code)
        descriptor = build_medication_semantic_descriptor(
            ev=ev,
            matched_code=resolution.matched_code,
            source_code=resolution.source_code,
            resolution_stage=resolution.stage,
            marker=marker,
            med_group_vocab=self.med_group_vocab,
            categorical_attr_vocabs=self.categorical_attrs,
            numeric_attr_cfgs=self.numeric_attrs,
        )
        return MedTokResolution(
            base_gid=resolution.base_gid,
            stage=resolution.stage,
            matched_code=resolution.matched_code,
            source_code=resolution.source_code,
            group_code=descriptor.group_code,
            semantic_label=descriptor.semantic_label,
            code_system=descriptor.code_system,
        )

    def resolve_group_code(self, ev: Any) -> Optional[str]:
        return self.resolve_event(ev).group_code

    def resolve_semantic_label(self, ev: Any) -> Optional[str]:
        return self.resolve_event(ev).semantic_label

    def _encode_categorical_attrs(self, ev: Any) -> Dict[str, int]:
        cat_attrs: Dict[str, int] = {}
        for attr_name, vocab in self.categorical_attrs.items():
            val = getattr(ev, attr_name, None)
            if val is None:
                continue
            text = str(val).strip()
            if not text:
                continue
            cat_attrs[attr_name] = vocab.encode(text.upper())
        return cat_attrs

    def _encode_numeric_attrs(self, ev: Any) -> Dict[str, float]:
        num_attrs: Dict[str, float] = {}
        for attr_name, cfg in self.numeric_attrs.items():
            val = getattr(ev, attr_name, None)
            try:
                parsed = float(val)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(parsed):
                continue
            num_attrs[attr_name] = float(cfg.normalize(parsed))
        return num_attrs

    def encode_event(self, ev: Any, dt_hours: float) -> List[EventToken]:
        raw_code = getattr(ev, "code", None)
        base_code, marker = self._strip_marker(raw_code)
        parent_codes = self._iter_parent_codes(ev)
        parent_codes.extend(self._lookup_parent_codes(raw_code=raw_code, base_code=base_code))
        # de-dupe preserving order
        parent_codes = list(dict.fromkeys(parent_codes))
        resolution = self.resolve_event(ev)
        base_gid = resolution.base_gid
        if base_gid is None:
            return []
        cat_attrs = self._encode_categorical_attrs(ev)
        if self._is_residual_gid(base_gid):
            cat_attrs["residual_fallback"] = 1
            if resolution.stage == "residual_exact":
                cat_attrs["residual_fallback_exact"] = 1
            elif resolution.stage == "residual_hash":
                cat_attrs["residual_fallback_hash"] = 1
        num_attrs = self._encode_numeric_attrs(ev)
        if self.category == TokenCategory.MEDICATION:
            descriptor = build_medication_semantic_descriptor(
                ev=ev,
                matched_code=resolution.matched_code,
                source_code=resolution.source_code,
                resolution_stage=resolution.stage,
                marker=marker,
                med_group_vocab=self.med_group_vocab,
                categorical_attr_vocabs=self.categorical_attrs,
                numeric_attr_cfgs=self.numeric_attrs,
            )
            cat_attrs = {}
            for attr_name, value in descriptor.categorical_attrs.items():
                if attr_name == "event_marker_label":
                    continue
                vocab = self.categorical_attrs.get(attr_name, None)
                if vocab is not None:
                    cat_attrs[attr_name] = vocab.encode(value)
            if descriptor.group_code is not None and self.med_group_vocab is not None:
                cat_attrs["med_group"] = self.med_group_vocab.encode(descriptor.group_code)
            code_system_id = int(
                MEDICATION_CODE_SYSTEM_TO_ID.get(str(descriptor.code_system).upper(), 0)
            )
            if code_system_id > 0:
                cat_attrs["med_code_system_id"] = code_system_id
            if self._is_residual_gid(base_gid):
                cat_attrs["residual_fallback"] = 1
                if resolution.stage == "residual_exact":
                    cat_attrs["residual_fallback_exact"] = 1
                elif resolution.stage == "residual_hash":
                    cat_attrs["residual_fallback_hash"] = 1
            num_attrs = dict(descriptor.numeric_attrs)
        if marker in self._marker_to_id:
            cat_attrs[self._marker_attr] = int(self._marker_to_id[marker])

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
