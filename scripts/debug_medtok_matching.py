"""
Diagnose MedTok matching coverage against a meds_reader DB.

This walks events, routes them via event_router.classify_code_to_category,
canonicalizes codes, and checks whether a MedTok vocab contains any candidate.
It highlights router mismatches (e.g., raw ICD codes routed to OTHER) and
vocab misses (no candidate landed in the vocab).

Usage (PowerShell):
    $env:PYTHONPATH="."; `
    $env:MEDS_READER_DB="C:\\path\\to\\meds_reader.db"; `
    python scripts/debug_medtok_matching.py --max-events 200000

You can point to MedTok artifacts via:
  - MEDTOK_CODE2EMBEDS or --code2embeds (full code2embeddings.json)
  - MEDTOK_VOCAB_DIR or --vocab-dir (expects diag_vocab.json / proc_vocab.json / med_vocab.json)
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import meds_reader as mr

from src.ehr_hier.data.event_router import classify_code_to_category
from src.ehr_hier.data.token_types import TokenCategory
from src.ehr_hier.tokenizers.medtok_canonicalize import (
    ensure_list,
    canonicalize_diagnosis_code,
    canonicalize_procedure_code,
    canonicalize_medication_code,
    extract_icd_from_meds_code,
    diagnosis_filter,
    procedure_filter,
    medication_filter,
)
from src.ehr_hier.tokenizers.medtok_loader import (
    CategoryVocab,
    build_vocab_from_code2embeddings,
    load_medtok_vocab,
)


def _offset(manifest: Dict, key: str, default: int) -> int:
    return int(manifest.get(key, {}).get("offset", default))


def _load_manifest(fp: Path) -> Dict:
    if fp.exists():
        return json.loads(fp.read_text())
    return {}


def _resolve_vocab_path(name: str, vocab_dir: Path) -> Path:
    if name == "diagnosis":
        return vocab_dir / "diag_vocab.json"
    if name == "procedure":
        return vocab_dir / "proc_vocab.json"
    return vocab_dir / "med_vocab.json"


def _build_vocab(
    name: str,
    *,
    offset: int,
    code2embeds_path: Optional[str],
    vocab_dir: Optional[Path],
    filter_fn: Callable[[str], bool],
) -> CategoryVocab:
    if code2embeds_path:
        return build_vocab_from_code2embeddings(
            code2embeds_path, offset=offset, name=name, filter_fn=filter_fn
        )
    if vocab_dir is None:
        raise FileNotFoundError("MEDTOK_VOCAB_DIR not provided and MEDTOK_CODE2EMBEDS missing")
    fp = _resolve_vocab_path(name, vocab_dir)
    if not fp.exists():
        raise FileNotFoundError(f"Missing vocab file: {fp}")
    return load_medtok_vocab(str(fp), offset=offset, name=name)


@dataclass
class CatStats:
    filtered_total: int = 0          # events the filter_fn says belong to this category
    filtered_routed: int = 0         # of the above, how many router sent to this category
    routed_total: int = 0            # events routed here by event_router
    parent_meta_present: int = 0
    parent_lookup_present: int = 0
    parent_recovered_hits: int = 0   # misses on base candidates that become hits via parent candidates
    hits: int = 0
    misses: int = 0
    no_candidate: int = 0            # canonicalizer produced zero candidates (raw included)
    not_in_vocab: int = 0            # had candidates but none hit vocab
    format_recoverable_misses: int = 0  # diagnosis-only: miss recoverable by extra ICD dot probing
    router_mismatch: Counter = field(default_factory=Counter)   # filter hit but router chose other
    miss_samples: Counter = field(default_factory=Counter)      # routed here but missed vocab
    format_recoverable_samples: Counter = field(default_factory=Counter)


def _dedupe_preserve(seq: Iterable[str]) -> list[str]:
    seen = set()
    out = []
    for item in seq:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


_ICD_ALNUM_RE = re.compile(r"[^A-Z0-9]")


def _all_single_dot_forms(code_no_dot: str) -> list[str]:
    base = _ICD_ALNUM_RE.sub("", str(code_no_dot).upper())
    if not base:
        return []
    cands = [base]
    for i in range(1, len(base)):
        cands.append(base[:i] + "." + base[i:])
    return _dedupe_preserve(cands)


def _diagnosis_format_probe_candidates(raw_code: str) -> list[str]:
    """
    Build an aggressive ICD diagnosis candidate set to quantify whether misses are
    formatting-related (dot placement / dotted-vs-undotted) rather than true vocab gaps.
    """
    s = str(raw_code).upper()
    icd = extract_icd_from_meds_code(s)
    if not icd and "//" in s:
        icd = s.split("//")[-1]
    if not icd:
        return []

    base = _ICD_ALNUM_RE.sub("", icd.upper())
    if not base:
        return []

    forms = _all_single_dot_forms(base)
    # If ICD-9 like, also probe trimmed-leading-zero forms.
    if base[0].isdigit():
        trimmed = base.lstrip("0")
        if trimmed and trimmed != base:
            forms.extend(_all_single_dot_forms(trimmed))
    forms = _dedupe_preserve(forms)

    prefix = "ICD10CM//" if base[0].isalpha() else "ICD9CM//"
    out: list[str] = []
    for f in forms:
        out.append(f"{prefix}{f}")
        out.append(f)
    return _dedupe_preserve(out)


def _extract_parent_codes_from_event(ev: object) -> list[str]:
    out: list[str] = []

    def _append(v: object) -> None:
        if v is None:
            return
        s = str(v).strip()
        if s:
            out.append(s)

    _append(getattr(ev, "parent_code", None))
    pcs = getattr(ev, "parent_codes", None)
    if pcs is None:
        return _dedupe_preserve(out)
    if isinstance(pcs, (list, tuple, set)):
        for v in pcs:
            _append(v)
        return _dedupe_preserve(out)
    if isinstance(pcs, str):
        s = pcs.strip()
        if not s:
            return _dedupe_preserve(out)
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
        return _dedupe_preserve(out)
    _append(pcs)
    return _dedupe_preserve(out)


def _load_parent_lookup_from_codes_parquet(fp: str | None) -> Dict[str, List[str]]:
    """
    Build code -> parent_codes lookup from MEDS metadata/codes.parquet.
    Keys are normalized to uppercase for alignment with routed event codes in this script.
    """
    if not fp:
        return {}
    path = Path(fp)
    if not path.exists():
        raise FileNotFoundError(f"codes.parquet not found: {path}")
    try:
        import pyarrow.parquet as pq
    except Exception as exc:
        raise ImportError("pyarrow is required to read codes.parquet for parent lookup") from exc

    tbl = pq.read_table(str(path), columns=["code", "parent_codes"])
    codes = tbl.column("code").to_pylist()
    parents = tbl.column("parent_codes").to_pylist()
    out: Dict[str, List[str]] = {}

    def _parse_parent_cell(v: object) -> List[str]:
        if v is None:
            return []
        if isinstance(v, (list, tuple, set)):
            return _dedupe_preserve(str(x).strip() for x in v if x is not None and str(x).strip())
        if isinstance(v, str):
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
                return _dedupe_preserve(str(x).strip() for x in parsed if x is not None and str(x).strip())
            return [s]
        return [str(v).strip()] if str(v).strip() else []

    for code, parent_cell in zip(codes, parents):
        if code is None:
            continue
        key = str(code).upper()
        pcs = _parse_parent_cell(parent_cell)
        if pcs:
            out[key] = pcs
    return out


def _build_resources(manifest: Dict, args: argparse.Namespace):
    code2embeds = args.code2embeds or os.getenv("MEDTOK_CODE2EMBEDS")
    vocab_dir_env = os.getenv("MEDTOK_VOCAB_DIR", args.vocab_dir)
    vocab_dir = Path(vocab_dir_env) if vocab_dir_env else None

    diag_vocab = _build_vocab(
        "diagnosis",
        offset=_offset(manifest, "diagnosis", 1_000_000),
        code2embeds_path=code2embeds,
        vocab_dir=vocab_dir,
        filter_fn=diagnosis_filter,
    )
    proc_vocab = _build_vocab(
        "procedure",
        offset=_offset(manifest, "procedure", 1_200_000),
        code2embeds_path=code2embeds,
        vocab_dir=vocab_dir,
        filter_fn=procedure_filter,
    )
    med_vocab = _build_vocab(
        "medication",
        offset=_offset(manifest, "medication", 1_400_000),
        code2embeds_path=code2embeds,
        vocab_dir=vocab_dir,
        filter_fn=medication_filter,
    )

    cat_resources = {
        TokenCategory.DIAGNOSIS: (diag_vocab, canonicalize_diagnosis_code, diagnosis_filter),
        TokenCategory.PROCEDURE: (proc_vocab, canonicalize_procedure_code, procedure_filter),
        TokenCategory.MEDICATION: (med_vocab, canonicalize_medication_code, medication_filter),
    }
    return cat_resources


def _summarize(cat: TokenCategory, stats: CatStats, top_k: int) -> None:
    def _pct(num: int, den: int) -> float:
        return (num / den * 100.0) if den else 0.0

    print(f"\n[{cat.name}]")
    print(
        f"  filter_hits={stats.filtered_total:,} | routed={stats.routed_total:,} "
        f"| filter->routed={stats.filtered_routed:,}"
    )
    print(
        f"  parent_meta_present={stats.parent_meta_present:,} "
        f"({(_pct(stats.parent_meta_present, stats.routed_total)):.1f}% of routed)"
    )
    print(
        f"  parent_lookup_present={stats.parent_lookup_present:,} "
        f"({(_pct(stats.parent_lookup_present, stats.routed_total)):.1f}% of routed)"
    )
    print(
        f"  hits={stats.hits:,} | misses={stats.misses:,} "
        f"| hit_rate={_pct(stats.hits, stats.routed_total):.1f}%"
    )
    if stats.parent_recovered_hits:
        print(
            f"  parent_recovered_hits={stats.parent_recovered_hits:,} "
            f"({(_pct(stats.parent_recovered_hits, stats.routed_total)):.2f}% absolute hit-rate gain)"
        )
    if stats.no_candidate or stats.not_in_vocab:
        print(f"  miss breakdown: no_cand={stats.no_candidate:,}, not_in_vocab={stats.not_in_vocab:,}")
    if cat == TokenCategory.DIAGNOSIS and stats.misses:
        rec = stats.format_recoverable_misses
        print(
            f"  diagnosis format probe: recoverable={rec:,} "
            f"({(rec / stats.misses * 100.0):.1f}% of diagnosis misses)"
        )
        if stats.format_recoverable_samples:
            print("  top format-recoverable diagnosis misses:")
            for code, count in stats.format_recoverable_samples.most_common(top_k):
                print(f"    {count:6d}x {code}")
    if stats.router_mismatch:
        print("  top router mismatches (filter says this category, router chose other):")
        for code, count in stats.router_mismatch.most_common(top_k):
            print(f"    {count:6d}x {code}")
    if stats.miss_samples:
        print("  top vocab misses (routed here, no candidate in vocab):")
        for code, count in stats.miss_samples.most_common(top_k):
            print(f"    {count:6d}x {code}")


def run(args: argparse.Namespace) -> None:
    db_path = Path(args.db or os.getenv("MEDS_READER_DB", ""))
    if not db_path.exists():
        raise FileNotFoundError(f"meds_reader DB missing; pass --db or set MEDS_READER_DB (got {db_path})")

    manifest = _load_manifest(Path("artifacts/vocab_manifest.json"))
    cat_resources = _build_resources(manifest, args)
    parent_lookup = _load_parent_lookup_from_codes_parquet(args.codes_parquet)
    if parent_lookup:
        print(f"Loaded parent lookup entries: {len(parent_lookup):,}")

    stats: Dict[TokenCategory, CatStats] = {
        TokenCategory.DIAGNOSIS: CatStats(),
        TokenCategory.PROCEDURE: CatStats(),
        TokenCategory.MEDICATION: CatStats(),
    }

    # filter functions used to cross-check event_router routing
    filter_map: Dict[TokenCategory, Callable[[str], bool]] = {
        TokenCategory.DIAGNOSIS: diagnosis_filter,
        TokenCategory.PROCEDURE: procedure_filter,
        TokenCategory.MEDICATION: medication_filter,
    }

    db = mr.SubjectDatabase(str(db_path))
    total_events = 0
    subjects_seen = 0
    for sid in db:
        subjects_seen += 1
        if args.max_subjects and subjects_seen > args.max_subjects:
            break
        subj = db[int(sid)]
        for ev in subj.events:
            total_events += 1
            if args.max_events and total_events > args.max_events:
                break

            code = getattr(ev, "code", None)
            if code is None:
                continue
            code_upper = str(code).upper()

            # Which categories would keep this code according to filter_fn?
            filter_hits = []
            for cat, filt in filter_map.items():
                try:
                    if filt(code_upper):
                        filter_hits.append(cat)
                        stats[cat].filtered_total += 1
                except Exception:
                    continue

            category = classify_code_to_category(code)

            if category not in cat_resources:
                # Router did not send to a MedTok-backed category; log mismatches
                for cat in filter_hits:
                    stats[cat].router_mismatch[code_upper] += 1
                continue

            vocab, canon_fn, _ = cat_resources[category]
            stats[category].routed_total += 1
            if category in filter_hits:
                stats[category].filtered_routed += 1
            event_parent_codes = _extract_parent_codes_from_event(ev)
            if event_parent_codes:
                stats[category].parent_meta_present += 1
            lookup_parent_codes = parent_lookup.get(code_upper, [])
            if lookup_parent_codes:
                stats[category].parent_lookup_present += 1

            try:
                base_cands = ensure_list(canon_fn(code)) + [code_upper]
            except Exception:
                base_cands = [code_upper]
            base_cands = _dedupe_preserve(base_cands)

            parent_cands: list[str] = []
            for pc in _dedupe_preserve(list(event_parent_codes) + list(lookup_parent_codes)):
                try:
                    parent_cands.extend(ensure_list(canon_fn(pc)))
                except Exception:
                    pass
                parent_cands.append(str(pc).upper())
            parent_cands = _dedupe_preserve(parent_cands)

            cands = _dedupe_preserve(parent_cands + base_cands)
            if not cands:
                stats[category].misses += 1
                stats[category].no_candidate += 1
                stats[category].miss_samples[code_upper] += 1
                continue

            hit_base = any(vocab.maybe_encode(cand) is not None for cand in base_cands)
            if hit_base:
                stats[category].hits += 1
                continue

            hit = False
            for cand in cands:
                gid = vocab.maybe_encode(cand)
                if gid is not None:
                    hit = True
                    break

            if hit:
                stats[category].hits += 1
                if parent_cands:
                    stats[category].parent_recovered_hits += 1
            else:
                stats[category].misses += 1
                stats[category].not_in_vocab += 1
                stats[category].miss_samples[code_upper] += 1
                if category == TokenCategory.DIAGNOSIS:
                    probe_cands = _diagnosis_format_probe_candidates(code_upper)
                    if probe_cands and any(vocab.maybe_encode(c) is not None for c in probe_cands):
                        stats[category].format_recoverable_misses += 1
                        stats[category].format_recoverable_samples[code_upper] += 1

        if args.max_events and total_events > args.max_events:
            break

    print(f"Processed {total_events:,} events across {subjects_seen:,} subjects from {db_path}")
    for cat in (TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION):
        _summarize(cat, stats[cat], args.top_k)

    if args.output_json:
        payload = {
            "db_path": str(db_path),
            "subjects_seen": int(subjects_seen),
            "events_seen": int(total_events),
            "categories": {},
        }
        for cat in (TokenCategory.DIAGNOSIS, TokenCategory.PROCEDURE, TokenCategory.MEDICATION):
            s = stats[cat]
            payload["categories"][cat.name] = {
                "filtered_total": int(s.filtered_total),
                "filtered_routed": int(s.filtered_routed),
                "routed_total": int(s.routed_total),
                "parent_meta_present": int(s.parent_meta_present),
                "parent_lookup_present": int(s.parent_lookup_present),
                "parent_recovered_hits": int(s.parent_recovered_hits),
                "hits": int(s.hits),
                "misses": int(s.misses),
                "no_candidate": int(s.no_candidate),
                "not_in_vocab": int(s.not_in_vocab),
                "format_recoverable_misses": int(s.format_recoverable_misses),
                "top_router_mismatch": s.router_mismatch.most_common(args.top_k),
                "top_miss_samples": s.miss_samples.most_common(args.top_k),
                "top_format_recoverable": s.format_recoverable_samples.most_common(args.top_k),
            }
        out_fp = Path(args.output_json)
        out_fp.parent.mkdir(parents=True, exist_ok=True)
        out_fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Wrote JSON report to {out_fp}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect MedTok matching coverage on a meds_reader DB.")
    parser.add_argument("--db", type=str, default=None, help="Path to meds_reader database (or set MEDS_READER_DB).")
    parser.add_argument(
        "--code2embeds",
        type=str,
        default=None,
        help="Path to MedTok code2embeddings.json (overrides vocab dir if set).",
    )
    parser.add_argument(
        "--vocab-dir",
        type=str,
        default="artifacts/medtok",
        help="Directory containing diag_vocab.json / proc_vocab.json / med_vocab.json.",
    )
    parser.add_argument(
        "--max-events",
        type=int,
        default=0,
        help="Stop after this many events (0 = no limit).",
    )
    parser.add_argument(
        "--max-subjects",
        type=int,
        default=0,
        help="Stop after this many subjects (0 = no limit).",
    )
    parser.add_argument("--top-k", type=int, default=20, help="How many miss/mismatch samples to print per category.")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to write a JSON summary.")
    parser.add_argument(
        "--codes-parquet",
        type=str,
        default=None,
        help="Optional MEDS metadata/codes.parquet to supply code->parent_codes lookup.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
