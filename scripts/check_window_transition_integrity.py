#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


STRUCTURAL_PREFIXES = {
    "DISCHARGE",
    "ED_OUT",
    "ED_REGISTRATION",
    "HOSPITAL_ADMISSION",
    "HOSPITAL_DISCHARGE",
    "ICU_ADMISSION",
    "ICU_DISCHARGE",
    "MEDS_DEATH",
    "TRANSFER_TO",
}

OPENING_RELEVANT_ACTIONS = {"close_open", "open_next"}
CLOSING_RELEVANT_ACTIONS = {"close_open", "close_current"}


def _parse_csv_set(arg: str | None) -> set[str]:
    if arg is None or not str(arg).strip():
        return set()
    out: set[str] = set()
    for part in str(arg).split(","):
        part = part.strip().upper()
        if part:
            out.add(part)
    return out


def _iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _counter_to_dict(counter: Counter[Any], *, limit: int | None = None) -> Dict[str, int]:
    items = counter.most_common(limit)
    return {str(key): int(value) for key, value in items}


def _normalize_action(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text in {"", "<none>", "None"}:
        return ""
    return text


def _frame_identity(frame: Mapping[str, Any] | None) -> str:
    if not frame:
        return "<missing>"
    for key in ("source_code", "semantic_label", "concept_code", "group_code", "payload_kind"):
        value = frame.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return "<missing>"


def _frame_prefix(frame: Mapping[str, Any] | None) -> str:
    ident = _frame_identity(frame)
    if ident == "<missing>":
        return ident
    return str(ident).split("//", 1)[0].strip().upper() or "<missing>"


def _has_transition_target(frame: Mapping[str, Any] | None) -> bool:
    if not frame:
        return False
    name = frame.get("transition_window_type_name")
    ident = frame.get("transition_window_type_id")
    return bool(str(name).strip() if name is not None else False) or ident is not None


def _is_structural(frame: Mapping[str, Any] | None) -> bool:
    return _frame_prefix(frame) in STRUCTURAL_PREFIXES


def _select_transition_frames(
    frames: Sequence[Mapping[str, Any]],
    *,
    action: str,
) -> tuple[List[Mapping[str, Any]], str]:
    action = _normalize_action(action)
    if not frames:
        return [], "missing"

    exact = [frame for frame in frames if _normalize_action(frame.get("transition_action")) == action]
    if exact:
        return exact, "transition_action"

    targeted = [frame for frame in frames if _has_transition_target(frame)]
    if targeted:
        return targeted, "transition_window_type"

    structural = [frame for frame in frames if _is_structural(frame)]
    if structural:
        return structural, "structural_fallback"

    return [], "missing"


def _compact_switch_view(sw: Mapping[str, Any], *, keep_frames: int = 3) -> Dict[str, Any]:
    return {
        "switch_index": sw.get("switch_index"),
        "action": sw.get("action"),
        "from_window_type": sw.get("from_window_type"),
        "to_window_type": sw.get("to_window_type"),
        "closing_time_hours": sw.get("closing_time_hours"),
        "opening_time_hours": sw.get("opening_time_hours"),
        "closing_frames": list(sw.get("closing_frames", []))[:keep_frames],
        "opening_frames": list(sw.get("opening_frames", []))[:keep_frames],
    }


def _append_limited(store: List[Dict[str, Any]], item: Dict[str, Any], *, limit: int) -> None:
    if len(store) < int(limit):
        store.append(item)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Strictly attribute semantic window transitions to transition-candidate frames only, "
            "and summarize terminal window states."
        )
    )
    ap.add_argument("--input_jsonl", required=True)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--max_examples", type=int, default=24)
    ap.add_argument(
        "--allowed_close_open_openers",
        default="TRANSFER_TO,ICU_ADMISSION",
        help="Allowed opening-driver prefixes for close_open.",
    )
    ap.add_argument(
        "--output_json",
        default=None,
    )
    args = ap.parse_args()

    input_path = Path(args.input_jsonl)
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    allowed_close_open_openers = _parse_csv_set(args.allowed_close_open_openers)

    opening_driver_prefix_by_action: Dict[str, Counter[str]] = {}
    opening_driver_identity_by_action: Dict[str, Counter[str]] = {}
    opening_driver_mode_by_action: Dict[str, Counter[str]] = {}
    closing_driver_prefix_by_action: Dict[str, Counter[str]] = {}
    closing_driver_identity_by_action: Dict[str, Counter[str]] = {}
    closing_driver_mode_by_action: Dict[str, Counter[str]] = {}
    opening_bundle_prefixes_when_transfer_to: Counter[str] = Counter()
    terminal_window_type_counts: Counter[str] = Counter()
    terminal_window_closing_action_counts: Counter[str] = Counter()
    terminal_window_frame_prefix_counts: Counter[str] = Counter()
    close_open_nonstandard_openers: List[Dict[str, Any]] = []
    close_open_transfer_to_with_icu_examples: List[Dict[str, Any]] = []
    terminal_inpatient_examples: List[Dict[str, Any]] = []
    terminal_unknown_examples: List[Dict[str, Any]] = []

    records_scanned = 0
    switches_scanned = 0
    close_open_transfer_to_with_icu_count = 0

    for rec in _iter_records(input_path):
        records_scanned += 1
        subject_id = int(rec.get("subject_id", -1))
        window_sequence = list(rec.get("window_sequence", []))
        switches = list(rec.get("switches", []))

        if window_sequence:
            last_window = window_sequence[-1]
            terminal_type = str(last_window.get("window_type", "<missing>"))
            terminal_closing_action = str(last_window.get("closing_action") or "<none>")
            terminal_frame_prefix = _frame_prefix(last_window.get("last_frame"))
            terminal_window_type_counts[terminal_type] += 1
            terminal_window_closing_action_counts[f"{terminal_type} | {terminal_closing_action}"] += 1
            terminal_window_frame_prefix_counts[f"{terminal_type} | {terminal_frame_prefix}"] += 1
            if terminal_type == "INPATIENT":
                _append_limited(
                    terminal_inpatient_examples,
                    {
                        "subject_id": subject_id,
                        "window_count": len(window_sequence),
                        "tail_window_types": [str(w.get("window_type", "<missing>")) for w in window_sequence[-6:]],
                        "terminal_window": last_window,
                        "tail_switches": [_compact_switch_view(sw) for sw in switches[-4:]],
                    },
                    limit=int(args.max_examples),
                )
            if terminal_type == "UNK":
                _append_limited(
                    terminal_unknown_examples,
                    {
                        "subject_id": subject_id,
                        "window_count": len(window_sequence),
                        "tail_window_types": [str(w.get("window_type", "<missing>")) for w in window_sequence[-6:]],
                        "terminal_window": last_window,
                        "tail_switches": [_compact_switch_view(sw) for sw in switches[-4:]],
                    },
                    limit=int(args.max_examples),
                )

        for sw in switches:
            action = str(sw.get("action", "<missing>"))
            switches_scanned += 1

            if action in OPENING_RELEVANT_ACTIONS:
                opening_frames = list(sw.get("opening_frames", []))
                selected, mode = _select_transition_frames(opening_frames, action=action)
                driver = selected[0] if selected else None
                prefix = _frame_prefix(driver)
                identity = _frame_identity(driver)
                opening_driver_prefix_by_action.setdefault(action, Counter())[prefix] += 1
                opening_driver_identity_by_action.setdefault(action, Counter())[identity] += 1
                opening_driver_mode_by_action.setdefault(action, Counter())[mode] += 1

                if action == "close_open":
                    bundle_prefixes = sorted({_frame_prefix(frame) for frame in opening_frames if _frame_prefix(frame) != "<missing>"})
                    for bundle_prefix in bundle_prefixes:
                        opening_bundle_prefixes_when_transfer_to[bundle_prefix] += int(prefix == "TRANSFER_TO")
                    if prefix == "TRANSFER_TO" and "ICU_ADMISSION" in bundle_prefixes:
                        close_open_transfer_to_with_icu_count += 1
                        _append_limited(
                            close_open_transfer_to_with_icu_examples,
                            {
                                "subject_id": subject_id,
                                "bundle_prefixes": bundle_prefixes,
                                "switch": _compact_switch_view(sw),
                            },
                            limit=int(args.max_examples),
                        )
                    if allowed_close_open_openers and prefix not in allowed_close_open_openers:
                        _append_limited(
                            close_open_nonstandard_openers,
                            {
                                "subject_id": subject_id,
                                "opening_driver_prefix": prefix,
                                "opening_driver_identity": identity,
                                "selection_mode": mode,
                                "bundle_prefixes": bundle_prefixes,
                                "switch": _compact_switch_view(sw),
                            },
                            limit=int(args.max_examples),
                        )

            if action in CLOSING_RELEVANT_ACTIONS:
                closing_frames = list(sw.get("closing_frames", []))
                selected, mode = _select_transition_frames(closing_frames, action=action)
                driver = selected[-1] if selected else None
                prefix = _frame_prefix(driver)
                identity = _frame_identity(driver)
                closing_driver_prefix_by_action.setdefault(action, Counter())[prefix] += 1
                closing_driver_identity_by_action.setdefault(action, Counter())[identity] += 1
                closing_driver_mode_by_action.setdefault(action, Counter())[mode] += 1

    summary = {
        "input_jsonl": str(input_path),
        "records_scanned": int(records_scanned),
        "switches_scanned": int(switches_scanned),
        "allowed_close_open_openers": sorted(str(x) for x in allowed_close_open_openers),
        "opening_driver_prefix_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(opening_driver_prefix_by_action.items())
        },
        "opening_driver_identity_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(opening_driver_identity_by_action.items())
        },
        "opening_driver_selection_mode_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(opening_driver_mode_by_action.items())
        },
        "closing_driver_prefix_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(closing_driver_prefix_by_action.items())
        },
        "closing_driver_identity_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(closing_driver_identity_by_action.items())
        },
        "closing_driver_selection_mode_by_action": {
            str(action): _counter_to_dict(counter, limit=int(args.top_k))
            for action, counter in sorted(closing_driver_mode_by_action.items())
        },
        "close_open_transfer_to_opening_bundle_prefixes": _counter_to_dict(
            opening_bundle_prefixes_when_transfer_to,
            limit=int(args.top_k),
        ),
        "close_open_transfer_to_with_icu_admission_in_bundle_count": int(close_open_transfer_to_with_icu_count),
        "close_open_transfer_to_with_icu_examples": close_open_transfer_to_with_icu_examples,
        "close_open_nonstandard_openers": close_open_nonstandard_openers,
        "terminal_window_type_counts": _counter_to_dict(terminal_window_type_counts, limit=int(args.top_k)),
        "terminal_window_type_with_closing_action_counts": _counter_to_dict(
            terminal_window_closing_action_counts,
            limit=int(args.top_k),
        ),
        "terminal_window_type_with_last_frame_prefix_counts": _counter_to_dict(
            terminal_window_frame_prefix_counts,
            limit=int(args.top_k),
        ),
        "terminal_inpatient_examples": terminal_inpatient_examples,
        "terminal_unknown_examples": terminal_unknown_examples,
    }

    print(json.dumps(summary, indent=2))
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Wrote transition-check JSON to {out_path}")


if __name__ == "__main__":
    main()
