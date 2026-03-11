from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch
import yaml

from src.ehr_hier.data.structural_codes import (
    load_structural_codebook_yaml,
    structural_surface_vocab_codes,
)


DEFAULT_SPARSE_VOCAB_JSON = "artifacts/token_vocab_sparse_v1.json"
DEFAULT_RUNTIME_LANE_ORDER: tuple[str, ...] = (
    "logits_struct",
    "logits_rvq",
    "logits_meas",
    "logits_medtok",
)
DEFAULT_RUNTIME_BLOCK_ORDER: tuple[str, ...] = (
    "special",
    "structural",
    "measurement_value",
    "measurement_code",
    "observation_code",
    "observation_value",
    "diagnosis",
    "diagnosis_residual",
    "procedure",
    "procedure_residual",
    "medication",
    "medication_residual",
)


def _load_json(path: str | Path) -> Dict[str, Any]:
    fp = Path(path)
    payload = json.loads(fp.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object at {fp}, got {type(payload)}")
    return payload


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    fp = Path(path)
    payload = yaml.safe_load(fp.read_text(encoding="utf-8"))
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise TypeError(f"Expected YAML object at {fp}, got {type(payload)}")
    return payload


def _safe_dict(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else {}


def _vocab_size_from_json(vocab_fp: str | Path | None) -> Optional[int]:
    if vocab_fp is None:
        return None
    fp = Path(vocab_fp)
    if not fp.exists():
        return None
    payload = _load_json(fp)
    vals = []
    for value in payload.values():
        try:
            vals.append(int(value))
        except Exception:
            continue
    if not vals:
        return None
    return max(vals) + 1


def _code2id_size(code2id_pt: str | Path | None) -> Optional[int]:
    if code2id_pt is None:
        return None
    fp = Path(code2id_pt)
    if not fp.exists():
        return None
    payload = torch.load(str(fp), map_location="cpu")
    if not isinstance(payload, dict):
        return None
    vals = []
    for value in payload.values():
        try:
            vals.append(int(value))
        except Exception:
            continue
    if not vals:
        return None
    return max(vals) + 1


def _rvq_size_from_ckpt(tokenizer_ckpt: str | Path | None) -> Optional[int]:
    if tokenizer_ckpt is None:
        return None
    fp = Path(tokenizer_ckpt)
    if not fp.exists():
        return None
    payload = torch.load(str(fp), map_location="cpu")
    if not isinstance(payload, dict):
        return None
    cfg = _safe_dict(payload.get("cfg", {}))
    num_codebooks = int(cfg.get("num_codebooks", 0))
    codebook_size = int(cfg.get("codebook_size", 0))
    if num_codebooks <= 0 or codebook_size <= 0:
        return None
    return int(num_codebooks * codebook_size)


def _family_entry(
    *,
    offset: int,
    source_size: int,
    family_type: str,
    runtime_head: Optional[str] = None,
    preserve_full_default: bool = False,
    runtime_mode: Optional[str] = None,
    meta: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "offset": int(offset),
        "source_size": int(max(0, source_size)),
        "family_type": str(family_type),
        "runtime_head": (str(runtime_head) if runtime_head else None),
        "preserve_full_default": bool(preserve_full_default),
    }
    if runtime_head is not None:
        payload["runtime_mode"] = str(runtime_mode or "identity")
    if meta:
        payload["meta"] = dict(meta)
    return payload


def load_sparse_vocab_contract(path: str | Path) -> Dict[str, Any]:
    payload = _load_json(path)
    families = _safe_dict(payload.get("families", {}))
    if not families:
        raise ValueError(f"sparse vocab contract missing non-empty 'families': {path}")
    return payload


def build_sparse_vocab_contract(
    *,
    tokenization_contract: str | Path = "configs/data/tokenization_v1.yaml",
    vocab_manifest: str | Path = "artifacts/vocab_manifest.json",
    structural_yaml: str | Path = "configs/data/structural_codes.yaml",
    medtok_vocab_dir: str | Path | None = None,
    medtok_attr_dir: str | Path | None = "artifacts/medtok_attrs",
    code2id_pt: str | Path | None = None,
    tokenizer_ckpt: str | Path | None = None,
    measurement_code_size: Optional[int] = None,
    rvq_size: Optional[int] = None,
) -> Dict[str, Any]:
    contract = _load_yaml(tokenization_contract)
    legacy_manifest = _load_json(vocab_manifest) if Path(vocab_manifest).exists() else {}
    frozen = _safe_dict(contract.get("frozen_ranges", {}))
    residual_cfg = _safe_dict(contract.get("residual_fallback", {}))
    markers_cfg = _safe_dict(contract.get("window_markers", {}))
    segmentation_cfg = _safe_dict(contract.get("window_segmentation", {}))

    def _offset(name: str, default: int) -> int:
        fr = _safe_dict(frozen.get(name, {}))
        if "offset" in fr:
            return int(fr["offset"])
        legacy = _safe_dict(legacy_manifest.get(name, {}))
        if "offset" in legacy:
            return int(legacy["offset"])
        return int(default)

    def _bucket(name: str, default: int) -> int:
        fr = _safe_dict(frozen.get(name, {}))
        if "buckets" in fr:
            return int(fr["buckets"])
        return int(residual_cfg.get("buckets", default))

    def _size_from_gap(lo: int, hi: int, default: int) -> int:
        gap = int(hi) - int(lo)
        return int(gap) if gap > 0 else int(default)

    special_offset = _offset("special", 0)
    special_reserved_max = int(_safe_dict(frozen.get("special", {})).get("reserved_max_id", 99))
    marker_type_offset = int(markers_cfg.get("type_token_offset", 10))
    marker_num_types = int(markers_cfg.get("num_types", 16))
    marker_end = int(markers_cfg.get("end_token_id", marker_type_offset + marker_num_types))
    marker_continue = int(markers_cfg.get("continue_token_id", marker_end + 1))
    special_size = max(
        int(special_reserved_max) + 1,
        int(marker_type_offset + marker_num_types),
        int(marker_end) + 1,
        int(marker_continue) + 1,
    )

    diag_offset = _offset("diagnosis", 1_000_000)
    diag_res_offset = _offset("diagnosis_residual", 1_160_000)
    proc_offset = _offset("procedure", 1_200_000)
    proc_res_offset = _offset("procedure_residual", 1_360_000)
    med_offset = _offset("medication", 1_400_000)
    med_res_offset = _offset("medication_residual", 1_800_000)
    med_route_offset = _offset("med_route", 1_600_000)
    med_form_offset = _offset("med_form", 1_620_000)
    med_freq_offset = _offset("med_freq", 1_640_000)
    med_unit_offset = _offset("med_unit", 1_660_000)
    med_dosage_offset = _offset("med_dosage", 1_680_000)
    med_rate_offset = _offset("med_rate", 1_700_000)
    med_duration_offset = _offset("med_duration", 1_720_000)
    meas_code_offset = _offset("measurement_code", 2_000_000)
    meas_val_offset = _offset("measurement_value", 2_100_000)
    structural_offset = _offset("structural", 2_200_000)
    obs_code_offset = _offset("observation_code", 2_300_000)
    obs_val_offset = _offset("observation_value", 2_320_000)

    medtok_dir = Path(medtok_vocab_dir) if medtok_vocab_dir is not None else None
    medtok_attr = Path(medtok_attr_dir) if medtok_attr_dir is not None else None

    diag_size = _vocab_size_from_json((medtok_dir / "diag_vocab.json") if medtok_dir is not None else None) or 20_000
    proc_size = _vocab_size_from_json((medtok_dir / "proc_vocab.json") if medtok_dir is not None else None) or 10_000
    med_size = _vocab_size_from_json((medtok_dir / "med_vocab.json") if medtok_dir is not None else None) or 25_000
    med_route_size = _vocab_size_from_json((medtok_attr / "route_vocab.json") if medtok_attr is not None else None) or 0
    med_form_size = _vocab_size_from_json((medtok_attr / "form_vocab.json") if medtok_attr is not None else None) or 0
    med_freq_size = _vocab_size_from_json((medtok_attr / "freq_vocab.json") if medtok_attr is not None else None) or 0
    med_unit_size = _vocab_size_from_json((medtok_attr / "unit_vocab.json") if medtok_attr is not None else None) or 0
    meas_code_size_eff = (
        int(measurement_code_size)
        if measurement_code_size is not None
        else (_code2id_size(code2id_pt) or 10_000)
    )
    rvq_size_eff = int(rvq_size) if rvq_size is not None else (_rvq_size_from_ckpt(tokenizer_ckpt) or 1_024)
    obs_code_size = _size_from_gap(obs_code_offset, obs_val_offset, 20_000)
    obs_val_size = _size_from_gap(obs_val_offset, _offset("structural_action", 2_400_000), 80_000)
    try:
        structural_codebook = load_structural_codebook_yaml(str(structural_yaml), default_offset=int(structural_offset))
        structural_size = max(1, int(len(structural_surface_vocab_codes(structural_codebook))))
    except Exception:
        structural_size = _size_from_gap(structural_offset, obs_code_offset, 100_000)

    families: Dict[str, Any] = {
        "special": _family_entry(
            offset=special_offset,
            source_size=special_size,
            family_type="special",
            runtime_head="logits_struct",
            preserve_full_default=True,
            meta={"reserved_max_id": int(special_reserved_max)},
        ),
        "diagnosis": _family_entry(
            offset=diag_offset,
            source_size=diag_size,
            family_type="semantic",
            runtime_head="logits_medtok",
        ),
        "diagnosis_residual": _family_entry(
            offset=diag_res_offset,
            source_size=_bucket("diagnosis_residual", 39_999),
            family_type="residual",
            runtime_head="logits_medtok",
        ),
        "procedure": _family_entry(
            offset=proc_offset,
            source_size=proc_size,
            family_type="semantic",
            runtime_head="logits_medtok",
        ),
        "procedure_residual": _family_entry(
            offset=proc_res_offset,
            source_size=_bucket("procedure_residual", 39_999),
            family_type="residual",
            runtime_head="logits_medtok",
        ),
        "medication": _family_entry(
            offset=med_offset,
            source_size=med_size,
            family_type="semantic",
            runtime_head="logits_medtok",
        ),
        "medication_residual": _family_entry(
            offset=med_res_offset,
            source_size=_bucket("medication_residual", 39_999),
            family_type="residual",
            runtime_head="logits_medtok",
        ),
        "med_route": _family_entry(
            offset=med_route_offset,
            source_size=med_route_size,
            family_type="med_attr",
        ),
        "med_form": _family_entry(
            offset=med_form_offset,
            source_size=med_form_size,
            family_type="med_attr",
        ),
        "med_freq": _family_entry(
            offset=med_freq_offset,
            source_size=med_freq_size,
            family_type="med_attr",
        ),
        "med_unit": _family_entry(
            offset=med_unit_offset,
            source_size=med_unit_size,
            family_type="med_attr",
        ),
        "med_dosage": _family_entry(
            offset=med_dosage_offset,
            source_size=8,
            family_type="numeric_attr_bins",
        ),
        "med_rate": _family_entry(
            offset=med_rate_offset,
            source_size=6,
            family_type="numeric_attr_bins",
        ),
        "med_duration": _family_entry(
            offset=med_duration_offset,
            source_size=6,
            family_type="numeric_attr_bins",
        ),
        "measurement_code": _family_entry(
            offset=meas_code_offset,
            source_size=meas_code_size_eff,
            family_type="measurement",
            runtime_head="logits_meas",
        ),
        "measurement_value": _family_entry(
            offset=meas_val_offset,
            source_size=rvq_size_eff,
            family_type="measurement_value",
            runtime_head="logits_rvq",
        ),
        "structural": _family_entry(
            offset=structural_offset,
            source_size=structural_size,
            family_type="structural",
            runtime_head="logits_struct",
            preserve_full_default=True,
        ),
        "observation_code": _family_entry(
            offset=obs_code_offset,
            source_size=obs_code_size,
            family_type="observation",
            runtime_head="logits_meas",
        ),
        "observation_value": _family_entry(
            offset=obs_val_offset,
            source_size=obs_val_size,
            family_type="observation",
            runtime_head="logits_meas",
        ),
    }

    residual_offsets = _safe_dict(residual_cfg.get("offsets", {}))
    residual_families = {
        "diagnosis": {
            "offset": int(residual_offsets.get("diagnosis", diag_res_offset)),
            "buckets": int(_bucket("diagnosis_residual", 39_999)),
        },
        "procedure": {
            "offset": int(residual_offsets.get("procedure", proc_res_offset)),
            "buckets": int(_bucket("procedure_residual", 39_999)),
        },
        "medication": {
            "offset": int(residual_offsets.get("medication", med_res_offset)),
            "buckets": int(_bucket("medication_residual", 39_999)),
        },
    }

    payload: Dict[str, Any] = {
        "version": "token_vocab_sparse_v1",
        "families": families,
        "window_markers": {
            "enabled": bool(markers_cfg.get("enabled", True)),
            "end_mode": str(markers_cfg.get("end_mode", "end_token")),
            "type_token_offset": int(marker_type_offset),
            "num_types": int(marker_num_types),
            "end_token_id": int(marker_end),
            "continue_token_id": int(marker_continue),
            "unk_type_id": int(markers_cfg.get("unk_type_id", 0)),
        },
        "window_segmentation": dict(segmentation_cfg),
        "residual_fallback": {
            "enabled": bool(residual_cfg.get("enabled", True)),
            "buckets": int(residual_cfg.get("buckets", 39_999)),
            "families": residual_families,
        },
        "runtime_lane_order": list(DEFAULT_RUNTIME_LANE_ORDER),
        "runtime_block_order": list(DEFAULT_RUNTIME_BLOCK_ORDER),
        "legacy_sources": {
            "tokenization_contract": str(tokenization_contract),
            "vocab_manifest": str(vocab_manifest),
            "structural_yaml": str(structural_yaml),
            "medtok_vocab_dir": str(medtok_vocab_dir) if medtok_vocab_dir is not None else None,
            "medtok_attr_dir": str(medtok_attr_dir) if medtok_attr_dir is not None else None,
            "code2id_pt": str(code2id_pt) if code2id_pt is not None else None,
            "tokenizer_ckpt": str(tokenizer_ckpt) if tokenizer_ckpt is not None else None,
        },
    }
    return payload


def build_legacy_manifest_from_sparse_contract(contract: Mapping[str, Any]) -> Dict[str, Any]:
    families = _safe_dict(contract.get("families", {}))
    manifest: Dict[str, Any] = {}
    for name, raw in families.items():
        fam = _safe_dict(raw)
        if "offset" not in fam:
            continue
        entry: Dict[str, Any] = {"offset": int(fam["offset"])}
        if "source_size" in fam:
            entry["size"] = int(fam["source_size"])
        if str(fam.get("family_type", "")) == "residual":
            entry["buckets"] = int(fam.get("source_size", 0))
        manifest[str(name)] = entry

    markers = _safe_dict(contract.get("window_markers", {}))
    if markers:
        manifest["window_type"] = {
            "offset": int(markers.get("type_token_offset", 10)),
            "size": int(markers.get("num_types", 0)),
        }
        if markers.get("end_token_id", None) is not None:
            manifest["window_end"] = {"id": int(markers["end_token_id"])}
        if markers.get("continue_token_id", None) is not None:
            manifest["window_continue"] = {"id": int(markers["continue_token_id"])}
    return manifest


def write_sparse_vocab_contract(contract: Mapping[str, Any], output_json: str | Path) -> Path:
    out_fp = Path(output_json)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(dict(contract), indent=2), encoding="utf-8")
    return out_fp

