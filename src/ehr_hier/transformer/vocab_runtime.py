from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
import yaml


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
    vals: List[int] = []
    for v in payload.values():
        try:
            vals.append(int(v))
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
    vals: List[int] = []
    for v in payload.values():
        try:
            vals.append(int(v))
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


@dataclass(frozen=True)
class DenseIdBlock:
    name: str
    head: str
    global_offset: int
    source_size: int
    dense_offset: int
    dense_size: int
    mode: str = "identity"  # identity | modulo

    @property
    def global_max(self) -> int:
        return int(self.global_offset) + int(self.source_size) - 1

    def serialize_routing(self) -> Dict[str, Any]:
        return {
            "offset": int(self.dense_offset),
            "size": int(self.dense_size),
            "name": str(self.name),
        }

    def serialize_meta(self) -> Dict[str, Any]:
        return {
            "name": str(self.name),
            "head": str(self.head),
            "global_offset": int(self.global_offset),
            "source_size": int(self.source_size),
            "global_max": int(self.global_max),
            "dense_offset": int(self.dense_offset),
            "dense_size": int(self.dense_size),
            "mode": str(self.mode),
        }


class DenseIdRemapper:
    def __init__(self, blocks: Sequence[DenseIdBlock], *, unk_dense_id: int = 0) -> None:
        self.blocks: List[DenseIdBlock] = list(blocks)
        self.unk_dense_id = int(unk_dense_id)
        self._validate()

    def _validate(self) -> None:
        by_range = sorted(self.blocks, key=lambda b: (int(b.global_offset), int(b.global_max)))
        prev_max = None
        for b in by_range:
            if int(b.source_size) <= 0:
                raise ValueError(f"Block {b.name} has non-positive source_size={b.source_size}")
            if int(b.dense_size) <= 0:
                raise ValueError(f"Block {b.name} has non-positive dense_size={b.dense_size}")
            if b.mode not in {"identity", "modulo"}:
                raise ValueError(f"Block {b.name} has unsupported mode={b.mode!r}")
            if prev_max is not None and int(b.global_offset) <= int(prev_max):
                raise ValueError(
                    f"Global range overlap: block={b.name} starts at {b.global_offset} after prev_max={prev_max}"
                )
            prev_max = int(b.global_max)

    def map_tensor(
        self,
        ids: torch.Tensor,
        *,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Dict[str, Any]]:
        out = torch.full_like(ids, fill_value=int(self.unk_dense_id))
        matched = torch.zeros_like(ids, dtype=torch.bool)
        block_hits: Dict[str, int] = {}

        for block in self.blocks:
            go = int(block.global_offset)
            hi = int(block.global_max)
            mask = (ids >= go) & (ids <= hi)
            if not mask.any():
                block_hits[block.name] = 0
                continue
            local = (ids[mask] - go).to(dtype=torch.long)
            if block.mode == "modulo" and int(block.dense_size) > 0:
                local = torch.remainder(local, int(block.dense_size))
            elif block.mode == "identity":
                if int(block.dense_size) < int(block.source_size):
                    local = torch.remainder(local, int(block.dense_size))
            mapped = int(block.dense_offset) + local
            out[mask] = mapped
            matched = matched | mask
            block_hits[block.name] = int(mask.sum().item())

        if valid_mask is not None:
            valid = valid_mask.to(dtype=torch.bool)
        else:
            valid = torch.ones_like(matched, dtype=torch.bool)
        unmapped = valid & ~matched
        stats = {
            "total_tokens": int(valid.sum().item()),
            "mapped_tokens": int((valid & matched).sum().item()),
            "unmapped_tokens": int(unmapped.sum().item()),
            "unmapped_frac": (
                float(unmapped.sum().item()) / float(valid.sum().item())
                if int(valid.sum().item()) > 0
                else 0.0
            ),
            "block_hits": dict(block_hits),
        }
        return out, stats

    def serialize(self) -> Dict[str, Any]:
        return {
            "unk_dense_id": int(self.unk_dense_id),
            "blocks": [b.serialize_meta() for b in self.blocks],
        }

    @classmethod
    def from_serialized(cls, payload: Mapping[str, Any]) -> "DenseIdRemapper":
        blocks_raw = payload.get("blocks", [])
        if not isinstance(blocks_raw, list):
            raise TypeError("serialized remapper payload must contain list field 'blocks'")
        blocks: List[DenseIdBlock] = []
        for item in blocks_raw:
            if not isinstance(item, dict):
                raise TypeError(f"serialized block must be dict, got {type(item)}")
            blocks.append(
                DenseIdBlock(
                    name=str(item["name"]),
                    head=str(item["head"]),
                    global_offset=int(item["global_offset"]),
                    source_size=int(item["source_size"]),
                    dense_offset=int(item["dense_offset"]),
                    dense_size=int(item["dense_size"]),
                    mode=str(item.get("mode", "identity")),
                )
            )
        unk_dense_id = int(payload.get("unk_dense_id", 0))
        return cls(blocks, unk_dense_id=unk_dense_id)


def build_runtime_vocab_and_remapper(
    *,
    tokenization_contract: str | Path = "configs/data/tokenization_v1.yaml",
    vocab_manifest: str | Path = "artifacts/vocab_manifest.json",
    medtok_vocab_dir: str | Path | None = None,
    code2id_pt: str | Path | None = None,
    tokenizer_ckpt: str | Path | None = None,
    measurement_code_size: Optional[int] = None,
    rvq_size: Optional[int] = None,
    structural_entity_dense_size: int = 65_536,
    structural_entity_source_size: int = 900_000,
) -> tuple[Dict[str, Any], DenseIdRemapper]:
    contract = _load_yaml(tokenization_contract)
    manifest = _load_json(vocab_manifest)
    frozen = _safe_dict(contract.get("frozen_ranges", {}))
    residual_cfg = _safe_dict(contract.get("residual_fallback", {}))
    markers_cfg = _safe_dict(contract.get("window_markers", {}))

    def _offset(name: str, default: int) -> int:
        m = _safe_dict(manifest.get(name, {}))
        if "offset" in m:
            return int(m["offset"])
        fr = _safe_dict(frozen.get(name, {}))
        if "offset" in fr:
            return int(fr["offset"])
        return int(default)

    def _bucket(name: str, default: int) -> int:
        fr = _safe_dict(frozen.get(name, {}))
        if "buckets" in fr:
            return int(fr["buckets"])
        if "buckets" in residual_cfg:
            return int(residual_cfg["buckets"])
        return int(default)

    special_offset = _offset("special", 0)
    special_reserved_max = int(_safe_dict(frozen.get("special", {})).get("reserved_max_id", 99))
    marker_type_offset = int(markers_cfg.get("type_token_offset", 10))
    marker_num_types = int(markers_cfg.get("num_types", 16))
    marker_end = int(markers_cfg.get("end_token_id", marker_type_offset + marker_num_types))
    marker_continue = int(markers_cfg.get("continue_token_id", marker_end + 1))
    special_max_needed = max(special_reserved_max, marker_type_offset + marker_num_types - 1, marker_end, marker_continue)
    special_size = int(special_max_needed + 1)

    diag_offset = _offset("diagnosis", 1_000_000)
    diag_res_offset = _offset("diagnosis_residual", 1_160_000)
    proc_offset = _offset("procedure", 1_200_000)
    proc_res_offset = _offset("procedure_residual", 1_360_000)
    med_offset = _offset("medication", 1_400_000)
    med_res_offset = _offset("medication_residual", 1_800_000)
    meas_code_offset = _offset("measurement_code", 2_000_000)
    meas_val_offset = _offset("measurement_value", 2_100_000)
    structural_offset = _offset("structural", 2_200_000)
    obs_code_offset = _offset("observation_code", 2_300_000)
    obs_val_offset = _offset("observation_value", 2_320_000)
    struct_action_offset = _offset("structural_action", 2_400_000)
    struct_entity_offset = _offset("structural_entity", 2_420_000)

    def _size_from_gap(lo: int, hi: int, default: int) -> int:
        gap = int(hi) - int(lo)
        if gap > 0:
            return int(gap)
        return int(default)

    medtok_dir = Path(medtok_vocab_dir) if medtok_vocab_dir is not None else None
    diag_size = _vocab_size_from_json((medtok_dir / "diag_vocab.json") if medtok_dir is not None else None) or 20_000
    proc_size = _vocab_size_from_json((medtok_dir / "proc_vocab.json") if medtok_dir is not None else None) or 10_000
    med_size = _vocab_size_from_json((medtok_dir / "med_vocab.json") if medtok_dir is not None else None) or 25_000

    meas_code_size_eff = (
        int(measurement_code_size)
        if measurement_code_size is not None
        else (_code2id_size(code2id_pt) or 10_000)
    )
    rvq_size_eff = int(rvq_size) if rvq_size is not None else (_rvq_size_from_ckpt(tokenizer_ckpt) or 1_024)
    obs_code_size = _size_from_gap(obs_code_offset, obs_val_offset, 20_000)
    obs_val_source_size = _size_from_gap(obs_val_offset, struct_action_offset, 80_000)
    structural_size = _size_from_gap(structural_offset, obs_code_offset, 100_000)
    struct_action_size = _size_from_gap(struct_action_offset, struct_entity_offset, 20_000)
    struct_entity_source_size_eff = max(1, int(structural_entity_source_size))
    struct_entity_dense_size_eff = max(1, min(int(structural_entity_dense_size), int(struct_entity_source_size_eff)))

    blocks_spec: List[tuple[str, str, int, int, int, str]] = [
        # (name, head, global_offset, source_size, dense_size, mode)
        ("special", "logits_struct", special_offset, special_size, special_size, "identity"),
        ("structural", "logits_struct", structural_offset, structural_size, structural_size, "identity"),
        ("structural_action", "logits_struct", struct_action_offset, struct_action_size, struct_action_size, "identity"),
        (
            "structural_entity",
            "logits_struct",
            struct_entity_offset,
            struct_entity_source_size_eff,
            struct_entity_dense_size_eff,
            "modulo" if struct_entity_dense_size_eff < struct_entity_source_size_eff else "identity",
        ),
        ("measurement_value", "logits_rvq", meas_val_offset, rvq_size_eff, rvq_size_eff, "identity"),
        ("measurement_code", "logits_meas", meas_code_offset, meas_code_size_eff, meas_code_size_eff, "identity"),
        ("observation_code", "logits_meas", obs_code_offset, obs_code_size, obs_code_size, "identity"),
        ("observation_value", "logits_meas", obs_val_offset, obs_val_source_size, obs_val_source_size, "identity"),
        ("diagnosis", "logits_medtok", diag_offset, diag_size, diag_size, "identity"),
        ("diagnosis_residual", "logits_medtok", diag_res_offset, _bucket("diagnosis_residual", 39_999), _bucket("diagnosis_residual", 39_999), "identity"),
        ("procedure", "logits_medtok", proc_offset, proc_size, proc_size, "identity"),
        ("procedure_residual", "logits_medtok", proc_res_offset, _bucket("procedure_residual", 39_999), _bucket("procedure_residual", 39_999), "identity"),
        ("medication", "logits_medtok", med_offset, med_size, med_size, "identity"),
        ("medication_residual", "logits_medtok", med_res_offset, _bucket("medication_residual", 39_999), _bucket("medication_residual", 39_999), "identity"),
    ]

    lane_order = ("logits_struct", "logits_rvq", "logits_meas", "logits_medtok")
    head_sizes: Dict[str, int] = {k: 0 for k in lane_order}
    blocks: List[DenseIdBlock] = []
    dense_cursor = 0
    for lane in lane_order:
        for name, head, go, src_size, dense_size, mode in blocks_spec:
            if head != lane:
                continue
            b = DenseIdBlock(
                name=str(name),
                head=str(head),
                global_offset=int(go),
                source_size=int(src_size),
                dense_offset=int(dense_cursor),
                dense_size=int(dense_size),
                mode=str(mode),
            )
            blocks.append(b)
            dense_cursor += int(dense_size)
            head_sizes[lane] += int(dense_size)

    remapper = DenseIdRemapper(blocks, unk_dense_id=0)

    routing: Dict[str, List[Dict[str, Any]]] = {k: [] for k in lane_order}
    for block in blocks:
        routing[block.head].append(block.serialize_routing())

    struct_offset_dense = 0
    rvq_offset_dense = struct_offset_dense + head_sizes["logits_struct"]
    meas_offset_dense = rvq_offset_dense + head_sizes["logits_rvq"]
    med_offset_dense = meas_offset_dense + head_sizes["logits_meas"]

    vocab_config: Dict[str, Any] = {
        "total_size": int(dense_cursor),
        "size_special": int(head_sizes["logits_struct"]),
        "size_rvq": int(head_sizes["logits_rvq"]),
        "size_meas_labels": int(head_sizes["logits_meas"]),
        "size_meds": int(head_sizes["logits_medtok"]),
        "offsets": {
            "SPECIAL": int(struct_offset_dense),
            "RVQ": int(rvq_offset_dense),
            "MEAS": int(meas_offset_dense),
            "MED": int(med_offset_dense),
        },
        "routing": routing,
        "window_markers": {
            "enabled": bool(markers_cfg.get("enabled", True)),
            "end_mode": str(markers_cfg.get("end_mode", "end_token")),
            "type_token_offset": int(marker_type_offset),
            "num_types": int(marker_num_types),
            "end_token_id": int(marker_end),
            "continue_token_id": int(marker_continue),
            "unk_type_id": int(markers_cfg.get("unk_type_id", 0)),
        },
        "dense_blocks": [b.serialize_meta() for b in blocks],
    }
    return vocab_config, remapper


def load_runtime_vocab_bundle(bundle_json: str | Path) -> tuple[Dict[str, Any], DenseIdRemapper]:
    payload = _load_json(bundle_json)
    vocab_config = _safe_dict(payload.get("vocab_config", {}))
    if not vocab_config:
        raise ValueError(f"bundle missing non-empty 'vocab_config': {bundle_json}")
    remap_payload = _safe_dict(payload.get("id_remapper", {}))
    remapper = DenseIdRemapper.from_serialized(remap_payload)
    return vocab_config, remapper
