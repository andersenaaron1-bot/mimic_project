from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch

from src.ehr_hier.tokenizers.vocab_contract import (
    _load_json,
    build_sparse_vocab_contract,
    load_sparse_vocab_contract,
    _safe_dict,
)


@dataclass(frozen=True)
class DenseIdBlock:
    name: str
    head: str
    global_offset: int
    source_size: int
    dense_offset: int
    dense_size: int
    mode: str = "identity"  # identity | modulo
    sparse_global_ids: tuple[int, ...] | None = None

    @property
    def global_max(self) -> int:
        if self.mode == "lookup" and self.sparse_global_ids:
            return int(max(self.sparse_global_ids))
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
            "sparse_global_ids": (
                [int(x) for x in self.sparse_global_ids]
                if self.sparse_global_ids is not None
                else None
            ),
        }


class DenseIdRemapper:
    def __init__(self, blocks: Sequence[DenseIdBlock], *, unk_dense_id: int = 0) -> None:
        self.blocks: List[DenseIdBlock] = list(blocks)
        self.unk_dense_id = int(unk_dense_id)
        self._lookup_ids_cpu: Dict[str, torch.Tensor] = {}
        self._validate()
        self._init_lookup_tables()

    def _init_lookup_tables(self) -> None:
        self._lookup_ids_cpu.clear()
        for b in self.blocks:
            if b.mode != "lookup":
                continue
            if not b.sparse_global_ids:
                raise ValueError(f"lookup block {b.name} has no sparse_global_ids")
            ids = sorted({int(x) for x in b.sparse_global_ids})
            self._lookup_ids_cpu[b.name] = torch.tensor(ids, dtype=torch.long)

    def _validate(self) -> None:
        by_range = sorted(self.blocks, key=lambda b: (int(b.global_offset), int(b.global_max)))
        prev_max = None
        for b in by_range:
            if int(b.source_size) <= 0:
                raise ValueError(f"Block {b.name} has non-positive source_size={b.source_size}")
            if int(b.dense_size) <= 0:
                raise ValueError(f"Block {b.name} has non-positive dense_size={b.dense_size}")
            if b.mode not in {"identity", "modulo", "lookup"}:
                raise ValueError(f"Block {b.name} has unsupported mode={b.mode!r}")
            if b.mode == "lookup":
                if b.sparse_global_ids is None or len(b.sparse_global_ids) == 0:
                    raise ValueError(f"lookup block {b.name} must provide non-empty sparse_global_ids")
                uniq = {int(x) for x in b.sparse_global_ids}
                if len(uniq) != len(b.sparse_global_ids):
                    raise ValueError(f"lookup block {b.name} has duplicate sparse_global_ids")
                if int(b.dense_size) != len(uniq):
                    raise ValueError(
                        f"lookup block {b.name} dense_size={b.dense_size} must equal number of sparse ids={len(uniq)}"
                    )
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
        if valid_mask is not None:
            valid = valid_mask.to(dtype=torch.bool)
        else:
            valid = torch.ones_like(matched, dtype=torch.bool)
        flat_ids = ids.reshape(-1).to(dtype=torch.long)
        flat_out = out.reshape(-1)
        flat_matched = matched.reshape(-1)
        flat_valid = valid.reshape(-1)

        for block in self.blocks:
            if block.mode == "lookup":
                ids_cpu = self._lookup_ids_cpu.get(block.name, None)
                if ids_cpu is None or ids_cpu.numel() == 0:
                    block_hits[block.name] = 0
                    continue
                ids_sorted = ids_cpu.to(device=flat_ids.device)
                lo = int(ids_sorted[0].item())
                hi = int(ids_sorted[-1].item())
                mask_range = (flat_ids >= lo) & (flat_ids <= hi)
                if not mask_range.any():
                    block_hits[block.name] = 0
                    continue
                idx_range = torch.nonzero(mask_range, as_tuple=False).squeeze(-1)
                cand = flat_ids[idx_range]
                pos = torch.searchsorted(ids_sorted, cand)
                in_bounds = pos < int(ids_sorted.numel())
                if not in_bounds.any():
                    block_hits[block.name] = 0
                    continue
                idx_in = idx_range[in_bounds]
                cand_in = cand[in_bounds]
                pos_in = pos[in_bounds]
                exact = ids_sorted[pos_in] == cand_in
                if not exact.any():
                    block_hits[block.name] = 0
                    continue
                hit_idx = idx_in[exact]
                hit_local = pos_in[exact].to(dtype=torch.long)
                flat_out[hit_idx] = int(block.dense_offset) + hit_local
                flat_matched[hit_idx] = True
                block_hits[block.name] = int(flat_valid[hit_idx].sum().item())
                continue

            go = int(block.global_offset)
            hi = int(block.global_max)
            mask = (flat_ids >= go) & (flat_ids <= hi)
            if not mask.any():
                block_hits[block.name] = 0
                continue
            local = (flat_ids[mask] - go).to(dtype=torch.long)
            if block.mode == "modulo" and int(block.dense_size) > 0:
                local = torch.remainder(local, int(block.dense_size))
            elif block.mode == "identity":
                if int(block.dense_size) < int(block.source_size):
                    local = torch.remainder(local, int(block.dense_size))
            mapped = int(block.dense_offset) + local
            flat_out[mask] = mapped
            flat_matched[mask] = True
            block_hits[block.name] = int((mask & flat_valid).sum().item())

        out = flat_out.view_as(out)
        matched = flat_matched.view_as(matched)
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
            sparse_ids_raw = item.get("sparse_global_ids", None)
            sparse_ids = None
            if sparse_ids_raw is not None:
                if not isinstance(sparse_ids_raw, list):
                    raise TypeError(
                        f"serialized block sparse_global_ids must be list or null, got {type(sparse_ids_raw)}"
                    )
                sparse_ids = tuple(int(x) for x in sparse_ids_raw)
            blocks.append(
                DenseIdBlock(
                    name=str(item["name"]),
                    head=str(item["head"]),
                    global_offset=int(item["global_offset"]),
                    source_size=int(item["source_size"]),
                    dense_offset=int(item["dense_offset"]),
                    dense_size=int(item["dense_size"]),
                    mode=str(item.get("mode", "identity")),
                    sparse_global_ids=sparse_ids,
                )
            )
        unk_dense_id = int(payload.get("unk_dense_id", 0))
        return cls(blocks, unk_dense_id=unk_dense_id)


def build_runtime_vocab_and_remapper(
    *,
    sparse_vocab_contract: str | Path | Mapping[str, Any] | None = None,
    tokenization_contract: str | Path = "configs/data/tokenization_v1.yaml",
    vocab_manifest: str | Path = "artifacts/vocab_manifest.json",
    structural_yaml: str | Path = "configs/data/structural_codes.yaml",
    medtok_vocab_dir: str | Path | None = None,
    medtok_attr_dir: str | Path | None = "artifacts/medtok_attrs",
    code2id_pt: str | Path | None = None,
    tokenizer_ckpt: str | Path | None = None,
    measurement_code_size: Optional[int] = None,
    rvq_size: Optional[int] = None,
) -> tuple[Dict[str, Any], DenseIdRemapper]:
    if sparse_vocab_contract is None:
        sparse_contract = build_sparse_vocab_contract(
            tokenization_contract=tokenization_contract,
            vocab_manifest=vocab_manifest,
            structural_yaml=structural_yaml,
            medtok_vocab_dir=medtok_vocab_dir,
            medtok_attr_dir=medtok_attr_dir,
            code2id_pt=code2id_pt,
            tokenizer_ckpt=tokenizer_ckpt,
            measurement_code_size=measurement_code_size,
            rvq_size=rvq_size,
        )
    elif isinstance(sparse_vocab_contract, Mapping):
        sparse_contract = dict(sparse_vocab_contract)
    else:
        sparse_contract = load_sparse_vocab_contract(sparse_vocab_contract)

    families = _safe_dict(sparse_contract.get("families", {}))
    if not families:
        raise ValueError("Sparse vocab contract must contain non-empty 'families'")
    markers_cfg = _safe_dict(sparse_contract.get("window_markers", {}))
    lane_order = tuple(
        str(x)
        for x in sparse_contract.get(
            "runtime_lane_order",
            ("logits_struct", "logits_rvq", "logits_meas", "logits_medtok"),
        )
    )
    runtime_block_order = [
        str(x)
        for x in sparse_contract.get(
            "runtime_block_order",
            (
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
            ),
        )
    ]
    blocks_spec: List[tuple[str, str, int, int, int, str]] = []
    for name in runtime_block_order:
        fam = _safe_dict(families.get(name, {}))
        head = fam.get("runtime_head", None)
        if head is None:
            continue
        source_size = int(fam.get("source_size", 0))
        if source_size <= 0:
            raise ValueError(f"Runtime family {name!r} has non-positive source_size={source_size}")
        blocks_spec.append(
            (
                str(name),
                str(head),
                int(fam["offset"]),
                int(source_size),
                int(source_size),
                str(fam.get("runtime_mode", "identity")),
            )
        )

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
            "type_token_offset": int(markers_cfg.get("type_token_offset", 10)),
            "num_types": int(markers_cfg.get("num_types", 0)),
            "end_token_id": int(markers_cfg.get("end_token_id", 0)),
            "continue_token_id": int(markers_cfg.get("continue_token_id", 0)),
            "unk_type_id": int(markers_cfg.get("unk_type_id", 0)),
        },
        "dense_blocks": [b.serialize_meta() for b in blocks],
        "sparse_vocab_contract": dict(sparse_contract),
    }
    return vocab_config, remapper


def build_compact_runtime_vocab_and_remapper(
    *,
    base_vocab_config: Mapping[str, Any],
    base_remapper: DenseIdRemapper,
    observed_ids_by_block: Mapping[str, Iterable[int]],
    preserve_full_blocks: Optional[Iterable[str]] = None,
) -> tuple[Dict[str, Any], DenseIdRemapper]:
    """
    Build a compact runtime vocab by retaining only observed global IDs per block.

    Any unobserved token for a compacted block maps to UNK at runtime.
    """
    preserve = {str(x) for x in (preserve_full_blocks or [])}
    routing_cfg = _safe_dict(base_vocab_config.get("routing", {}))
    lane_order = tuple(routing_cfg.keys()) or (
        "logits_struct",
        "logits_rvq",
        "logits_meas",
        "logits_medtok",
    )
    head_sizes: Dict[str, int] = {k: 0 for k in lane_order}

    compact_blocks: List[DenseIdBlock] = []
    dense_cursor = 0
    for b in base_remapper.blocks:
        obs_raw = observed_ids_by_block.get(str(b.name), [])
        obs = sorted({int(x) for x in obs_raw})

        if b.name in preserve:
            keep_block = DenseIdBlock(
                name=str(b.name),
                head=str(b.head),
                global_offset=int(b.global_offset),
                source_size=int(b.source_size),
                dense_offset=int(dense_cursor),
                dense_size=int(b.dense_size),
                mode=str(b.mode),
                sparse_global_ids=b.sparse_global_ids,
            )
        else:
            if not obs:
                # Keep one fallback slot for lane stability.
                obs = [int(b.global_offset)]
            min_id = int(min(obs))
            max_id = int(max(obs))
            keep_block = DenseIdBlock(
                name=str(b.name),
                head=str(b.head),
                global_offset=int(min_id),
                source_size=int(max(1, max_id - min_id + 1)),
                dense_offset=int(dense_cursor),
                dense_size=int(len(obs)),
                mode="lookup",
                sparse_global_ids=tuple(int(x) for x in obs),
            )

        compact_blocks.append(keep_block)
        dense_cursor += int(keep_block.dense_size)
        head_sizes[str(keep_block.head)] = int(head_sizes.get(str(keep_block.head), 0)) + int(
            keep_block.dense_size
        )

    remapper = DenseIdRemapper(compact_blocks, unk_dense_id=int(base_remapper.unk_dense_id))
    routing: Dict[str, List[Dict[str, Any]]] = {k: [] for k in lane_order}
    for block in compact_blocks:
        routing[str(block.head)].append(block.serialize_routing())

    struct_offset_dense = 0
    rvq_offset_dense = struct_offset_dense + head_sizes["logits_struct"]
    meas_offset_dense = rvq_offset_dense + head_sizes["logits_rvq"]
    med_offset_dense = meas_offset_dense + head_sizes["logits_meas"]

    vocab_out: Dict[str, Any] = {
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
        "window_markers": dict(base_vocab_config.get("window_markers", {}) or {}),
        "dense_blocks": [b.serialize_meta() for b in compact_blocks],
        "sparse_vocab_contract": dict(base_vocab_config.get("sparse_vocab_contract", {}) or {}),
    }
    return vocab_out, remapper


def load_runtime_vocab_bundle(bundle_json: str | Path) -> tuple[Dict[str, Any], DenseIdRemapper]:
    payload = _load_json(bundle_json)
    vocab_config = _safe_dict(payload.get("vocab_config", {}))
    if not vocab_config:
        raise ValueError(f"bundle missing non-empty 'vocab_config': {bundle_json}")
    remap_payload = _safe_dict(payload.get("id_remapper", {}))
    remapper = DenseIdRemapper.from_serialized(remap_payload)
    return vocab_config, remapper
