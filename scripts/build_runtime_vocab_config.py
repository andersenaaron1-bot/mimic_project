#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ehr_hier.tokenizers.vocab_contract import (
    build_sparse_vocab_contract,
    load_sparse_vocab_contract,
)
from src.ehr_hier.transformer.vocab_runtime import build_runtime_vocab_and_remapper


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build runtime transformer vocab_config + sparse->dense remap metadata "
            "from tokenization contract and manifest."
        )
    )
    ap.add_argument("--tokenization_yaml", default="configs/data/tokenization_v1.yaml")
    ap.add_argument("--vocab_manifest", default="artifacts/vocab_manifest.json")
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--sparse_vocab_json", default=None)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", default=None)
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
    ap.add_argument("--allow_smoke_medtok", action="store_true")
    ap.add_argument("--code2id_pt", default=None)
    ap.add_argument("--tokenizer_ckpt", default=None)
    ap.add_argument("--measurement_code_size", type=int, default=None)
    ap.add_argument("--rvq_size", type=int, default=None)
    ap.add_argument("--output_sparse_json", default=None)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    if args.sparse_vocab_json:
        sparse_contract = load_sparse_vocab_contract(args.sparse_vocab_json)
    else:
        sparse_contract = build_sparse_vocab_contract(
            tokenization_contract=args.tokenization_yaml,
            vocab_manifest=args.vocab_manifest,
            structural_yaml=args.structural_yaml,
            medtok_code2embeds=args.medtok_code2embeds,
            medtok_vocab_dir=args.medtok_vocab_dir,
            medtok_attr_dir=args.medtok_attr_dir,
            code2id_pt=args.code2id_pt,
            tokenizer_ckpt=args.tokenizer_ckpt,
            measurement_code_size=args.measurement_code_size,
            rvq_size=args.rvq_size,
            allow_smoke_medtok=bool(args.allow_smoke_medtok),
        )

    vocab_config, remapper = build_runtime_vocab_and_remapper(
        sparse_vocab_contract=sparse_contract,
    )

    payload = {
        "sparse_vocab_contract": sparse_contract,
        "vocab_config": vocab_config,
        "id_remapper": remapper.serialize(),
    }
    if args.output_sparse_json:
        sparse_fp = Path(args.output_sparse_json)
        sparse_fp.parent.mkdir(parents=True, exist_ok=True)
        sparse_fp.write_text(json.dumps(sparse_contract, indent=2), encoding="utf-8")
    out_fp = Path(args.output_json)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(
        {
            "output_json": str(out_fp),
            "total_size": int(vocab_config["total_size"]),
            "size_special": int(vocab_config["size_special"]),
            "size_rvq": int(vocab_config["size_rvq"]),
            "size_meas_labels": int(vocab_config["size_meas_labels"]),
            "size_meds": int(vocab_config["size_meds"]),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
