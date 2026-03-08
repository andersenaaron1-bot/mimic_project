#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    ap.add_argument("--medtok_vocab_dir", default=None)
    ap.add_argument("--code2id_pt", default=None)
    ap.add_argument("--tokenizer_ckpt", default=None)
    ap.add_argument("--measurement_code_size", type=int, default=None)
    ap.add_argument("--rvq_size", type=int, default=None)
    ap.add_argument("--structural_entity_dense_size", type=int, default=65_536)
    ap.add_argument("--structural_entity_source_size", type=int, default=900_000)
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    vocab_config, remapper = build_runtime_vocab_and_remapper(
        tokenization_contract=args.tokenization_yaml,
        vocab_manifest=args.vocab_manifest,
        medtok_vocab_dir=args.medtok_vocab_dir,
        code2id_pt=args.code2id_pt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        measurement_code_size=args.measurement_code_size,
        rvq_size=args.rvq_size,
        structural_entity_dense_size=args.structural_entity_dense_size,
        structural_entity_source_size=args.structural_entity_source_size,
    )

    payload = {
        "vocab_config": vocab_config,
        "id_remapper": remapper.serialize(),
    }
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
            "structural_entity_dense_size": int(args.structural_entity_dense_size),
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
