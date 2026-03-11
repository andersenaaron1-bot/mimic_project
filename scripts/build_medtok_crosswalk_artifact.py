#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.ehr_hier.tokenizers.medtok_crosswalk import build_medtok_crosswalk_artifact  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a MedTok crosswalk artifact for procedure and medication from "
            "stable MIMIC concept maps and optional MEDS metadata codes.parquet."
        )
    )
    ap.add_argument(
        "--concept_map_dir",
        default="_tmp_meds_etl/src/meds_etl/mimic/concept_map",
        help="Directory containing inputevents_to_rxnorm.csv / proc_itemid.csv / proc_datetimeevents.csv.",
    )
    ap.add_argument(
        "--codes_parquet",
        default=None,
        help="Optional MEDS metadata/codes.parquet to add code/description -> parent-based aliases.",
    )
    ap.add_argument("--output_json", required=True)
    args = ap.parse_args()

    payload = build_medtok_crosswalk_artifact(
        concept_map_dir=args.concept_map_dir,
        codes_parquet=args.codes_parquet,
    )
    out_fp = Path(args.output_json)
    out_fp.parent.mkdir(parents=True, exist_ok=True)
    out_fp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(
        json.dumps(
            {
                "output_json": str(out_fp),
                "sources": payload.get("sources", {}),
                "family_summaries": {
                    fam: info.get("summary", {})
                    for fam, info in payload.get("families", {}).items()
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
