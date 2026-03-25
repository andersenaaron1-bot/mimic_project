#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.audit_tokenization_flow import (  # noqa: E402
    _build_measurement_config,
    _build_static_artifacts,
    _build_struct_vocab,
    _load_subject_ids,
    _load_tokenization_contract,
    _resolve_residual_policy,
)
from src.ehr_hier.data.compile_dataset import compile_dataset  # noqa: E402
from src.ehr_hier.data.structural_codes import structural_surface_vocab_codes  # noqa: E402
from src.ehr_hier.tokenizers.base_encoder import build_base_encoders  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Compile subject timelines under the frozen tokenization v1 contract and "
            "write a precompiled index/manifest for long-run transformer training."
        )
    )
    ap.add_argument("--meds_reader_db", required=True)
    ap.add_argument("--splits_parquet", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--max_subjects", type=int, default=None)
    ap.add_argument("--sample_seed", type=int, default=1337)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_workers", type=int, default=None)
    ap.add_argument("--num_output_shards", type=int, default=100)
    ap.add_argument("--skip_existing", action="store_true")
    ap.add_argument("--progress_every", type=int, default=100)

    ap.add_argument("--tokenization_yaml", default="configs/data/tokenization_v1.yaml")
    ap.add_argument("--structural_yaml", default="configs/data/structural_codes.yaml")
    ap.add_argument("--sparse_vocab_json", default=None)
    ap.add_argument("--medtok_code2embeds", default=None)
    ap.add_argument("--medtok_vocab_dir", required=True)
    ap.add_argument("--medtok_attr_dir", default="artifacts/medtok_attrs")
    ap.add_argument("--medtok_crosswalk_json", default=None)
    ap.add_argument("--codes_parquet_parent_lookup", default=None)

    ap.add_argument("--code2id_pt", required=True)
    ap.add_argument("--stats_pt", required=True)
    ap.add_argument("--cvae_ckpt", required=True)
    ap.add_argument("--tokenizer_ckpt", required=True)

    ap.add_argument("--disable_residual_fallback", action="store_true")
    ap.add_argument("--residual_fallback_buckets", type=int, default=39999)
    ap.add_argument("--diag_residual_offset", type=int, default=None)
    ap.add_argument("--proc_residual_offset", type=int, default=None)
    ap.add_argument("--med_residual_offset", type=int, default=None)
    args = ap.parse_args()

    tokenization_contract = _load_tokenization_contract(args.tokenization_yaml)
    artifact_args = argparse.Namespace(
        structural_yaml=args.structural_yaml,
        sparse_vocab_json=args.sparse_vocab_json,
        medtok_code2embeds=args.medtok_code2embeds,
        medtok_vocab_dir=args.medtok_vocab_dir,
        medtok_attr_dir=args.medtok_attr_dir,
        medtok_crosswalk_json=args.medtok_crosswalk_json,
        code2id_pt=args.code2id_pt,
        tokenizer_ckpt=args.tokenizer_ckpt,
        codes_parquet_parent_lookup=args.codes_parquet_parent_lookup,
    )
    artifacts = _build_static_artifacts(artifact_args)

    residual_enabled, residual_buckets, residual_offsets = _resolve_residual_policy(
        args,
        tokenization_contract=tokenization_contract,
    )
    struct_codes_union = set(structural_surface_vocab_codes(artifacts.structural_codebook))
    struct_vocab = _build_struct_vocab(struct_codes_union, manifest=artifacts.manifest)
    meas_cfg = _build_measurement_config(args, artifacts=artifacts)
    if meas_cfg is None:
        raise ValueError("Missing measurement artifacts; need code2id/stats/cvae/tokenizer checkpoints.")

    encoders = build_base_encoders(
        meas_cfg,
        diag_vocab=artifacts.diag_vocab,
        proc_vocab=artifacts.proc_vocab,
        med_vocab=artifacts.med_vocab,
        struct_vocab=struct_vocab,
        med_attr_vocabs=artifacts.med_attr_vocabs,
        med_numeric_attrs=artifacts.med_numeric_attrs,
        medtok_parent_lookup=artifacts.medtok_parent_lookup,
        medtok_crosswalks=artifacts.medtok_crosswalks,
        residual_fallback_vocabs=artifacts.residual_fallback_vocabs,
        enable_residual_fallback=bool(residual_enabled),
        residual_fallback_buckets=int(residual_buckets),
        residual_fallback_offsets=dict(residual_offsets),
    )

    subject_ids = _load_subject_ids(
        str(args.splits_parquet),
        str(args.split),
        int(args.max_subjects) if args.max_subjects is not None else None,
        sample_seed=int(args.sample_seed),
    )
    print(
        json.dumps(
            {
                "event": "compile_timelines_config",
                "split": str(args.split),
                "subject_count": int(len(subject_ids)),
                "output_dir": str(args.output_dir),
                "num_workers": None if args.num_workers is None else int(args.num_workers),
                "num_output_shards": int(args.num_output_shards),
                "skip_existing": bool(args.skip_existing),
                "progress_every": int(args.progress_every),
            },
            indent=2,
        ),
        flush=True,
    )
    manifest = compile_dataset(
        db_path=str(args.meds_reader_db),
        encoders=encoders,
        output_dir=str(args.output_dir),
        structural_codebook=artifacts.structural_codebook,
        qual_obs_code_vocab=artifacts.obs_code_vocab,
        qual_obs_value_vocab=artifacts.obs_value_vocab,
        qual_obs_tail_policy=artifacts.obs_tail_policy,
        num_workers=args.num_workers,
        subject_ids=subject_ids,
        num_output_shards=int(args.num_output_shards),
        splits_parquet=str(args.splits_parquet),
        write_index=True,
        skip_existing=bool(args.skip_existing),
        progress_every=int(args.progress_every),
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
