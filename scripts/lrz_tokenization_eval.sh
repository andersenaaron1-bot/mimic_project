#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lrz_tokenization_env.sh
source "${SCRIPT_DIR%/}/lrz_tokenization_env.sh"

job="${1:-help}"

case "$job" in
  help|-h|--help)
    cat <<'EOF'
Usage:
  source scripts/lrz_tokenization_env.sh
  scripts/lrz_tokenization_eval.sh <subcommand>

Subcommands:
  paths            Print the resolved LRZ tokenization paths.
  sanity           Check that repo/data/artifacts are reachable in the CPU container.
  crosswalk        Build the MedTok crosswalk artifact.
  decision-table   Build the 20k train decision table.
  compressed-vocabs
                   Build compressed MedTok vocabs plus exact fallback vocabs.
  sparse-vocab     Build the sparse/base vocab contract.
  audit-1000       Run the 1000-subject tokenization audit.
  compact-vocab    Build the compact runtime vocab from 20k subjects.
  freeze-check     Run the v1 freeze gate.
  debug-medtok-full
                   Audit recoverability against the full MedTok embedding universe.
EOF
    ;;

  paths)
    lrz_tok_show_paths
    ;;

  sanity)
    lrz_tok_cpu 2 4G 00:05:00 \
      "[ -d /workspace/ehr-hier ] && [ -e \"$DB\" ] && [ -f \"$SPLITS\" ] && [ -f \"$CODES_PARQUET\" ] && [ -d \"$MEDTOK_VOC_BASE\" ] && echo paths_ok"
    ;;

  crosswalk)
    lrz_tok_cpu 4 8G 00:20:00 \
      "python scripts/build_medtok_crosswalk_artifact.py --concept_map_dir _tmp_meds_etl/src/meds_etl/mimic/concept_map --codes_parquet \"$CODES_PARQUET\" --output_json \"$CROSSWALK_JSON_CT\" 2>&1 | tee \"$EVAL_DIR_CT/build_medtok_crosswalk_v1.log\""
    ;;

  decision-table)
    lrz_tok_cpu 8 48G 04:00:00 \
      "python scripts/build_uncaptured_decision_table.py --meds_reader_db \"$DB\" --splits_parquet \"$SPLITS\" --split train --sample_subjects 20000 --sample_strategy random --sample_seed 13 --top_k 0 --medtok_vocab_dir \"$MEDTOK_VOC_BASE\" --code2id_pt \"$ART/code2id.pt\" --structural_yaml configs/data/structural_codes.yaml --output_csv \"$DECISION_CSV_CT\" --output_json \"$EVAL_DIR_CT/decision_table_train_20k.json\" --progress_every 250 2>&1 | tee \"$EVAL_DIR_CT/decision_table_train_20k.log\""
    ;;

  compressed-vocabs)
    lrz_tok_cpu 4 16G 00:45:00 \
      "python scripts/build_compressed_medtok_vocabs.py --decision_csv \"$DECISION_CSV_CT\" --out_dir \"$MEDTOK_VOC_FINAL_CT\" --medtok_vocab_dir \"$MEDTOK_VOC_BASE\" --medtok_crosswalk_json \"$CROSSWALK_JSON_CT\" --diag_fallback_target_coverage 0.98 --proc_fallback_target_coverage 0.98 --med_fallback_target_coverage 0.98 --output_report_json \"$EVAL_DIR_CT/medtok_vocab_report_v1.json\" 2>&1 | tee \"$EVAL_DIR_CT/build_compressed_medtok_vocabs.log\""
    ;;

  sparse-vocab)
    lrz_tok_cpu 4 8G 00:20:00 \
      "python scripts/build_sparse_vocab_contract.py --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --medtok_vocab_dir \"$MEDTOK_VOC_FINAL_CT\" --code2id_pt \"$ART/code2id.pt\" --tokenizer_ckpt \"$ART/value_tokenizer.pt\" --output_json \"$SPARSE_VOC_CT\" 2>&1 | tee \"$EVAL_DIR_CT/build_sparse_vocab_contract.log\""
    ;;

  audit-1000)
    lrz_tok_cpu 8 64G 02:00:00 \
      "python scripts/audit_tokenization_flow.py --meds_reader_db \"$DB\" --splits_parquet \"$SPLITS\" --split train --max_subjects 1000 --sample_seed 13 --sparse_vocab_json \"$SPARSE_VOC_CT\" --medtok_vocab_dir \"$MEDTOK_VOC_FINAL_CT\" --medtok_crosswalk_json \"$CROSSWALK_JSON_CT\" --codes_parquet_parent_lookup \"$CODES_PARQUET\" --code2id_pt \"$ART/code2id.pt\" --stats_pt \"$ART/stats.pt\" --cvae_ckpt \"$ART/cvae_ckpt.pt\" --tokenizer_ckpt \"$ART/value_tokenizer.pt\" --progress_every 50 --output_json \"$AUDIT_JSON_CT\" 2>&1 | tee \"$EVAL_DIR_CT/tokenization_audit_train_1000.log\""
    ;;

  compact-vocab)
    lrz_tok_cpu 8 32G 06:00:00 \
      "python scripts/build_compact_runtime_vocab.py --meds_reader_db \"$DB\" --splits_parquet \"$SPLITS\" --split train --max_subjects 20000 --sample_seed 13 --workers 0 --subject_chunk_size 128 --progress_every 500 --sparse_vocab_json \"$SPARSE_VOC_CT\" --medtok_vocab_dir \"$MEDTOK_VOC_FINAL_CT\" --medtok_crosswalk_json \"$CROSSWALK_JSON_CT\" --codes_parquet_parent_lookup \"$CODES_PARQUET\" --code2id_pt \"$ART/code2id.pt\" --stats_pt \"$ART/stats.pt\" --cvae_ckpt \"$ART/cvae_ckpt.pt\" --tokenizer_ckpt \"$ART/value_tokenizer.pt\" --out_json \"$RUNTIME_VOC_CT\" 2>&1 | tee \"$EVAL_DIR_CT/runtime_vocab_compact_v1_20k.log\""
    ;;

  freeze-check)
    lrz_tok_cpu 2 4G 00:10:00 \
      "python scripts/check_tokenization_freeze_v1.py --audit_json \"$AUDIT_JSON_CT\" --runtime_vocab_json \"$RUNTIME_VOC_CT\" --require_no_residual_hash --min_mapped_rate_medication 0.98 2>&1 | tee \"$EVAL_DIR_CT/check_tokenization_freeze_v1.log\""
    ;;

  debug-medtok-full)
    lrz_tok_cpu 8 32G 02:00:00 \
      "python scripts/debug_medtok_matching.py --db \"$DB\" --code2embeds \"$MEDTOK_CODE2EMBEDS\" --codes-parquet \"$CODES_PARQUET\" --crosswalk-json \"$CROSSWALK_JSON_CT\" --max-subjects 1000 --top-k 20 --output-json \"$EVAL_DIR_CT/debug_medtok_matching_1000.json\" 2>&1 | tee \"$EVAL_DIR_CT/debug_medtok_matching_1000.log\""
    ;;

  *)
    lrz_die "Unknown subcommand: $job"
    ;;
esac
