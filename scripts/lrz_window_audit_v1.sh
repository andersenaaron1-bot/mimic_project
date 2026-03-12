#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "${SCRIPT_DIR%/}/lib/lrz_secure_runtime.sh"

CRITICAL_PATHS_SH="${CRITICAL_PATHS_SH:-$HOME/mimic_project/analysis/lrz_fs_inventory/latest/critical_paths.sh}"
[[ -f "$CRITICAL_PATHS_SH" ]] || lrz_die "critical_paths.sh not found: $CRITICAL_PATHS_SH"
# shellcheck source=/dev/null
source "$CRITICAL_PATHS_SH"

MAX_SUBJECTS="${MAX_SUBJECTS:-1000}"
SAMPLE_SEED="${SAMPLE_SEED:-13}"
MICRO_WINDOW_MAX_TOKENS="${MICRO_WINDOW_MAX_TOKENS:-4}"
MICRO_WINDOW_MAX_DURATION_HOURS="${MICRO_WINDOW_MAX_DURATION_HOURS:-1.0}"
PROGRESS_EVERY="${PROGRESS_EVERY:-50}"
WINDOW_AUDIT_JSON="${WINDOW_AUDIT_JSON:-$EVAL_DIR/window_contract_train_${MAX_SUBJECTS}.json}"
WINDOW_AUDIT_LOG="${WINDOW_AUDIT_LOG:-$EVAL_DIR/window_contract_train_${MAX_SUBJECTS}.log}"

srun --immediate=180 \
  -p lrz-cpu \
  --qos=cpu \
  --cpus-per-task=8 \
  --mem=64G \
  --time=02:00:00 \
  --container-image="$IMAGE_CPU" \
  --container-mounts="$CPU_MOUNTS" \
  bash -lc '
    set -euo pipefail
    mkdir -p "$EVAL_DIR"
    export PYTHONPATH="$REPO_CT:/deps${PYTHONPATH:+:$PYTHONPATH}"
    cd "$REPO_CT"
    if [ -d "$MEDTOK_VOC_FINAL" ]; then
      MEDTOK_VOC_FOR_AUDIT="$MEDTOK_VOC_FINAL"
    else
      MEDTOK_VOC_FOR_AUDIT="$MEDTOK_VOC_BASE"
    fi
    if [ -f "$CROSSWALK_JSON" ]; then
      CROSSWALK_ARGS=(--medtok_crosswalk_json "$CROSSWALK_JSON")
    else
      CROSSWALK_ARGS=()
    fi
    python scripts/audit_window_contract_v1.py \
      --meds_reader_db "$DB" \
      --splits_parquet "$SPLITS" \
      --split train \
      --max_subjects "'"$MAX_SUBJECTS"'" \
      --sample_seed "'"$SAMPLE_SEED"'" \
      --medtok_vocab_dir "$MEDTOK_VOC_FOR_AUDIT" \
      "${CROSSWALK_ARGS[@]}" \
      --codes_parquet_parent_lookup "$CODES_PARQUET" \
      --code2id_pt "$ART/code2id.pt" \
      --stats_pt "$ART/stats.pt" \
      --cvae_ckpt "$ART/cvae_ckpt.pt" \
      --tokenizer_ckpt "$ART/value_tokenizer.pt" \
      --tokenization_yaml configs/data/tokenization_v1.yaml \
      --micro_window_max_tokens "'"$MICRO_WINDOW_MAX_TOKENS"'" \
      --micro_window_max_duration_hours "'"$MICRO_WINDOW_MAX_DURATION_HOURS"'" \
      --progress_every "'"$PROGRESS_EVERY"'" \
      --output_json "'"$WINDOW_AUDIT_JSON"'" \
      2>&1 | tee "'"$WINDOW_AUDIT_LOG"'"
  '
