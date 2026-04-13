#!/usr/bin/env bash
#SBATCH --job-name=fmv2_phase3_eval
#SBATCH --partition=lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8,lrz-dgx-1-v100x8,lrz-v100x2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=log_%j.out
#SBATCH --error=log_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "${SCRIPT_DIR%/}/scripts/lib/lrz_secure_runtime.sh"
lrz_setup_job_hardening

lrz_require_cmd srun

REPO_DIR="${REPO_DIR:-$HOME/mimic_project}"
[[ -d "$REPO_DIR" ]] || lrz_die "Repository path not found: $REPO_DIR"

CRITICAL_PATHS_SH="${CRITICAL_PATHS_SH:-$REPO_DIR/analysis/lrz_fs_inventory/latest/critical_paths.sh}"
if [[ -f "$CRITICAL_PATHS_SH" ]]; then
  # shellcheck source=/dev/null
  source "$CRITICAL_PATHS_SH"
fi

DSS_HOST="${DSS_HOST:-/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2}"
DEPS_HOST="${DEPS_HOST:-$DSS_HOST/containers/runtime_pydeps}"
IMAGE_LOCAL="${IMAGE_LOCAL:-$DSS_HOST/containers/pytorch-2.5.1-cuda12.4.sqsh}"
IMAGE_REMOTE="${IMAGE_REMOTE:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
IMAGE="${IMAGE:-$(lrz_pick_container_image "$IMAGE_LOCAL" "$IMAGE_REMOTE")}"

RUN_TAG="${RUN_TAG:-phase3_eval_$(date +%Y%m%d_%H%M%S)}"
OUT_DIR_HOST="${OUT_DIR_HOST:-${OUT_DIR:-$DSS_HOST/etl/$RUN_TAG}}"
mkdir -p "$OUT_DIR_HOST"

RUN_TEST_GATE="${RUN_TEST_GATE:-1}"
RUN_EVAL_ONLY="${RUN_EVAL_ONLY:-1}"
RUN_PRECEDENT_BUILD="${RUN_PRECEDENT_BUILD:-1}"

TRAIN_SPLIT="${TRAIN_SPLIT:-train}"
EVAL_SPLIT="${EVAL_SPLIT:-tuning}"
PRECEDENT_SPLIT="${PRECEDENT_SPLIT:-train}"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-admission_chain}"
DEVICE="${DEVICE:-cuda}"
MAX_EVAL_SUBJECTS="${MAX_EVAL_SUBJECTS:-512}"
PRECEDENT_MAX_SUBJECTS="${PRECEDENT_MAX_SUBJECTS:-32}"
TOKENIZATION_YAML="${TOKENIZATION_YAML:-configs/data/tokenization_v1.yaml}"
STRUCTURAL_YAML="${STRUCTURAL_YAML:-configs/data/structural_codes.yaml}"
TRAIN_EXTRA_ARGS="${TRAIN_EXTRA_ARGS:-}"
INDEX_EXTRA_ARGS="${INDEX_EXTRA_ARGS:-}"
ALLOW_SMOKE_MEDTOK="${ALLOW_SMOKE_MEDTOK:-0}"

PRECOMPILED_TRAIN_ROOT="${PRECOMPILED_TRAIN_ROOT:-${PRECOMPILED_ROOT:-}}"
PRECOMPILED_EVAL_ROOT="${PRECOMPILED_EVAL_ROOT:-$PRECOMPILED_TRAIN_ROOT}"
SPLITS_PARQUET="${SPLITS_PARQUET:-${SPLITS:-}}"
RUNTIME_VOCAB_JSON="${RUNTIME_VOCAB_JSON:-${RUNTIME_VOC:-}}"
SPARSE_VOCAB_JSON="${SPARSE_VOCAB_JSON:-${SPARSE_VOC:-}}"
MEDTOK_VOC_DIR="${MEDTOK_VOC_DIR:-${MEDTOK_VOC_FINAL:-${MEDTOK_VOC_BASE:-}}}"
MEDTOK_ATTR_DIR="${MEDTOK_ATTR_DIR:-}"
MEDTOK_CROSSWALK_JSON="${MEDTOK_CROSSWALK_JSON:-${CROSSWALK_JSON:-}}"
CODES_PARQUET_PARENT_LOOKUP="${CODES_PARQUET_PARENT_LOOKUP:-${CODES_PARQUET:-}}"
ART_HOST="${ART_HOST:-${ART:-}}"
CODE2ID_PT="${CODE2ID_PT:-${ART_HOST:+$ART_HOST/code2id.pt}}"
STATS_PT="${STATS_PT:-${ART_HOST:+$ART_HOST/stats.pt}}"
CVAE_CKPT="${CVAE_CKPT:-${ART_HOST:+$ART_HOST/cvae_ckpt.pt}}"
TOKENIZER_CKPT="${TOKENIZER_CKPT:-${ART_HOST:+$ART_HOST/value_tokenizer.pt}}"
CKPT="${CKPT:-}"
INDEX_OUTPUT_HOST="${INDEX_OUTPUT_HOST:-$OUT_DIR_HOST/precedent_index$(if [[ "${PRECEDENT_MAX_SUBJECTS:-0}" =~ ^[0-9]+$ ]] && (( PRECEDENT_MAX_SUBJECTS > 0 )); then printf '_smoke_%s' "$PRECEDENT_MAX_SUBJECTS"; else printf '_full'; fi).pt}"

to_container_path() {
  local host_path="${1:-}"
  if [[ -z "$host_path" ]]; then
    printf '%s\n' ""
    return 0
  fi
  case "$host_path" in
    /dss/*|/workspace/*|/deps/*)
      printf '%s\n' "$host_path"
      ;;
    "$DSS_HOST"/*)
      printf '/dss/%s\n' "${host_path#"$DSS_HOST"/}"
      ;;
    "$DSS_HOST")
      printf '/dss\n'
      ;;
    "$REPO_DIR"/*)
      printf '/workspace/ehr-hier/%s\n' "${host_path#"$REPO_DIR"/}"
      ;;
    "$REPO_DIR")
      printf '/workspace/ehr-hier\n'
      ;;
    "$DEPS_HOST"/*)
      printf '/deps/%s\n' "${host_path#"$DEPS_HOST"/}"
      ;;
    "$DEPS_HOST")
      printf '/deps\n'
      ;;
    *)
      printf '%s\n' "$host_path"
      ;;
  esac
}

resolve_repo_path() {
  local path_value="${1:-}"
  if [[ -z "$path_value" ]]; then
    printf '%s\n' ""
    return 0
  fi
  case "$path_value" in
    /*) printf '%s\n' "$path_value" ;;
    *) printf '%s/%s\n' "$REPO_DIR" "$path_value" ;;
  esac
}

export REPO_CT="/workspace/ehr-hier"
export DSS_CT="/dss"
export DEPS_CT="/deps"
export OUT_DIR_CT
OUT_DIR_CT="$(to_container_path "$OUT_DIR_HOST")"
export PRECOMPILED_TRAIN_ROOT_CT
PRECOMPILED_TRAIN_ROOT_CT="$(to_container_path "$PRECOMPILED_TRAIN_ROOT")"
export PRECOMPILED_EVAL_ROOT_CT
PRECOMPILED_EVAL_ROOT_CT="$(to_container_path "$PRECOMPILED_EVAL_ROOT")"
export SPLITS_PARQUET_CT
SPLITS_PARQUET_CT="$(to_container_path "$SPLITS_PARQUET")"
export RUNTIME_VOCAB_JSON_CT
RUNTIME_VOCAB_JSON_CT="$(to_container_path "$RUNTIME_VOCAB_JSON")"
export SPARSE_VOCAB_JSON_CT
SPARSE_VOCAB_JSON_CT="$(to_container_path "$SPARSE_VOCAB_JSON")"
export MEDTOK_VOC_DIR_CT
MEDTOK_VOC_DIR_CT="$(to_container_path "$MEDTOK_VOC_DIR")"
export MEDTOK_ATTR_DIR_CT
MEDTOK_ATTR_DIR_CT="$(to_container_path "$MEDTOK_ATTR_DIR")"
export MEDTOK_CROSSWALK_JSON_CT
MEDTOK_CROSSWALK_JSON_CT="$(to_container_path "$MEDTOK_CROSSWALK_JSON")"
export CODES_PARQUET_PARENT_LOOKUP_CT
CODES_PARQUET_PARENT_LOOKUP_CT="$(to_container_path "$CODES_PARQUET_PARENT_LOOKUP")"
export CODE2ID_PT_CT
CODE2ID_PT_CT="$(to_container_path "$CODE2ID_PT")"
export STATS_PT_CT
STATS_PT_CT="$(to_container_path "$STATS_PT")"
export CVAE_CKPT_CT
CVAE_CKPT_CT="$(to_container_path "$CVAE_CKPT")"
export TOKENIZER_CKPT_CT
TOKENIZER_CKPT_CT="$(to_container_path "$TOKENIZER_CKPT")"
export CKPT_CT
CKPT_CT="$(to_container_path "$CKPT")"
export INDEX_OUTPUT_CT
INDEX_OUTPUT_CT="$(to_container_path "$INDEX_OUTPUT_HOST")"
export TOKENIZATION_YAML_CT
TOKENIZATION_YAML_CT="$(to_container_path "$(resolve_repo_path "$TOKENIZATION_YAML")")"
export STRUCTURAL_YAML_CT
STRUCTURAL_YAML_CT="$(to_container_path "$(resolve_repo_path "$STRUCTURAL_YAML")")"

repo_mount="${REPO_DIR}:/workspace/ehr-hier:ro"
mounts="${DSS_HOST}:/dss,${repo_mount}"
if [[ -d "$DEPS_HOST" ]]; then
  mounts="${mounts},${DEPS_HOST}:/deps"
fi

runtime_root="$(lrz_make_runtime_root)"
cleanup() {
  rm -rf "$runtime_root"
}
trap cleanup EXIT

cat >"$OUT_DIR_HOST/test_env_${SLURM_JOB_ID:-manual}.txt" <<EOF
IMAGE=$IMAGE
REPO_DIR=$REPO_DIR
DSS_HOST=$DSS_HOST
OUT_DIR_HOST=$OUT_DIR_HOST
OUT_DIR_CT=$OUT_DIR_CT
PRECOMPILED_TRAIN_ROOT=$PRECOMPILED_TRAIN_ROOT
PRECOMPILED_EVAL_ROOT=$PRECOMPILED_EVAL_ROOT
SPLITS_PARQUET=$SPLITS_PARQUET
RUNTIME_VOCAB_JSON=$RUNTIME_VOCAB_JSON
SPARSE_VOCAB_JSON=$SPARSE_VOCAB_JSON
MEDTOK_VOC_DIR=$MEDTOK_VOC_DIR
MEDTOK_CROSSWALK_JSON=$MEDTOK_CROSSWALK_JSON
CODES_PARQUET_PARENT_LOOKUP=$CODES_PARQUET_PARENT_LOOKUP
CODE2ID_PT=$CODE2ID_PT
STATS_PT=$STATS_PT
CVAE_CKPT=$CVAE_CKPT
TOKENIZER_CKPT=$TOKENIZER_CKPT
CKPT=$CKPT
INDEX_OUTPUT_HOST=$INDEX_OUTPUT_HOST
RUN_TEST_GATE=$RUN_TEST_GATE
RUN_EVAL_ONLY=$RUN_EVAL_ONLY
RUN_PRECEDENT_BUILD=$RUN_PRECEDENT_BUILD
TRAJECTORY_MODE=$TRAJECTORY_MODE
TRAIN_SPLIT=$TRAIN_SPLIT
EVAL_SPLIT=$EVAL_SPLIT
PRECEDENT_SPLIT=$PRECEDENT_SPLIT
MAX_EVAL_SUBJECTS=$MAX_EVAL_SUBJECTS
PRECEDENT_MAX_SUBJECTS=$PRECEDENT_MAX_SUBJECTS
TRAIN_EXTRA_ARGS=$TRAIN_EXTRA_ARGS
INDEX_EXTRA_ARGS=$INDEX_EXTRA_ARGS
EOF

cat >"$runtime_root/run_inside.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail

truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    *) return 1 ;;
  esac
}

require_env() {
  local name="$1"
  local value="${!name:-}"
  [[ -n "$value" ]] || {
    echo "Missing required env: $name" >&2
    exit 1
  }
}

cd "$REPO_CT"
mkdir -p "$OUT_DIR_CT"

if [[ -d "$DEPS_CT" ]]; then
  export PYTHONPATH="$REPO_CT:$DEPS_CT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$REPO_CT${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

project_vision_tests=(
  tests/test_subject_timeline_builder.py
  tests/test_precompiled_dataset.py
  tests/test_vocab_runtime_and_collator_remap.py
  tests/test_event_router_and_base_encoders.py
  tests/test_tokenization_freeze_check.py
  tests/test_event_composer_path.py
  tests/test_window_state_packet.py
  tests/test_episodic_memory.py
  tests/test_latent_health_state.py
  tests/test_marked_tte_losses.py
  tests/test_event_dt_nll_loss.py
  tests/test_precedent_memory.py
)

echo "Start on $(hostname) at $(date)"
echo "Repo: $REPO_CT"
echo "Out:  $OUT_DIR_CT"

if truthy "${RUN_TEST_GATE:-1}"; then
  pytest -q -p no:cacheprovider "${project_vision_tests[@]}" 2>&1 | tee "$OUT_DIR_CT/pytest_phase3_gate.log"
  python scripts/build_precedent_index.py --help >"$OUT_DIR_CT/build_precedent_index_help.txt"
  python scripts/train_transformer_v1.py --help >"$OUT_DIR_CT/train_transformer_v1_help.txt"
fi

if truthy "${RUN_EVAL_ONLY:-1}"; then
  require_env CKPT_CT
  require_env PRECOMPILED_TRAIN_ROOT_CT
  require_env PRECOMPILED_EVAL_ROOT_CT
  require_env SPLITS_PARQUET_CT
  require_env MEDTOK_VOC_DIR_CT
  require_env CODE2ID_PT_CT
  require_env TOKENIZER_CKPT_CT

  eval_out="$OUT_DIR_CT/eval_only"
  mkdir -p "$eval_out"

  eval_cmd=(
    python scripts/train_transformer_v1.py
    --eval_only
    --resume_from "$CKPT_CT"
    --precompiled_train_root "$PRECOMPILED_TRAIN_ROOT_CT"
    --precompiled_eval_root "$PRECOMPILED_EVAL_ROOT_CT"
    --splits_parquet "$SPLITS_PARQUET_CT"
    --train_split "${TRAIN_SPLIT:-train}"
    --eval_split "${EVAL_SPLIT:-tuning}"
    --trajectory_mode "${TRAJECTORY_MODE:-admission_chain}"
    --tokenization_yaml "$TOKENIZATION_YAML_CT"
    --structural_yaml "$STRUCTURAL_YAML_CT"
    --medtok_vocab_dir "$MEDTOK_VOC_DIR_CT"
    --code2id_pt "$CODE2ID_PT_CT"
    --tokenizer_ckpt "$TOKENIZER_CKPT_CT"
    --output_dir "$eval_out"
    --device "${DEVICE:-cuda}"
  )
  if [[ -n "${MAX_EVAL_SUBJECTS:-}" ]]; then
    eval_cmd+=(--max_eval_subjects "$MAX_EVAL_SUBJECTS")
  fi
  if [[ -n "${RUNTIME_VOCAB_JSON_CT:-}" ]]; then
    eval_cmd+=(--runtime_vocab_json "$RUNTIME_VOCAB_JSON_CT")
  fi
  if [[ -n "${SPARSE_VOCAB_JSON_CT:-}" ]]; then
    eval_cmd+=(--sparse_vocab_json "$SPARSE_VOCAB_JSON_CT")
  fi
  if [[ -n "${MEDTOK_ATTR_DIR_CT:-}" ]]; then
    eval_cmd+=(--medtok_attr_dir "$MEDTOK_ATTR_DIR_CT")
  fi
  if [[ -n "${MEDTOK_CROSSWALK_JSON_CT:-}" ]]; then
    eval_cmd+=(--medtok_crosswalk_json "$MEDTOK_CROSSWALK_JSON_CT")
  fi
  if [[ -n "${CODES_PARQUET_PARENT_LOOKUP_CT:-}" ]]; then
    eval_cmd+=(--codes_parquet_parent_lookup "$CODES_PARQUET_PARENT_LOOKUP_CT")
  fi
  if [[ -n "${STATS_PT_CT:-}" ]]; then
    eval_cmd+=(--stats_pt "$STATS_PT_CT")
  fi
  if [[ -n "${CVAE_CKPT_CT:-}" ]]; then
    eval_cmd+=(--cvae_ckpt "$CVAE_CKPT_CT")
  fi
  if truthy "${ALLOW_SMOKE_MEDTOK:-0}"; then
    eval_cmd+=(--allow_smoke_medtok)
  fi
  if [[ -n "${TRAIN_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_eval_args=( ${TRAIN_EXTRA_ARGS} )
    eval_cmd+=("${extra_eval_args[@]}")
  fi

  "${eval_cmd[@]}" 2>&1 | tee "$eval_out/eval_only.log"
fi

if truthy "${RUN_PRECEDENT_BUILD:-1}"; then
  require_env CKPT_CT
  require_env PRECOMPILED_TRAIN_ROOT_CT
  require_env INDEX_OUTPUT_CT

  index_cmd=(
    python scripts/build_precedent_index.py
    --precompiled_root "$PRECOMPILED_TRAIN_ROOT_CT"
    --split "${PRECEDENT_SPLIT:-train}"
    --trajectory_mode "${TRAJECTORY_MODE:-admission_chain}"
    --resume_from "$CKPT_CT"
    --output_path "$INDEX_OUTPUT_CT"
    --device "${DEVICE:-cuda}"
    --tokenization_yaml "$TOKENIZATION_YAML_CT"
    --structural_yaml "$STRUCTURAL_YAML_CT"
  )
  if [[ -n "${SPLITS_PARQUET_CT:-}" ]]; then
    index_cmd+=(--splits_parquet "$SPLITS_PARQUET_CT")
  fi
  if [[ -n "${RUNTIME_VOCAB_JSON_CT:-}" ]]; then
    index_cmd+=(--runtime_vocab_json "$RUNTIME_VOCAB_JSON_CT")
  fi
  if [[ -n "${SPARSE_VOCAB_JSON_CT:-}" ]]; then
    index_cmd+=(--sparse_vocab_json "$SPARSE_VOCAB_JSON_CT")
  fi
  if [[ -n "${MEDTOK_VOC_DIR_CT:-}" ]]; then
    index_cmd+=(--medtok_vocab_dir "$MEDTOK_VOC_DIR_CT")
  fi
  if [[ -n "${MEDTOK_ATTR_DIR_CT:-}" ]]; then
    index_cmd+=(--medtok_attr_dir "$MEDTOK_ATTR_DIR_CT")
  fi
  if [[ -n "${MEDTOK_CROSSWALK_JSON_CT:-}" ]]; then
    index_cmd+=(--medtok_crosswalk_json "$MEDTOK_CROSSWALK_JSON_CT")
  fi
  if [[ -n "${CODES_PARQUET_PARENT_LOOKUP_CT:-}" ]]; then
    index_cmd+=(--codes_parquet_parent_lookup "$CODES_PARQUET_PARENT_LOOKUP_CT")
  fi
  if [[ -n "${CODE2ID_PT_CT:-}" ]]; then
    index_cmd+=(--code2id_pt "$CODE2ID_PT_CT")
  fi
  if [[ -n "${TOKENIZER_CKPT_CT:-}" ]]; then
    index_cmd+=(--tokenizer_ckpt "$TOKENIZER_CKPT_CT")
  fi
  if truthy "${ALLOW_SMOKE_MEDTOK:-0}"; then
    index_cmd+=(--allow_smoke_medtok)
  fi
  if [[ "${PRECEDENT_MAX_SUBJECTS:-0}" =~ ^[0-9]+$ ]] && (( PRECEDENT_MAX_SUBJECTS > 0 )); then
    index_cmd+=(--max_subjects "$PRECEDENT_MAX_SUBJECTS")
  fi
  if [[ -n "${INDEX_EXTRA_ARGS:-}" ]]; then
    # shellcheck disable=SC2206
    extra_index_args=( ${INDEX_EXTRA_ARGS} )
    index_cmd+=("${extra_index_args[@]}")
  fi

  mkdir -p "$(dirname "$INDEX_OUTPUT_CT")"
  "${index_cmd[@]}" 2>&1 | tee "$OUT_DIR_CT/build_precedent_index.log"
fi

echo "Completed at $(date)"
EOF

chmod 700 "$runtime_root/run_inside.sh"

lrz_log "Using image: $IMAGE"
lrz_log "Output dir: $OUT_DIR_HOST"
lrz_log "Repo dir: $REPO_DIR"
lrz_log "Mounts: $mounts,$runtime_root:/workspace/runtime"

srun --ntasks=1 \
  --container-image="$IMAGE" \
  --container-mounts="$mounts,$runtime_root:/workspace/runtime" \
  bash /workspace/runtime/run_inside.sh
