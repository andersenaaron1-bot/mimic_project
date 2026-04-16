#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SOURCE_DEFAULT="$(cd "${SCRIPT_DIR%/}/.." && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "${SCRIPT_DIR%/}/lib/lrz_secure_runtime.sh"
lrz_setup_job_hardening

REPO="${REPO:-$REPO_SOURCE_DEFAULT}"
REPO_CT="${REPO_CT:-/workspace/ehr-hier}"
ROOT_DSS_DEFAULT="/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/${USER}"
DSS_HOST="${DSS_HOST:-$ROOT_DSS_DEFAULT}"
DEPS_HOST="${DEPS_HOST:-$DSS_HOST/containers/runtime_pydeps}"
DSS_ARTIFACTS_HOST="${DSS_ARTIFACTS_HOST:-/dss/artifacts}"

IMAGE_CPU="${IMAGE_CPU:-docker://ghcr.io#andersenaaron1-bot/ehr-pipeline-cpu:0.3.1-pipeline-cpu-min}"
IMAGE_LOCAL="${IMAGE_LOCAL:-$DSS_HOST/containers/pytorch-2.5.1-cuda12.4.sqsh}"
IMAGE_REMOTE="${IMAGE_REMOTE:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
IMAGE_GPU="${IMAGE_GPU:-$(lrz_pick_container_image "$IMAGE_LOCAL" "$IMAGE_REMOTE")}"

LRZ_CPU_PARTITION="${LRZ_CPU_PARTITION:-lrz-cpu}"
LRZ_CPU_QOS="${LRZ_CPU_QOS:-cpu}"
LRZ_GPU_PARTITION="${LRZ_GPU_PARTITION:-lrz-hgx-h100-94x4}"
LRZ_GPU_GRES="${LRZ_GPU_GRES:-gpu:1}"

CPU_MOUNTS_DEFAULT="$REPO:$REPO_CT,$DSS_HOST:/dss,$DEPS_HOST:/deps"
if [[ -d "$DSS_ARTIFACTS_HOST" ]]; then
  CPU_MOUNTS_DEFAULT="${CPU_MOUNTS_DEFAULT},$DSS_ARTIFACTS_HOST:/dss-artifacts"
fi
CPU_MOUNTS="${CPU_MOUNTS:-$CPU_MOUNTS_DEFAULT}"
GPU_MOUNTS="${GPU_MOUNTS:-$CPU_MOUNTS}"

pick_latest_dir() {
  local pattern="$1"
  local out=""
  out="$(ls -td $pattern 2>/dev/null | head -1 || true)"
  printf '%s' "$out"
}

pick_latest_find() {
  local root="$1"
  local name="$2"
  shift 2
  local out=""
  out="$(find "$root" "$@" -name "$name" 2>/dev/null | sort | tail -1 || true)"
  printf '%s' "$out"
}

pick_first_existing() {
  local cand=""
  for cand in "$@"; do
    [[ -n "$cand" ]] || continue
    if [[ -e "$cand" ]]; then
      printf '%s' "$cand"
      return 0
    fi
  done
  return 1
}

host_to_ct() {
  local p="$1"
  if [[ "$p" == "$DSS_HOST"* ]]; then
    printf '/dss%s' "${p#"$DSS_HOST"}"
    return 0
  fi
  if [[ "$p" == "$DSS_ARTIFACTS_HOST"* ]]; then
    printf '/dss-artifacts%s' "${p#"$DSS_ARTIFACTS_HOST"}"
    return 0
  fi
  if [[ "$p" == "$REPO"* ]]; then
    printf '%s%s' "$REPO_CT" "${p#"$REPO"}"
    return 0
  fi
  printf '%s' "$p"
}

timestamp() {
  date +%Y%m%d_%H%M%S
}

resolve_context() {
  ART_HOST="${ART_HOST:-$(pick_latest_dir "$DSS_HOST/etl/pipeline_artifacts_*")}"
  DB_HOST="${DB_HOST:-$(pick_first_existing \
    "$DSS_HOST/etl/meds_reader_db_mimiciv_20260226_033711/mimiciv.db" \
    "$(find "$DSS_HOST/etl" -type f -path '*/meds_reader_db_mimiciv_*/mimiciv.db' 2>/dev/null | sort | tail -1 || true)")}"
  SPLITS_HOST="${SPLITS_HOST:-$(pick_first_existing \
    "$DSS_HOST/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/subject_splits.parquet" \
    "$(find "$DSS_HOST/etl" -type f -path '*/MEDS_cohort/metadata/subject_splits.parquet' 2>/dev/null | sort | tail -1 || true)")}"
  CODES_PARQUET_HOST="${CODES_PARQUET_HOST:-$(pick_first_existing \
    "$DSS_HOST/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/codes.parquet" \
    "$(find "$DSS_HOST/etl" -type f -path '*/MEDS_cohort/metadata/codes.parquet' 2>/dev/null | sort | tail -1 || true)")}"
  MEDTOK_VOC_BASE_HOST="${MEDTOK_VOC_BASE_HOST:-$(pick_first_existing \
    "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/medtok_compressed_v1" \
    "$(find "$DSS_HOST/etl" -maxdepth 3 -type d -name 'medtok_compressed_v1' 2>/dev/null | sort | tail -1 || true)")}"
  MEDTOK_VOC_FINAL_HOST="${MEDTOK_VOC_FINAL_HOST:-$(pick_first_existing \
    "$DSS_HOST/etl/tokenization_v1_eval/medtok_compressed_v1_exact" \
    "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/tokenization_v1_eval/medtok_compressed_v1_exact")}"
  CROSSWALK_JSON_HOST="${CROSSWALK_JSON_HOST:-$(pick_first_existing \
    "$DSS_HOST/etl/tokenization_v1_eval/medtok_crosswalk_v1.json" \
    "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/tokenization_v1_eval/medtok_crosswalk_v1.json" \
    "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/medtok_crosswalk_v1.json")}"
  MEDTOK_CODE2EMBEDS_HOST="${MEDTOK_CODE2EMBEDS_HOST:-$(pick_first_existing \
    "$DSS_HOST/artifacts/medtok/code2embeddings.json" \
    "/dss/artifacts/medtok/code2embeddings.json")}"

  [[ -n "$ART_HOST" && -d "$ART_HOST" ]] || lrz_die "Could not resolve ART_HOST."
  [[ -n "$DB_HOST" && -e "$DB_HOST" ]] || lrz_die "Could not resolve DB_HOST."
  [[ -n "$SPLITS_HOST" && -f "$SPLITS_HOST" ]] || lrz_die "Could not resolve SPLITS_HOST."
  [[ -n "$CODES_PARQUET_HOST" && -f "$CODES_PARQUET_HOST" ]] || lrz_die "Could not resolve CODES_PARQUET_HOST."
  [[ -n "$MEDTOK_VOC_BASE_HOST" && -d "$MEDTOK_VOC_BASE_HOST" ]] || lrz_die "Could not resolve MEDTOK_VOC_BASE_HOST."
  [[ -n "$MEDTOK_VOC_FINAL_HOST" && -d "$MEDTOK_VOC_FINAL_HOST" ]] || lrz_die "Could not resolve MEDTOK_VOC_FINAL_HOST."
  [[ -n "$CROSSWALK_JSON_HOST" && -f "$CROSSWALK_JSON_HOST" ]] || lrz_die "Could not resolve CROSSWALK_JSON_HOST."
  [[ -n "$MEDTOK_CODE2EMBEDS_HOST" && -f "$MEDTOK_CODE2EMBEDS_HOST" ]] || lrz_die "Could not resolve MEDTOK_CODE2EMBEDS_HOST."

  ART="${ART:-$(host_to_ct "$ART_HOST")}"
  DB="${DB:-$(host_to_ct "$DB_HOST")}"
  SPLITS="${SPLITS:-$(host_to_ct "$SPLITS_HOST")}"
  CODES_PARQUET="${CODES_PARQUET:-$(host_to_ct "$CODES_PARQUET_HOST")}"
  MEDTOK_VOC_BASE="${MEDTOK_VOC_BASE:-$(host_to_ct "$MEDTOK_VOC_BASE_HOST")}"
  MEDTOK_VOC_FINAL_CT="${MEDTOK_VOC_FINAL_CT:-$(host_to_ct "$MEDTOK_VOC_FINAL_HOST")}"
  CROSSWALK_JSON_CT="${CROSSWALK_JSON_CT:-$(host_to_ct "$CROSSWALK_JSON_HOST")}"
  MEDTOK_CODE2EMBEDS="${MEDTOK_CODE2EMBEDS:-$(host_to_ct "$MEDTOK_CODE2EMBEDS_HOST")}"

  CURRENT_PHASE55_PRECOMP_HOST="${CURRENT_PHASE55_PRECOMP_HOST:-$(pick_latest_dir "$DSS_HOST/etl/precompiled_transformer_v2_phase55_*")}"
  CURRENT_PHASE55_PRECOMP_CT="${CURRENT_PHASE55_PRECOMP_CT:-$(host_to_ct "$CURRENT_PHASE55_PRECOMP_HOST")}"
}

show_usage() {
  cat <<'EOF'
Usage:
  bash scripts/lrz_phase55.sh <subcommand> [args]

Subcommands:
  show
      Print the resolved canonical LRZ phase55 paths.

  precompile [--label LABEL] [--split train|tuning|both]
             [--out-host HOST_PATH]
      Build a fresh sparse vocab under the precompile root, then compile train
      and/or tuning precompiled timelines. Updates:
        /dss/.../etl/precompiled_transformer_v2_phase55_latest

  train-baseline [--precomp-host HOST_PATH] [--label LABEL]
      Launch the current core_marked_latent baseline against an existing
      phase55 precompiled root. Updates:
        /dss/.../etl/fmv2_phase55_latest

  window-preview [--precomp-host HOST_PATH] [--max-subjects N]
                 [--sample-seed S] [--label LABEL]
      Run fast representative window-sequence inspection directly from
      precompiled timelines. Dumps only:
        - ordered semantic window types
        - boundary switch actions
        - event frames that caused each switch
      Updates:
        /dss/.../etl/window_preview_phase55_latest

  window-preview-report [--input-host JSONL] [--top-k N]
                        [--label LABEL]
      Analyze a window-preview JSONL and summarize which frame sources are
      actually causing switches, including suspicious close_open examples.
      Updates:
        /dss/.../etl/window_preview_phase55_latest/window_sequence_report_*.json

  window-transition-check [--input-host JSONL] [--top-k N]
                          [--label LABEL]
      Run the strict transition-driver attribution check over a window-preview
      JSONL. This only scores transition-relevant sides of each action and
      reports:
        - transition-candidate opening/closing drivers
        - TRANSFER_TO bundles that also contain ICU_ADMISSION
        - terminal window-type distribution
        - example terminal INPATIENT/UNK timelines

  cleanup-report
      Print redundant v1/v1-eval artifact candidates that can be removed after
      consolidation.
EOF
}

update_latest_link() {
  local target="$1"
  local latest_link="$2"
  mkdir -p "$(dirname "$latest_link")"
  ln -sfn "$target" "$latest_link"
}

run_build_sparse_into_root() {
  local out_host="$1"
  local out_ct="$2"
  local sparse_host="$out_host/token_vocab_sparse_phase55.json"
  local sparse_ct="$out_ct/token_vocab_sparse_phase55.json"
  [[ -f "$sparse_host" ]] && {
    printf '%s\n' "$sparse_host"
    return 0
  }
  mkdir -p "$out_host"
  srun -p "$LRZ_CPU_PARTITION" \
    --qos="$LRZ_CPU_QOS" \
    --cpus-per-task=8 \
    --mem=32G \
    --time=00:30:00 \
    --container-image="$IMAGE_CPU" \
    --container-mounts="$CPU_MOUNTS" \
    bash -lc "set -euo pipefail; mkdir -p '$out_ct'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/build_sparse_vocab_contract.py --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --medtok_vocab_dir '$MEDTOK_VOC_FINAL_CT' --medtok_attr_dir artifacts/medtok_attrs --code2id_pt '$ART/code2id.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' --output_json '$sparse_ct' 2>&1 | tee '$out_ct/build_sparse_vocab_contract.log'"
  printf '%s\n' "$sparse_host"
}

assert_train_baseline_surface() {
  local trainer_py="$REPO/scripts/train_transformer_v1.py"
  [[ -f "$trainer_py" ]] || lrz_die "Missing trainer script: $trainer_py"
  grep -q -- '--objective_preset' "$trainer_py" || lrz_die "Current train_transformer_v1.py is missing --objective_preset. Pull the latest branch before training."
  grep -q -- '--global_context_mode' "$trainer_py" || lrz_die "Current train_transformer_v1.py is missing --global_context_mode. Pull the latest branch before training."
  grep -q -- '--carry_state_across_segments' "$trainer_py" || lrz_die "Current train_transformer_v1.py is missing --carry_state_across_segments. Pull the latest branch before training."
  grep -q -- '--disable_exact_memory' "$trainer_py" || lrz_die "Current train_transformer_v1.py is missing --disable_exact_memory. Pull the latest branch before training."
  grep -q -- '--disable_precedent_memory' "$trainer_py" || lrz_die "Current train_transformer_v1.py is missing --disable_precedent_memory. Pull the latest branch before training."
}

subcmd_show() {
  resolve_context
  cat <<EOF
REPO=$REPO
REPO_CT=$REPO_CT
DSS_HOST=$DSS_HOST
DEPS_HOST=$DEPS_HOST
IMAGE_CPU=$IMAGE_CPU
IMAGE_GPU=$IMAGE_GPU
CPU_MOUNTS=$CPU_MOUNTS
GPU_MOUNTS=$GPU_MOUNTS
ART_HOST=$ART_HOST
DB_HOST=$DB_HOST
SPLITS_HOST=$SPLITS_HOST
CODES_PARQUET_HOST=$CODES_PARQUET_HOST
MEDTOK_VOC_BASE_HOST=$MEDTOK_VOC_BASE_HOST
MEDTOK_VOC_FINAL_HOST=$MEDTOK_VOC_FINAL_HOST
CROSSWALK_JSON_HOST=$CROSSWALK_JSON_HOST
MEDTOK_CODE2EMBEDS_HOST=$MEDTOK_CODE2EMBEDS_HOST
ART=$ART
DB=$DB
SPLITS=$SPLITS
CODES_PARQUET=$CODES_PARQUET
MEDTOK_VOC_BASE=$MEDTOK_VOC_BASE
MEDTOK_VOC_FINAL_CT=$MEDTOK_VOC_FINAL_CT
CROSSWALK_JSON_CT=$CROSSWALK_JSON_CT
MEDTOK_CODE2EMBEDS=$MEDTOK_CODE2EMBEDS
CURRENT_PHASE55_PRECOMP_HOST=$CURRENT_PHASE55_PRECOMP_HOST
CURRENT_PHASE55_PRECOMP_CT=$CURRENT_PHASE55_PRECOMP_CT
LATEST_PRECOMP_LINK=$DSS_HOST/etl/precompiled_transformer_v2_phase55_latest
LATEST_RUN_LINK=$DSS_HOST/etl/fmv2_phase55_latest
LATEST_WINDOW_PREVIEW_LINK=$DSS_HOST/etl/window_preview_phase55_latest
EOF
}

subcmd_precompile() {
  resolve_context
  local label="medmark_fullsubject"
  local split="both"
  local out_host=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --label) label="$2"; shift 2 ;;
      --split) split="$2"; shift 2 ;;
      --out-host) out_host="$2"; shift 2 ;;
      *) lrz_die "Unknown precompile arg: $1" ;;
    esac
  done
  if [[ -z "$out_host" ]]; then
    out_host="$DSS_HOST/etl/precompiled_transformer_v2_phase55_${label}_$(timestamp)"
  fi
  local out_ct
  out_ct="$(host_to_ct "$out_host")"
  local sparse_host
  sparse_host="$(run_build_sparse_into_root "$out_host" "$out_ct")"
  local sparse_ct
  sparse_ct="$(host_to_ct "$sparse_host")"

  mkdir -p "$out_host/logs"
  if [[ "$split" == "train" || "$split" == "both" ]]; then
    srun -p "$LRZ_CPU_PARTITION" \
      --qos="$LRZ_CPU_QOS" \
      --cpus-per-task=32 \
      --mem=128G \
      --time=09:00:00 \
      --container-image="$IMAGE_CPU" \
      --container-mounts="$CPU_MOUNTS" \
      bash -lc "set -euo pipefail; mkdir -p '$out_ct'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/compile_timelines_v2.py --meds_reader_db '$DB' --splits_parquet '$SPLITS' --split train --output_dir '$out_ct/train_full' --num_workers 32 --num_output_shards 256 --chunksize 1 --progress_every 500 --trajectory_mode full_subject --post_discharge_cutoff_days 31.0 --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --sparse_vocab_json '$sparse_ct' --medtok_vocab_dir '$MEDTOK_VOC_FINAL_CT' --medtok_crosswalk_json '$CROSSWALK_JSON_CT' --codes_parquet_parent_lookup '$CODES_PARQUET' --code2id_pt '$ART/code2id.pt' --stats_pt '$ART/stats.pt' --cvae_ckpt '$ART/cvae_ckpt.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' 2>&1 | tee '$out_ct/compile_train_full.log'"
  fi
  if [[ "$split" == "tuning" || "$split" == "both" ]]; then
    srun -p "$LRZ_CPU_PARTITION" \
      --qos="$LRZ_CPU_QOS" \
      --cpus-per-task=16 \
      --mem=64G \
      --time=02:00:00 \
      --container-image="$IMAGE_CPU" \
      --container-mounts="$CPU_MOUNTS" \
      bash -lc "set -euo pipefail; mkdir -p '$out_ct'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/compile_timelines_v2.py --meds_reader_db '$DB' --splits_parquet '$SPLITS' --split tuning --max_subjects 1024 --sample_seed 1337 --output_dir '$out_ct/tuning_1024' --num_workers 16 --num_output_shards 64 --chunksize 1 --progress_every 100 --trajectory_mode full_subject --post_discharge_cutoff_days 31.0 --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --sparse_vocab_json '$sparse_ct' --medtok_vocab_dir '$MEDTOK_VOC_FINAL_CT' --medtok_crosswalk_json '$CROSSWALK_JSON_CT' --codes_parquet_parent_lookup '$CODES_PARQUET' --code2id_pt '$ART/code2id.pt' --stats_pt '$ART/stats.pt' --cvae_ckpt '$ART/cvae_ckpt.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' 2>&1 | tee '$out_ct/compile_tuning_1024.log'"
  fi

  update_latest_link "$out_host" "$DSS_HOST/etl/precompiled_transformer_v2_phase55_latest"
  printf 'PRECOMP_HOST=%s\n' "$out_host"
  printf 'PRECOMP_CT=%s\n' "$out_ct"
  printf 'SPARSE_VOC_HOST=%s\n' "$sparse_host"
  printf 'LATEST_LINK=%s\n' "$DSS_HOST/etl/precompiled_transformer_v2_phase55_latest"
}

subcmd_train_baseline() {
  resolve_context
  assert_train_baseline_surface
  local precomp_host="${CURRENT_PHASE55_PRECOMP_HOST:-}"
  local label="core_marked_latent"
  local run_host=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --precomp-host) precomp_host="$2"; shift 2 ;;
      --label) label="$2"; shift 2 ;;
      --out-host) run_host="$2"; shift 2 ;;
      *) lrz_die "Unknown train-baseline arg: $1" ;;
    esac
  done
  [[ -n "$precomp_host" && -d "$precomp_host" ]] || lrz_die "Set --precomp-host or create a phase55 precompile first."
  local precomp_ct sparse_host sparse_ct runtime_host runtime_ct
  precomp_ct="$(host_to_ct "$precomp_host")"
  sparse_host="$precomp_host/token_vocab_sparse_phase55.json"
  sparse_ct="$(host_to_ct "$sparse_host")"
  runtime_host="$precomp_host/runtime_vocab_phase55_full.json"
  runtime_ct="$(host_to_ct "$runtime_host")"
  [[ -f "$sparse_host" ]] || lrz_die "Sparse vocab missing under precompile root: $sparse_host"
  if [[ -z "$run_host" ]]; then
    run_host="$DSS_HOST/etl/fmv2_phase55_${label}_$(timestamp)"
  fi
  local run_ct
  run_ct="$(host_to_ct "$run_host")"
  mkdir -p "$run_host/logs"

  srun -p "$LRZ_GPU_PARTITION" \
    --gres="$LRZ_GPU_GRES" \
    --cpus-per-task=8 \
    --mem=64G \
    --time=12:00:00 \
    --container-image="$IMAGE_GPU" \
    --container-mounts="$GPU_MOUNTS" \
    bash -lc "set -euo pipefail; mkdir -p '$run_ct/$label'; export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/train_transformer_v1.py --precompiled_train_root '$precomp_ct/train_full' --precompiled_eval_root '$precomp_ct/tuning_1024' --splits_parquet '$SPLITS' --train_split train --eval_split tuning --trajectory_mode full_subject --post_discharge_cutoff_days 31.0 --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --sparse_vocab_json '$sparse_ct' --medtok_vocab_dir '$MEDTOK_VOC_FINAL_CT' --medtok_crosswalk_json '$CROSSWALK_JSON_CT' --codes_parquet_parent_lookup '$CODES_PARQUET' --runtime_vocab_json '$runtime_ct' --code2id_pt '$ART/code2id.pt' --stats_pt '$ART/stats.pt' --cvae_ckpt '$ART/cvae_ckpt.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' --output_dir '$run_ct/$label' --objective_preset world_model_mttee --global_context_mode latent_state --carry_state_across_segments --disable_exact_memory --disable_precedent_memory --batch_size 8 --eval_batch_size 8 --num_workers 6 --prefetch_factor 4 --grad_accum_steps 2 --max_steps 5000 --save_every_steps 1000 --eval_every_steps 1000 --max_eval_batches 64 --d_model 256 --num_heads 4 --d_ff 512 --num_local_layers 2 --num_global_layers 2 --num_chunk_layers 1 --max_windows 24 --max_chunks_per_window 4 --max_len_per_window 96 --device cuda --seed 1337 2>&1 | tee '$run_ct/$label/train.log'"

  update_latest_link "$run_host" "$DSS_HOST/etl/fmv2_phase55_latest"
  printf 'RUN_HOST=%s\n' "$run_host"
  printf 'RUN_CT=%s\n' "$run_ct"
  printf 'LATEST_LINK=%s\n' "$DSS_HOST/etl/fmv2_phase55_latest"
}

subcmd_window_preview() {
  local precomp_host="${CURRENT_PHASE55_PRECOMP_HOST:-$(pick_latest_dir "$DSS_HOST/etl/precompiled_transformer_v2_phase55_*")}"
  local label="train"
  local max_subjects="500"
  local sample_seed="13"
  local out_host=""
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --precomp-host) precomp_host="$2"; shift 2 ;;
      --label) label="$2"; shift 2 ;;
      --max-subjects) max_subjects="$2"; shift 2 ;;
      --sample-seed) sample_seed="$2"; shift 2 ;;
      --out-host) out_host="$2"; shift 2 ;;
      *) lrz_die "Unknown window-preview arg: $1" ;;
    esac
  done
  [[ -n "$precomp_host" && -d "$precomp_host" ]] || lrz_die "Set --precomp-host or create a phase55 precompile first."
  local precomp_ct
  precomp_ct="$(host_to_ct "$precomp_host")"
  if [[ -z "$out_host" ]]; then
    out_host="$DSS_HOST/etl/window_preview_phase55_${label}_${max_subjects}_$(timestamp)"
  fi
  local out_ct
  out_ct="$(host_to_ct "$out_host")"
  mkdir -p "$out_host"

  srun -p "$LRZ_CPU_PARTITION" \
    --qos="$LRZ_CPU_QOS" \
    --cpus-per-task=16 \
    --mem=64G \
    --time=01:00:00 \
    --container-image="$IMAGE_CPU" \
    --container-mounts="$CPU_MOUNTS" \
    bash -lc "set -euo pipefail; mkdir -p '$out_ct'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/inspect_precompiled_window_sequences.py --precompiled_root '$precomp_ct/train_full' --max_subjects '$max_subjects' --sample_seed '$sample_seed' --progress_every 50 --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --output_jsonl '$out_ct/window_sequences_train_${max_subjects}.jsonl' --output_summary_json '$out_ct/window_sequences_train_${max_subjects}_summary.json'"

  update_latest_link "$out_host" "$DSS_HOST/etl/window_preview_phase55_latest"
  printf 'WINDOW_PREVIEW_HOST=%s\n' "$out_host"
  printf 'WINDOW_PREVIEW_CT=%s\n' "$out_ct"
  printf 'LATEST_LINK=%s\n' "$DSS_HOST/etl/window_preview_phase55_latest"
}

subcmd_window_preview_report() {
  local latest_host="${DSS_HOST}/etl/window_preview_phase55_latest"
  local input_host=""
  local top_k="40"
  local label="window_sequence_report"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --input-host) input_host="$2"; shift 2 ;;
      --top-k) top_k="$2"; shift 2 ;;
      --label) label="$2"; shift 2 ;;
      *) lrz_die "Unknown window-preview-report arg: $1" ;;
    esac
  done
  if [[ -z "$input_host" ]]; then
    input_host="$(ls -t "${latest_host}"/window_sequences_train_*.jsonl 2>/dev/null | head -1 || true)"
  fi
  [[ -n "$input_host" && -f "$input_host" ]] || lrz_die "Set --input-host to a window sequence JSONL."
  local report_host report_ct input_ct
  report_host="$(dirname "$input_host")/${label}_$(timestamp).json"
  report_ct="$(host_to_ct "$report_host")"
  input_ct="$(host_to_ct "$input_host")"

  srun -p "$LRZ_CPU_PARTITION" \
    --qos="$LRZ_CPU_QOS" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:20:00 \
    --container-image="$IMAGE_CPU" \
    --container-mounts="$CPU_MOUNTS" \
    bash -lc "set -euo pipefail; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/analyze_window_sequence_preview.py --input_jsonl '$input_ct' --top_k '$top_k' --output_json '$report_ct'"

  printf 'WINDOW_PREVIEW_REPORT_HOST=%s\n' "$report_host"
  printf 'WINDOW_PREVIEW_REPORT_CT=%s\n' "$report_ct"
}

subcmd_window_transition_check() {
  local latest_host="${DSS_HOST}/etl/window_preview_phase55_latest"
  local input_host=""
  local top_k="40"
  local label="window_transition_check"
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --input-host) input_host="$2"; shift 2 ;;
      --top-k) top_k="$2"; shift 2 ;;
      --label) label="$2"; shift 2 ;;
      *) lrz_die "Unknown window-transition-check arg: $1" ;;
    esac
  done
  if [[ -z "$input_host" ]]; then
    input_host="$(ls -t "${latest_host}"/window_sequences_train_*.jsonl 2>/dev/null | head -1 || true)"
  fi
  [[ -n "$input_host" && -f "$input_host" ]] || lrz_die "Set --input-host to a window sequence JSONL."
  local report_host report_ct input_ct
  report_host="$(dirname "$input_host")/${label}_$(timestamp).json"
  report_ct="$(host_to_ct "$report_host")"
  input_ct="$(host_to_ct "$input_host")"

  srun -p "$LRZ_CPU_PARTITION" \
    --qos="$LRZ_CPU_QOS" \
    --cpus-per-task=4 \
    --mem=16G \
    --time=00:20:00 \
    --container-image="$IMAGE_CPU" \
    --container-mounts="$CPU_MOUNTS" \
    bash -lc "set -euo pipefail; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/check_window_transition_integrity.py --input_jsonl '$input_ct' --top_k '$top_k' --output_json '$report_ct'"

  printf 'WINDOW_TRANSITION_CHECK_HOST=%s\n' "$report_host"
  printf 'WINDOW_TRANSITION_CHECK_CT=%s\n' "$report_ct"
}

print_cleanup_entry() {
  local path="$1"
  [[ -e "$path" ]] || return 0
  local size="unknown"
  size="$(du -sh "$path" 2>/dev/null | awk '{print $1}' || true)"
  printf '  %s\t%s\n' "${size:-unknown}" "$path"
}

subcmd_cleanup_report() {
  resolve_context
  cat <<EOF
Cleanup candidates after phase55 consolidation
=============================================
Safe to review first:
EOF
  print_cleanup_entry "$DSS_HOST/etl/tokenization_v1_eval"
  print_cleanup_entry "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/tokenization_v1_eval"
  print_cleanup_entry "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/runtime_vocab_v1.json"
  print_cleanup_entry "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/runtime_vocab_compact_v1_20k_cpu.json"
  print_cleanup_entry "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/runtime_vocab_compact_v1_20k_parallel.json"
  print_cleanup_entry "$DSS_HOST/etl/pipeline_artifacts_20260226_033711/token_vocab_sparse_v1.json"
  cat <<EOF

Likely obsolete precompiled roots to review manually:
EOF
  print_cleanup_entry "$DSS_HOST/etl/precompiled_transformer_v2_phase5_admchain_20260414_004159"
  print_cleanup_entry "$DSS_HOST/etl/precompiled_transformer_v2_fullsubject_orfix_v1"
  print_cleanup_entry "$DSS_HOST/etl/precompiled_transformer_v2_sizecheck_1k"
  cat <<EOF

Current phase55 precompile retained:
  ${CURRENT_PHASE55_PRECOMP_HOST:-<none>}
EOF
}

main() {
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    help|-h|--help) show_usage ;;
    show) subcmd_show "$@" ;;
    precompile) subcmd_precompile "$@" ;;
    train-baseline) subcmd_train_baseline "$@" ;;
    window-preview) subcmd_window_preview "$@" ;;
    window-preview-report) subcmd_window_preview_report "$@" ;;
    window-transition-check) subcmd_window_transition_check "$@" ;;
    cleanup-report) subcmd_cleanup_report "$@" ;;
    *) lrz_die "Unknown subcommand: $cmd" ;;
  esac
}

main "$@"
