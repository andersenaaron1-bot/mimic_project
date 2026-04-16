#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DEFAULT="$(cd "${SCRIPT_DIR%/}/.." && pwd)"

REPO="${REPO:-$REPO_DEFAULT}"
REPO_CT="${REPO_CT:-/workspace/ehr-hier}"
DSS_HOST="${DSS_HOST:-/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/${USER}}"
DEPS_HOST="${DEPS_HOST:-$DSS_HOST/containers/runtime_pydeps}"
GPU_MOUNTS="${GPU_MOUNTS:-$REPO:$REPO_CT,$DSS_HOST:/dss,$DEPS_HOST:/deps}"
IMAGE_GPU="${IMAGE_GPU:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
GPU_PARTITION="${GPU_PARTITION:-lrz-hgx-h100-94x4}"
GPU_GRES="${GPU_GRES:-gpu:1}"

ART="${ART:-/dss/etl/pipeline_artifacts_20260226_033711}"
SPLITS="${SPLITS:-/dss/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/subject_splits.parquet}"
CODES_PARQUET="${CODES_PARQUET:-/dss/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/codes.parquet}"
MEDTOK_VOC_FINAL_CT="${MEDTOK_VOC_FINAL_CT:-/dss/etl/tokenization_v1_eval/medtok_compressed_v1_exact}"
CROSSWALK_JSON_CT="${CROSSWALK_JSON_CT:-/dss/etl/tokenization_v1_eval/medtok_crosswalk_v1.json}"

timestamp() {
  date +%Y%m%d_%H%M%S
}

host_to_ct() {
  local p="$1"
  if [[ "$p" == "$DSS_HOST"* ]]; then
    printf '/dss%s' "${p#"$DSS_HOST"}"
    return 0
  fi
  if [[ "$p" == "$REPO"* ]]; then
    printf '%s%s' "$REPO_CT" "${p#"$REPO"}"
    return 0
  fi
  printf '%s' "$p"
}

pick_latest_precomp() {
  ls -td "$DSS_HOST"/etl/precompiled_transformer_v2_phase55_* 2>/dev/null | head -1 || true
}

show_usage() {
  cat <<'EOF'
Usage:
  bash scripts/lrz_submit_phase55_decision_runs.sh [--precomp-host HOST_PATH] [--run-root-host HOST_PATH]

Submits the three immediate decision runs:
  - fmv2_latent_nomemory
  - fmv2_transformer_nomemory
  - fmv2_latent_patient

Defaults:
  --precomp-host : latest /dss/.../etl/precompiled_transformer_v2_phase55_*
  --run-root-host: /dss/.../etl/fmv2_phase55_decision_<timestamp>
EOF
}

PRECOMP_HOST=""
RUN_HOST=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --precomp-host) PRECOMP_HOST="$2"; shift 2 ;;
    --run-root-host) RUN_HOST="$2"; shift 2 ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; show_usage; exit 1 ;;
  esac
done

if [[ -z "$PRECOMP_HOST" ]]; then
  PRECOMP_HOST="$(pick_latest_precomp)"
fi
[[ -n "$PRECOMP_HOST" && -d "$PRECOMP_HOST" ]] || { echo "Could not resolve precompile root." >&2; exit 1; }

if [[ -z "$RUN_HOST" ]]; then
  RUN_HOST="$DSS_HOST/etl/fmv2_phase55_decision_$(timestamp)"
fi

PRECOMP_CT="$(host_to_ct "$PRECOMP_HOST")"
RUN_CT="$(host_to_ct "$RUN_HOST")"
SPARSE_V2_CT="$PRECOMP_CT/token_vocab_sparse_phase55.json"

[[ -f "$PRECOMP_HOST/token_vocab_sparse_phase55.json" ]] || { echo "Missing sparse vocab under $PRECOMP_HOST" >&2; exit 1; }
[[ -f "$REPO/scripts/train_transformer_v1.py" ]] || { echo "Missing trainer script under $REPO" >&2; exit 1; }
grep -q -- '--objective_preset' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --objective_preset" >&2; exit 1; }
grep -q -- '--global_context_mode' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --global_context_mode" >&2; exit 1; }
grep -q -- '--carry_state_across_segments' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --carry_state_across_segments" >&2; exit 1; }
grep -q -- '--disable_exact_memory' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --disable_exact_memory" >&2; exit 1; }
grep -q -- '--disable_precedent_memory' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --disable_precedent_memory" >&2; exit 1; }

mkdir -p "$RUN_HOST/logs"

submit_run() {
  local name="$1"
  local mode="$2"
  local exact_flag="$3"
  local precedent_flag="$4"
  local out_ct="$RUN_CT/$name"
  local log_host="$RUN_HOST/logs/${name}_%j.out"
  local runtime_ct="$out_ct/runtime_vocab.json"

  sbatch --parsable \
    --job-name="$name" \
    --partition="$GPU_PARTITION" \
    --gres="$GPU_GRES" \
    --cpus-per-task=8 \
    --mem=64G \
    --time=12:00:00 \
    --output="$log_host" \
    --wrap="set -euo pipefail; mkdir -p '$out_ct'; srun --ntasks=1 --container-image='$IMAGE_GPU' --container-mounts='$GPU_MOUNTS' bash -lc 'set -euo pipefail; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH=$REPO_CT:/deps\${PYTHONPATH:+:\$PYTHONPATH}; cd $REPO_CT; python scripts/train_transformer_v1.py --precompiled_train_root $PRECOMP_CT/train_full --precompiled_eval_root $PRECOMP_CT/tuning_1024 --splits_parquet $SPLITS --train_split train --eval_split tuning --trajectory_mode full_subject --post_discharge_cutoff_days 31.0 --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --sparse_vocab_json $SPARSE_V2_CT --medtok_vocab_dir $MEDTOK_VOC_FINAL_CT --medtok_crosswalk_json $CROSSWALK_JSON_CT --codes_parquet_parent_lookup $CODES_PARQUET --runtime_vocab_json $runtime_ct --code2id_pt $ART/code2id.pt --stats_pt $ART/stats.pt --cvae_ckpt $ART/cvae_ckpt.pt --tokenizer_ckpt $ART/value_tokenizer.pt --output_dir $out_ct --objective_preset world_model_mttee --global_context_mode $mode --carry_state_across_segments $exact_flag $precedent_flag --batch_size 8 --eval_batch_size 8 --num_workers 6 --prefetch_factor 4 --grad_accum_steps 2 --max_steps 5000 --save_every_steps 1000 --eval_every_steps 1000 --max_eval_batches 64 --d_model 256 --num_heads 4 --d_ff 512 --num_local_layers 2 --num_global_layers 2 --num_chunk_layers 1 --max_windows 24 --max_chunks_per_window 4 --max_len_per_window 96 --device cuda --amp_dtype bf16 2>&1 | tee $out_ct/train.log'"
}

J_LATENT_NOMEM="$(submit_run fmv2_latent_nomemory latent_state --disable_exact_memory --disable_precedent_memory)"
J_TRANSFORMER_NOMEM="$(submit_run fmv2_transformer_nomemory transformer --disable_exact_memory --disable_precedent_memory)"
J_LATENT_PATIENT="$(submit_run fmv2_latent_patient latent_state --enable_exact_memory --disable_precedent_memory)"

printf 'PRECOMP_HOST=%s\n' "$PRECOMP_HOST"
printf 'RUN_HOST=%s\n' "$RUN_HOST"
printf 'J_LATENT_NOMEM=%s\n' "$J_LATENT_NOMEM"
printf 'J_TRANSFORMER_NOMEM=%s\n' "$J_TRANSFORMER_NOMEM"
printf 'J_LATENT_PATIENT=%s\n' "$J_LATENT_PATIENT"
squeue -j "$J_LATENT_NOMEM,$J_TRANSFORMER_NOMEM,$J_LATENT_PATIENT"
