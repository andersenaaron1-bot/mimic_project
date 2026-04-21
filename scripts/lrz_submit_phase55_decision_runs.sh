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
GPU_PARTITION="${GPU_PARTITION:-auto}"
GPU_GRES="${GPU_GRES:-gpu:1}"
AMP_DTYPE="${AMP_DTYPE:-bf16}"
LRZ_PUBLIC_GPU_PARTITIONS_FAST="${LRZ_PUBLIC_GPU_PARTITIONS_FAST:-lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8}"
LRZ_PUBLIC_GPU_PARTITIONS_LEGACY="${LRZ_PUBLIC_GPU_PARTITIONS_LEGACY:-lrz-dgx-1-v100x8,lrz-dgx-1-p100x8,lrz-hpe-p100x4,lrz-v100x2}"

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

normalize_partition_csv() {
  local raw="$1"
  printf '%s\n' "$raw" | tr ' ' ',' | tr -s ',' | sed 's/^,//; s/,$//'
}

discover_gpu_partitions() {
  local allow_csv="$1"
  if ! command -v sinfo >/dev/null 2>&1; then
    return 1
  fi
  sinfo -h -o '%P|%G|%a' \
    | awk -F'|' '
        $2 ~ /gpu/ && $3 == "up" {
          gsub(/\*/, "", $1)
          print $1
        }
      ' \
    | awk -v allow=",$allow_csv," 'index(allow, "," $0 ",") > 0 { print }' \
    | sort -u \
    | paste -sd, -
}

resolve_gpu_partitions() {
  local requested="$1"
  if [[ -z "$requested" || "$requested" == "auto" ]]; then
    local discovered=""
    discovered="$(discover_gpu_partitions "$LRZ_PUBLIC_GPU_PARTITIONS_FAST" || true)"
    if [[ -n "$discovered" ]]; then
      normalize_partition_csv "$discovered"
      return 0
    fi
    printf '%s\n' "$LRZ_PUBLIC_GPU_PARTITIONS_FAST"
    return 0
  fi
  if [[ "$requested" == "auto-public-legacy" || "$requested" == "auto-legacy" ]]; then
    local allow_csv="$LRZ_PUBLIC_GPU_PARTITIONS_FAST,$LRZ_PUBLIC_GPU_PARTITIONS_LEGACY"
    local discovered=""
    discovered="$(discover_gpu_partitions "$allow_csv" || true)"
    if [[ -n "$discovered" ]]; then
      normalize_partition_csv "$discovered"
      return 0
    fi
    normalize_partition_csv "$allow_csv"
    return 0
  fi
  normalize_partition_csv "$requested"
}

show_usage() {
  cat <<'EOF'
Usage:
  bash scripts/lrz_submit_phase55_decision_runs.sh [--precomp-host HOST_PATH] [--run-root-host HOST_PATH] [--gpu-partitions PARTITIONS] [--dry-run]

Submits the three immediate decision runs:
  - fmv2_latent_nomemory
  - fmv2_transformer_nomemory
  - fmv2_latent_patient

Defaults:
  --precomp-host : latest /dss/.../etl/precompiled_transformer_v2_phase55_*
  --run-root-host: /dss/.../etl/fmv2_phase55_decision_<timestamp>
  --gpu-partitions: auto, auto-public-legacy, or comma/space-separated Slurm partitions
  --dry-run      : generate and syntax-check sbatch files without submitting
  --print-gpu-partitions: print discovered GPU partitions and exit

Default auto partitions are public LRZ non-MIG A100/H100 partitions from the
repo-local LRZ compute docs:
  lrz-hgx-h100-94x4,lrz-hgx-a100-80x4,lrz-dgx-a100-80x8

MCML, test, and MIG partitions are intentionally excluded from auto. V100/P100
legacy public partitions are excluded from auto because the default command uses
bf16. To use them explicitly, set AMP_DTYPE=fp16.
EOF
}

PRECOMP_HOST=""
RUN_HOST=""
DRY_RUN=0
PRINT_GPU_PARTITIONS=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --precomp-host) PRECOMP_HOST="$2"; shift 2 ;;
    --run-root-host) RUN_HOST="$2"; shift 2 ;;
    --gpu-partitions|--partition|--partitions) GPU_PARTITION="$2"; shift 2 ;;
    --dry-run|--validate-only) DRY_RUN=1; shift ;;
    --print-gpu-partitions) PRINT_GPU_PARTITIONS=1; shift ;;
    -h|--help) show_usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; show_usage; exit 1 ;;
  esac
done

GPU_PARTITION_REQUESTED="$GPU_PARTITION"
GPU_PARTITION="$(resolve_gpu_partitions "$GPU_PARTITION")"

if [[ "$PRINT_GPU_PARTITIONS" == "1" ]]; then
  printf '%s\n' "$GPU_PARTITION"
  exit 0
fi

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
[[ -f "$PRECOMP_HOST/train_full/index.csv" ]] || { echo "Missing train index under $PRECOMP_HOST/train_full" >&2; exit 1; }
[[ -f "$PRECOMP_HOST/train_full/manifest.json" ]] || { echo "Missing train manifest under $PRECOMP_HOST/train_full" >&2; exit 1; }
[[ -f "$PRECOMP_HOST/tuning_1024/index.csv" ]] || { echo "Missing tuning index under $PRECOMP_HOST/tuning_1024" >&2; exit 1; }
[[ -f "$PRECOMP_HOST/tuning_1024/manifest.json" ]] || { echo "Missing tuning manifest under $PRECOMP_HOST/tuning_1024" >&2; exit 1; }
[[ -f "$REPO/scripts/train_transformer_v1.py" ]] || { echo "Missing trainer script under $REPO" >&2; exit 1; }
grep -q -- '--objective_preset' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --objective_preset" >&2; exit 1; }
grep -q -- '--global_context_mode' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --global_context_mode" >&2; exit 1; }
grep -q -- '--carry_state_across_segments' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --carry_state_across_segments" >&2; exit 1; }
grep -q -- '--disable_exact_memory' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --disable_exact_memory" >&2; exit 1; }
grep -q -- '--disable_precedent_memory' "$REPO/scripts/train_transformer_v1.py" || { echo "Trainer surface is stale: missing --disable_precedent_memory" >&2; exit 1; }
[[ -n "$IMAGE_GPU" ]] || { echo "IMAGE_GPU is empty." >&2; exit 1; }
[[ -n "$GPU_MOUNTS" && "$GPU_MOUNTS" != :* ]] || { echo "GPU_MOUNTS is empty or malformed: $GPU_MOUNTS" >&2; exit 1; }
[[ -n "$GPU_PARTITION" ]] || { echo "GPU_PARTITION resolved empty." >&2; exit 1; }
if [[ "$GPU_PARTITION_REQUESTED" == *legacy* || "$GPU_PARTITION" == *v100* || "$GPU_PARTITION" == *p100* ]]; then
  [[ "$AMP_DTYPE" == "fp16" ]] || {
    echo "Legacy V100/P100 partitions require AMP_DTYPE=fp16; current AMP_DTYPE=$AMP_DTYPE" >&2
    exit 1
  }
fi
[[ "$PRECOMP_CT" == /dss/* ]] || { echo "PRECOMP_CT should be a container /dss path, got $PRECOMP_CT" >&2; exit 1; }
[[ "$RUN_CT" == /dss/* ]] || { echo "RUN_CT should be a container /dss path, got $RUN_CT" >&2; exit 1; }

mkdir -p "$RUN_HOST/logs" "$RUN_HOST/submit_meta"
touch "$RUN_HOST/.write_test" && rm -f "$RUN_HOST/.write_test"

submit_run() {
  local name="$1"
  local mode="$2"
  local exact_flag="$3"
  local precedent_flag="$4"
  local out_host="$RUN_HOST/$name"
  local out_ct="$RUN_CT/$name"
  local log_host="$RUN_HOST/logs/${name}_%j.out"
  local runtime_ct="$out_ct/runtime_vocab.json"
  local submit_host="$RUN_HOST/submit_meta/${name}.sbatch"

  cat >"$submit_host" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=$name
#SBATCH --partition=$GPU_PARTITION
#SBATCH --gres=$GPU_GRES
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=$log_host
set -euo pipefail
mkdir -p '$out_host'
srun --ntasks=1 --container-image='$IMAGE_GPU' --container-mounts='$GPU_MOUNTS' bash -lc "set -euo pipefail; mkdir -p '$out_ct'; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/train_transformer_v1.py --precompiled_train_root '$PRECOMP_CT/train_full' --precompiled_eval_root '$PRECOMP_CT/tuning_1024' --splits_parquet '$SPLITS' --train_split train --eval_split tuning --trajectory_mode full_subject --post_discharge_cutoff_days 31.0 --tokenization_yaml configs/data/tokenization_v1.yaml --structural_yaml configs/data/structural_codes.yaml --sparse_vocab_json '$SPARSE_V2_CT' --medtok_vocab_dir '$MEDTOK_VOC_FINAL_CT' --medtok_crosswalk_json '$CROSSWALK_JSON_CT' --codes_parquet_parent_lookup '$CODES_PARQUET' --runtime_vocab_json '$runtime_ct' --code2id_pt '$ART/code2id.pt' --stats_pt '$ART/stats.pt' --cvae_ckpt '$ART/cvae_ckpt.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' --output_dir '$out_ct' --objective_preset world_model_mttee --global_context_mode '$mode' --carry_state_across_segments $exact_flag $precedent_flag --batch_size 8 --eval_batch_size 8 --num_workers 6 --prefetch_factor 4 --grad_accum_steps 2 --max_steps 5000 --save_every_steps 1000 --eval_every_steps 1000 --max_eval_batches 64 --d_model 256 --num_heads 4 --d_ff 512 --num_local_layers 2 --num_global_layers 2 --num_chunk_layers 1 --max_windows 24 --max_chunks_per_window 4 --max_len_per_window 96 --device cuda --amp_dtype '$AMP_DTYPE' 2>&1 | tee '$out_ct/train.log'"
EOF
  chmod 700 "$submit_host"
  bash -n "$submit_host"
  if command -v sbatch >/dev/null 2>&1 && sbatch --help 2>&1 | grep -q -- '--test-only'; then
    sbatch --test-only "$submit_host" >/dev/null
  fi
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%s\n' "$submit_host"
    return 0
  fi
  sbatch --parsable "$submit_host"
}

J_LATENT_NOMEM="$(submit_run fmv2_latent_nomemory latent_state --disable_exact_memory --disable_precedent_memory)"
J_TRANSFORMER_NOMEM="$(submit_run fmv2_transformer_nomemory transformer --disable_exact_memory --disable_precedent_memory)"
J_LATENT_PATIENT="$(submit_run fmv2_latent_patient latent_state --enable_exact_memory --disable_precedent_memory)"

printf 'PRECOMP_HOST=%s\n' "$PRECOMP_HOST"
printf 'RUN_HOST=%s\n' "$RUN_HOST"
printf 'GPU_PARTITION=%s\n' "$GPU_PARTITION"
printf 'AMP_DTYPE=%s\n' "$AMP_DTYPE"
printf 'J_LATENT_NOMEM=%s\n' "$J_LATENT_NOMEM"
printf 'J_TRANSFORMER_NOMEM=%s\n' "$J_TRANSFORMER_NOMEM"
printf 'J_LATENT_PATIENT=%s\n' "$J_LATENT_PATIENT"
if [[ "$DRY_RUN" == "1" ]]; then
  printf 'DRY_RUN=1; no jobs submitted.\n'
else
  squeue -j "$J_LATENT_NOMEM,$J_TRANSFORMER_NOMEM,$J_LATENT_PATIENT"
fi
