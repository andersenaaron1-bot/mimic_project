#!/usr/bin/env bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH -t 1-23:00:00
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "$SCRIPT_DIR/lib/lrz_secure_runtime.sh"
lrz_setup_job_hardening

IMAGE_LOCAL="${IMAGE_LOCAL:-/dss/dssfs04/PROJECT/containers/ehr-train.sqsh}"
IMAGE_REMOTE="${IMAGE_REMOTE:-docker://nvcr.io/nvidia/pytorch:24.10-py3}"
IMAGE="${IMAGE:-$(lrz_pick_container_image "$IMAGE_LOCAL" "$IMAGE_REMOTE")}"

REPO_DIR="${REPO_DIR:-/dss/dsshome1/$USER/ehr-hier}"
DATA_DIR="${DATA_DIR:-/dss/dssfs04/PROJECT/datasets}"
DATA_ENC_ARCHIVE="${DATA_ENC_ARCHIVE:-}"     # Optional .tar.zst.age

OUT_DIR="${OUT_DIR:-/dss/dssfs04/PROJECT/outputs}"
OUT_ENC_ARCHIVE="${OUT_ENC_ARCHIVE:-}"       # Optional .tar.zst.age
RUN_META_DIR="${RUN_META_DIR:-${OUT_DIR%/}/_run_meta}"

STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-auto}"     # auto|true|false
KEEP_PLAINTEXT_OUT="${KEEP_PLAINTEXT_OUT:-1}"
REPO_READONLY="${REPO_READONLY:-1}"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
TRAIN_ENTRY="${TRAIN_ENTRY:-train.py}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-data.root=/workspace/data out.dir=/workspace/outputs trainer.max_steps=${MAX_STEPS}}"
LAUNCH_CMD="${LAUNCH_CMD:-torchrun --nproc_per_node=${GPUS_PER_NODE} ${TRAIN_ENTRY} ${TRAIN_OVERRIDES}}"

mkdir -p "$OUT_DIR" "$RUN_META_DIR"

runtime_root="$(lrz_make_runtime_root)"
cleanup() {
  rm -rf "$runtime_root"
}
trap cleanup EXIT

stage_local=0
case "${STAGE_TO_LOCAL,,}" in
  1|true|yes|on) stage_local=1 ;;
  0|false|no|off) stage_local=0 ;;
  auto)
    if [[ -n "$DATA_ENC_ARCHIVE" ]]; then
      stage_local=1
    fi
    ;;
  *) lrz_die "Invalid STAGE_TO_LOCAL value: $STAGE_TO_LOCAL" ;;
esac

if ! lrz_is_truthy "$KEEP_PLAINTEXT_OUT" && [[ -z "$OUT_ENC_ARCHIVE" ]]; then
  lrz_die "KEEP_PLAINTEXT_OUT=0 requires OUT_ENC_ARCHIVE."
fi

if (( stage_local )); then
  data_mount="$(lrz_stage_input_dir "$DATA_DIR" "$DATA_ENC_ARCHIVE" "$runtime_root" "train_input")"
  output_mount="${runtime_root%/}/train_output"
  mkdir -p "$output_mount"
else
  if [[ -n "$DATA_ENC_ARCHIVE" ]]; then
    lrz_die "DATA_ENC_ARCHIVE requires STAGE_TO_LOCAL=auto/true."
  fi
  [[ -d "$DATA_DIR" ]] || lrz_die "Training data directory not found: $DATA_DIR"
  data_mount="$DATA_DIR"
  output_mount="$OUT_DIR"
fi

repo_mount="${REPO_DIR}:/workspace/ehr-hier"
if lrz_is_truthy "$REPO_READONLY"; then
  repo_mount="${repo_mount}:ro"
fi
mounts="${data_mount}:/workspace/data,${output_mount}:/workspace/outputs,${repo_mount}"

if [[ -f "$IMAGE" ]]; then
  lrz_sha256_file "$IMAGE" >"${RUN_META_DIR%/}/image_${SLURM_JOB_ID:-manual}.sha256" || true
fi
printf '%s\n' "$IMAGE" >"${RUN_META_DIR%/}/image_${SLURM_JOB_ID:-manual}.txt"

srun --ntasks=1 \
  --container-image="$IMAGE" \
  --container-mounts="$mounts" \
  bash -lc "set -euo pipefail; cd /workspace/ehr-hier; ${LAUNCH_CMD}"

if (( stage_local )); then
  if lrz_is_truthy "$KEEP_PLAINTEXT_OUT"; then
    lrz_log "Syncing training outputs to $OUT_DIR"
    lrz_sync_dir "$output_mount" "$OUT_DIR"
  fi
  if [[ -n "$OUT_ENC_ARCHIVE" ]]; then
    lrz_log "Encrypting training outputs to $OUT_ENC_ARCHIVE"
    lrz_encrypt_dir_to_archive "$output_mount" "$OUT_ENC_ARCHIVE"
  fi
else
  if [[ -n "$OUT_ENC_ARCHIVE" ]]; then
    lrz_log "Encrypting training outputs from $OUT_DIR to $OUT_ENC_ARCHIVE"
    lrz_encrypt_dir_to_archive "$OUT_DIR" "$OUT_ENC_ARCHIVE"
    if ! lrz_is_truthy "$KEEP_PLAINTEXT_OUT"; then
      find "$OUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
  fi
fi
