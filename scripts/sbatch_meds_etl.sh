#!/usr/bin/env bash
#SBATCH -J meds_etl
#SBATCH -p lrz-cpu
#SBATCH -t 2-00:00:00                # walltime (D-HH:MM:SS)
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=0                      # or set a fixed value like 256G
#SBATCH -o slurm-%x-%j.out
#SBATCH -e slurm-%x-%j.err
# SBATCH -A <ACCOUNT>                # if your cluster requires an account

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "$SCRIPT_DIR/lib/lrz_secure_runtime.sh"
lrz_setup_job_hardening

IMAGE_LOCAL="${IMAGE_LOCAL:-/dss/<proj>/containers/meds-etl.sqsh}"
IMAGE_REMOTE="${IMAGE_REMOTE:-}"
IMAGE="${IMAGE:-$(lrz_pick_container_image "$IMAGE_LOCAL" "$IMAGE_REMOTE")}"

RAW_MIMIC_DIR="${RAW_MIMIC_DIR:-/dss/<proj>/mimic-iv-2.2}"
RAW_MIMIC_ENC_ARCHIVE="${RAW_MIMIC_ENC_ARCHIVE:-}"   # Optional .tar.zst.age
OUT_DIR="${OUT_DIR:-/dss/<proj>/meds_mimiciv_2.2}"
OUT_ENC_ARCHIVE="${OUT_ENC_ARCHIVE:-}"               # Optional .tar.zst.age

STAGE_TO_LOCAL="${STAGE_TO_LOCAL:-auto}"             # auto|true|false
KEEP_PLAINTEXT_OUT="${KEEP_PLAINTEXT_OUT:-1}"        # 1 keeps OUT_DIR updated

LOGDIR="${LOGDIR:-/dss/<proj>/logs}"
NUM_PROC="${NUM_PROC:-${SLURM_CPUS_PER_TASK:-32}}"
BACKEND="${BACKEND:-polars}"

if [[ -z "${NUM_SHARDS:-}" ]]; then
  if [[ "${BACKEND}" == "cpp" ]]; then
    NUM_SHARDS="$NUM_PROC"
  else
    # polars: prefer fewer shards unless memory pressure requires more.
    NUM_SHARDS=$(( NUM_PROC / 2 ))
    if (( NUM_SHARDS < 8 )); then
      NUM_SHARDS=8
    fi
  fi
fi

mkdir -p "$LOGDIR" "$OUT_DIR"

if [[ "$IMAGE_LOCAL" == *"<proj>"* || "$RAW_MIMIC_DIR" == *"<proj>"* || "$OUT_DIR" == *"<proj>"* || "$LOGDIR" == *"<proj>"* ]]; then
  lrz_die "Replace /dss/<proj>/... placeholders via env vars before submission."
fi

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
    if [[ -n "$RAW_MIMIC_ENC_ARCHIVE" ]]; then
      stage_local=1
    fi
    ;;
  *) lrz_die "Invalid STAGE_TO_LOCAL value: $STAGE_TO_LOCAL" ;;
esac

if ! lrz_is_truthy "$KEEP_PLAINTEXT_OUT" && [[ -z "$OUT_ENC_ARCHIVE" ]]; then
  lrz_die "KEEP_PLAINTEXT_OUT=0 requires OUT_ENC_ARCHIVE."
fi

input_mount="$(lrz_stage_input_dir "$RAW_MIMIC_DIR" "$RAW_MIMIC_ENC_ARCHIVE" "$runtime_root" "mimic_input")"
if (( stage_local )); then
  output_mount="${runtime_root%/}/meds_output"
  mkdir -p "$output_mount"
else
  output_mount="$OUT_DIR"
fi

mounts="${input_mount}:/data/mimic,${output_mount}:/data/meds"
export POLARS_MAX_THREADS="$NUM_PROC"

srun --container-image="$IMAGE" \
  --container-mounts="$mounts" \
  bash -lc "set -euo pipefail; meds_etl_mimic /data/mimic /data/meds --num_proc $NUM_PROC --num_shards $NUM_SHARDS --backend $BACKEND"

if (( stage_local )); then
  if lrz_is_truthy "$KEEP_PLAINTEXT_OUT"; then
    lrz_log "Syncing ETL output to $OUT_DIR"
    lrz_sync_dir "$output_mount" "$OUT_DIR"
  fi
  if [[ -n "$OUT_ENC_ARCHIVE" ]]; then
    lrz_log "Encrypting ETL output to $OUT_ENC_ARCHIVE"
    lrz_encrypt_dir_to_archive "$output_mount" "$OUT_ENC_ARCHIVE"
  fi
else
  if [[ -n "$OUT_ENC_ARCHIVE" ]]; then
    lrz_log "Encrypting ETL output from $OUT_DIR to $OUT_ENC_ARCHIVE"
    lrz_encrypt_dir_to_archive "$OUT_DIR" "$OUT_ENC_ARCHIVE"
    if ! lrz_is_truthy "$KEEP_PLAINTEXT_OUT"; then
      find "$OUT_DIR" -mindepth 1 -maxdepth 1 -exec rm -rf {} +
    fi
  fi
fi
