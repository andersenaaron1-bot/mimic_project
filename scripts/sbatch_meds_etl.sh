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

runtime_lib=""
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR%/}/scripts/lib/lrz_secure_runtime.sh" ]]; then
  runtime_lib="${SLURM_SUBMIT_DIR%/}/scripts/lib/lrz_secure_runtime.sh"
elif [[ -f "$SCRIPT_DIR/lib/lrz_secure_runtime.sh" ]]; then
  runtime_lib="$SCRIPT_DIR/lib/lrz_secure_runtime.sh"
fi
[[ -n "$runtime_lib" ]] || { echo "ERROR: cannot find scripts/lib/lrz_secure_runtime.sh" >&2; exit 1; }
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "$runtime_lib"
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
ETL_TIMEOUT_SECONDS="${ETL_TIMEOUT_SECONDS:-0}"

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

default_etl_cmd="meds_etl_mimic /data/mimic /data/meds --num_proc $NUM_PROC --num_shards $NUM_SHARDS --backend $BACKEND"
ETL_CMD="${ETL_CMD:-$default_etl_cmd}"

mkdir -p "$LOGDIR" "$OUT_DIR"

if [[ -z "$IMAGE_REMOTE" && "$IMAGE_LOCAL" == *"<proj>"* ]]; then
  lrz_die "Replace /dss/<proj>/... placeholders via env vars before submission."
fi
if [[ "$RAW_MIMIC_DIR" == *"<proj>"* || "$OUT_DIR" == *"<proj>"* || "$LOGDIR" == *"<proj>"* ]]; then
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

# Some LRZ nodes do not provide host-side `age`. If input is AGE-encrypted,
# decrypt inside the ETL container and stage plaintext in runtime_root.
need_container_age_decrypt=0
if [[ -n "$RAW_MIMIC_ENC_ARCHIVE" ]]; then
  case "${EHR_ENCRYPTION_BACKEND:-auto}" in
    age|AGE) need_container_age_decrypt=1 ;;
    auto|AUTO)
      if [[ "$RAW_MIMIC_ENC_ARCHIVE" == *.age ]]; then
        need_container_age_decrypt=1
      fi
      ;;
  esac
fi

if (( need_container_age_decrypt )) && ! lrz_have_cmd age; then
  [[ -n "${EHR_AGE_IDENTITY_FILE:-}" ]] || lrz_die "Set EHR_AGE_IDENTITY_FILE to decrypt AGE archives."
  [[ -f "${EHR_AGE_IDENTITY_FILE}" ]] || lrz_die "AGE identity file not found: ${EHR_AGE_IDENTITY_FILE}"
  [[ -f "${RAW_MIMIC_ENC_ARCHIVE}" ]] || lrz_die "Encrypted archive not found: ${RAW_MIMIC_ENC_ARCHIVE}"

  input_mount="${runtime_root%/}/mimic_input"
  mkdir -p "$input_mount"
  lrz_log "Host age not found. Decrypting ${RAW_MIMIC_ENC_ARCHIVE} via container into ${input_mount}"

  enc_dir="$(dirname "$RAW_MIMIC_ENC_ARCHIVE")"
  enc_base="$(basename "$RAW_MIMIC_ENC_ARCHIVE")"
  key_dir="$(dirname "$EHR_AGE_IDENTITY_FILE")"
  key_base="$(basename "$EHR_AGE_IDENTITY_FILE")"

  decrypt_mounts="${enc_dir}:/enc_in,${key_dir}:/keys,${input_mount}:/out"
  srun --container-image="$IMAGE" \
    --container-mounts="$decrypt_mounts" \
    bash -lc "set -euo pipefail; age --decrypt -i /keys/${key_base} /enc_in/${enc_base} | zstd -d -T0 -q | tar -C /out -xf -"
else
  input_mount="$(lrz_stage_input_dir "$RAW_MIMIC_DIR" "$RAW_MIMIC_ENC_ARCHIVE" "$runtime_root" "mimic_input")"
fi
if (( stage_local )); then
  output_mount="${runtime_root%/}/meds_output"
  mkdir -p "$output_mount"
else
  output_mount="$OUT_DIR"
fi

mounts="${input_mount}:/data/mimic,${output_mount}:/data/meds"
export POLARS_MAX_THREADS="$NUM_PROC"

lrz_log "Using image: $IMAGE"
lrz_log "Input mount: $input_mount -> /data/mimic"
lrz_log "Output mount: $output_mount -> /data/meds"
lrz_log "ETL command: $ETL_CMD"

srun --container-image="$IMAGE" \
  --container-mounts="$mounts" \
  env ROOT_OUTPUT_DIR=/data/meds RAW_INPUT_DIR=/data/mimic ETL_CMD="$ETL_CMD" ETL_TIMEOUT_SECONDS="$ETL_TIMEOUT_SECONDS" \
  bash -lc 'set -euo pipefail; if [[ "${ETL_TIMEOUT_SECONDS:-0}" =~ ^[0-9]+$ ]] && (( ETL_TIMEOUT_SECONDS > 0 )); then timeout "${ETL_TIMEOUT_SECONDS}s" bash -lc "$ETL_CMD"; else bash -lc "$ETL_CMD"; fi'

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
