#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "$SCRIPT_DIR/lib/lrz_secure_runtime.sh"
lrz_setup_job_hardening

PART="${1:-lrz-hgx-h100-94x4}"
IMAGE_LOCAL="${IMAGE_LOCAL:-/dss/dssfs04/PROJECT/containers/ehr-train.sqsh}"
IMAGE_REMOTE="${IMAGE_REMOTE:-docker://nvcr.io/nvidia/pytorch:24.10-py3}"
IMG="${2:-$(lrz_pick_container_image "$IMAGE_LOCAL" "$IMAGE_REMOTE")}"

REPO="${REPO:-/dss/dsshome1/$USER/ehr-hier}"
DATA="${DATA:-/dss/dssfs04/PROJECT}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
TIME_LIMIT="${TIME_LIMIT:-04:00:00}"
REPO_READONLY="${REPO_READONLY:-0}"

[[ -d "$REPO" ]] || lrz_die "Repository path not found: $REPO"
[[ -d "$DATA" ]] || lrz_die "Data path not found: $DATA"

repo_mount="${REPO}:/workspace/ehr-hier"
if lrz_is_truthy "$REPO_READONLY"; then
  repo_mount="${repo_mount}:ro"
fi
mounts="${DATA}:/workspace/data,${repo_mount}"

srun --pty \
  -p "$PART" \
  --gres="gpu:${GPUS}" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --time="$TIME_LIMIT" \
  --container-image="$IMG" \
  --container-mounts="$mounts" \
  bash -lc "cd /workspace/ehr-hier; exec bash"
