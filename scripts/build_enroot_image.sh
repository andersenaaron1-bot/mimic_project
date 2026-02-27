#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Build a local container image and convert it to Enroot/Pyxis .sqsh format.

Usage:
  scripts/build_enroot_image.sh --target train --output /dss/<proj>/containers/ehr-train.sqsh
  scripts/build_enroot_image.sh --target etl   --output /dss/<proj>/containers/meds-etl.sqsh
  scripts/build_enroot_image.sh --target pipeline-cpu --output /dss/<proj>/containers/ehr-pipeline-cpu.sqsh
  scripts/build_enroot_image.sh --target train-overlay --output /dss/<proj>/containers/ehr-train-overlay.sqsh
  scripts/build_enroot_image.sh --source-image docker://nvcr.io/nvidia/pytorch:24.10-py3 --output /dss/<proj>/containers/pytorch.sqsh

Options:
  --target <train|etl|pipeline-cpu|train-overlay>
                             Use repo Dockerfile presets.
  --dockerfile <path>        Override Dockerfile path.
  --context <path>           Build context path (default: repo root).
  --tag <name:tag>           Local image tag used during build/import.
  --output <path.sqsh>       Required output .sqsh path.
  --source-image <uri>       Skip local build and import this image directly.
  --help                     Show this help.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGET=""
DOCKERFILE=""
CONTEXT_DIR="$REPO_ROOT"
IMAGE_TAG=""
OUTPUT=""
SOURCE_IMAGE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="${2:-}"; shift 2 ;;
    --dockerfile) DOCKERFILE="${2:-}"; shift 2 ;;
    --context) CONTEXT_DIR="${2:-}"; shift 2 ;;
    --tag) IMAGE_TAG="${2:-}"; shift 2 ;;
    --output) OUTPUT="${2:-}"; shift 2 ;;
    --source-image) SOURCE_IMAGE="${2:-}"; shift 2 ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "$TARGET" ]]; then
  case "$TARGET" in
    train)
      DOCKERFILE="${DOCKERFILE:-$REPO_ROOT/containers/Dockerfile}"
      IMAGE_TAG="${IMAGE_TAG:-ehr-train:24.10-py3}"
      ;;
    etl)
      DOCKERFILE="${DOCKERFILE:-$REPO_ROOT/containers/Dockerfile.etl}"
      IMAGE_TAG="${IMAGE_TAG:-ehr-meds-etl:latest}"
      ;;
    pipeline-cpu)
      DOCKERFILE="${DOCKERFILE:-$REPO_ROOT/containers/Dockerfile.pipeline_cpu}"
      IMAGE_TAG="${IMAGE_TAG:-ehr-pipeline-cpu:py311}"
      ;;
    train-overlay)
      DOCKERFILE="${DOCKERFILE:-$REPO_ROOT/containers/Dockerfile.train_overlay}"
      IMAGE_TAG="${IMAGE_TAG:-ehr-train-overlay:24.10-py3}"
      ;;
    *)
      echo "Invalid --target value: $TARGET" >&2
      exit 1
      ;;
  esac
fi

if [[ -z "$OUTPUT" ]]; then
  echo "Missing required option: --output" >&2
  usage
  exit 1
fi

if [[ -z "$SOURCE_IMAGE" && -z "$DOCKERFILE" ]]; then
  echo "Set --source-image or provide --target/--dockerfile." >&2
  exit 1
fi

if ! command -v enroot >/dev/null 2>&1; then
  echo "enroot is required but not available on PATH." >&2
  exit 1
fi

mkdir -p "$(dirname "$OUTPUT")"

if [[ -n "$SOURCE_IMAGE" ]]; then
  echo "Importing remote image: $SOURCE_IMAGE"
  enroot import -o "$OUTPUT" "$SOURCE_IMAGE"
else
  [[ -f "$DOCKERFILE" ]] || { echo "Dockerfile not found: $DOCKERFILE" >&2; exit 1; }

  if command -v docker >/dev/null 2>&1; then
    ENGINE="docker"
    IMPORT_URI="dockerd://${IMAGE_TAG}"
  elif command -v podman >/dev/null 2>&1; then
    ENGINE="podman"
    IMPORT_URI="podman://${IMAGE_TAG}"
  else
    echo "No local builder found. Install docker/podman or use --source-image." >&2
    exit 1
  fi

  echo "Building image with $ENGINE: $IMAGE_TAG"
  "$ENGINE" build -f "$DOCKERFILE" -t "$IMAGE_TAG" "$CONTEXT_DIR"

  echo "Converting to Enroot image: $OUTPUT"
  enroot import -o "$OUTPUT" "$IMPORT_URI"
fi

if command -v sha256sum >/dev/null 2>&1; then
  sha256sum "$OUTPUT" >"${OUTPUT}.sha256"
elif command -v shasum >/dev/null 2>&1; then
  shasum -a 256 "$OUTPUT" >"${OUTPUT}.sha256"
fi

echo "Done: $OUTPUT"
