#!/usr/bin/env bash
PART=${1:-lrz-hgx-h100-94x4}
IMG=${2:-docker://nvcr.io/nvidia/pytorch:24.10-py3}
REPO=${REPO:-/dss/dsshome1/$USER/ehr-hier}
DATA=${DATA:-/dss/dssfs04/PROJECT}
salloc -p $PART --gres=gpu:1
srun --pty --container-image="$IMG" \
  --container-mounts=$DATA:/workspace/data,$REPO:/workspace/ehr-hier \
  bash