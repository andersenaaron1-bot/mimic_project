#!/usr/bin/env bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:4
#SBATCH -t 1-23:00:00
#SBATCH -o log_%j.out
#SBATCH -e log_%j.err
#SBATCH --container-image="docker://nvcr.io/nvidia/pytorch:24.10-py3"
#SBATCH --container-mounts=/dss/dssfs04/PROJECT:/workspace/data,/dss/dsshome1/<user>/ehr-hier:/workspace/ehr-hier
srun bash -lc '
cd /workspace/ehr-hier && \
torchrun --nproc_per_node=4 train.py \
  data.root=/workspace/data/datasets \
  out.dir=/workspace/data/outputs \
  trainer.max_steps=200
'
