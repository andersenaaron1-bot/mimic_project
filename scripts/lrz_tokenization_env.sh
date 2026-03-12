#!/usr/bin/env bash
# shellcheck shell=bash

if [[ -n "${_LRZ_TOKENIZATION_ENV_SH:-}" ]]; then
  return 0
fi
_LRZ_TOKENIZATION_ENV_SH=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_SOURCE_DEFAULT="$(cd "${SCRIPT_DIR%/}/.." && pwd)"

# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "${SCRIPT_DIR%/}/lib/lrz_secure_runtime.sh"
lrz_setup_job_hardening

export DSS_HOST="${DSS_HOST:-/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2}"
export DEPS_HOST="${DEPS_HOST:-$DSS_HOST/containers/runtime_pydeps}"
export DSS_ARTIFACTS_HOST="${DSS_ARTIFACTS_HOST:-/dss/artifacts}"
if [[ -d "$DSS_HOST/mimic_project" ]]; then
  REPO_DEFAULT="$DSS_HOST/mimic_project"
else
  REPO_DEFAULT="$REPO_SOURCE_DEFAULT"
fi
export REPO="${REPO:-$REPO_DEFAULT}"
export IMAGE_CPU="${IMAGE_CPU:-docker://ghcr.io#andersenaaron1-bot/ehr-pipeline-cpu:0.3.1-pipeline-cpu-min}"

export LRZ_CPU_PARTITION="${LRZ_CPU_PARTITION:-lrz-cpu}"
export LRZ_CPU_QOS="${LRZ_CPU_QOS:-cpu}"
export LRZ_CPU_IMMEDIATE="${LRZ_CPU_IMMEDIATE:-180}"

_cpu_mounts_default="$REPO:/workspace/ehr-hier,$DSS_HOST:/dss,$DEPS_HOST:/deps"
if [[ -d "$DSS_ARTIFACTS_HOST" ]]; then
  _cpu_mounts_default="${_cpu_mounts_default},$DSS_ARTIFACTS_HOST:/dss-artifacts"
fi
export CPU_MOUNTS="${CPU_MOUNTS:-$_cpu_mounts_default}"

# Stable container-visible dataset/artifact paths on LRZ.
export ART="${ART:-/dss/etl/pipeline_artifacts_20260226_033711}"
export DB="${DB:-/dss/etl/meds_reader_db_mimiciv_20260226_033711/mimiciv.db}"
export SPLITS="${SPLITS:-/dss/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/subject_splits.parquet}"
export CODES_PARQUET="${CODES_PARQUET:-/dss/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/codes.parquet}"
export MEDTOK_VOC_BASE="${MEDTOK_VOC_BASE:-/dss/etl/pipeline_artifacts_20260226_033711/medtok_compressed_v1}"
export DSS_MEDTOK_CODE2EMBEDS_HOST="${DSS_MEDTOK_CODE2EMBEDS_HOST:-$DSS_HOST/artifacts/medtok/code2embeddings.json}"
if [[ -f "$DSS_MEDTOK_CODE2EMBEDS_HOST" ]]; then
  export MEDTOK_CODE2EMBEDS="${MEDTOK_CODE2EMBEDS:-/dss/artifacts/medtok/code2embeddings.json}"
else
  export MEDTOK_CODE2EMBEDS="${MEDTOK_CODE2EMBEDS:-/dss-artifacts/medtok/code2embeddings.json}"
fi

# Keep evaluation outputs under the DSS project tree for LRZ high-I/O jobs.
export LRZ_TOKENIZATION_EVAL_HOST="${LRZ_TOKENIZATION_EVAL_HOST:-$DSS_HOST/etl/tokenization_v1_eval}"
export LRZ_TOKENIZATION_EVAL_CT="${LRZ_TOKENIZATION_EVAL_CT:-/dss/etl/tokenization_v1_eval}"

export EVAL_DIR_HOST="$LRZ_TOKENIZATION_EVAL_HOST"
export EVAL_DIR_CT="$LRZ_TOKENIZATION_EVAL_CT"

export MEDTOK_VOC_FINAL_HOST="${MEDTOK_VOC_FINAL_HOST:-$EVAL_DIR_HOST/medtok_compressed_v1_exact}"
export MEDTOK_VOC_FINAL_CT="${MEDTOK_VOC_FINAL_CT:-$EVAL_DIR_CT/medtok_compressed_v1_exact}"

export CROSSWALK_JSON_HOST="${CROSSWALK_JSON_HOST:-$EVAL_DIR_HOST/medtok_crosswalk_v1.json}"
export CROSSWALK_JSON_CT="${CROSSWALK_JSON_CT:-$EVAL_DIR_CT/medtok_crosswalk_v1.json}"

export DECISION_CSV_HOST="${DECISION_CSV_HOST:-$EVAL_DIR_HOST/decision_table_train_20k.csv}"
export DECISION_CSV_CT="${DECISION_CSV_CT:-$EVAL_DIR_CT/decision_table_train_20k.csv}"

export SPARSE_VOC_HOST="${SPARSE_VOC_HOST:-$EVAL_DIR_HOST/token_vocab_sparse_v1.json}"
export SPARSE_VOC_CT="${SPARSE_VOC_CT:-$EVAL_DIR_CT/token_vocab_sparse_v1.json}"

export AUDIT_JSON_HOST="${AUDIT_JSON_HOST:-$EVAL_DIR_HOST/tokenization_audit_train_1000.json}"
export AUDIT_JSON_CT="${AUDIT_JSON_CT:-$EVAL_DIR_CT/tokenization_audit_train_1000.json}"

export RUNTIME_VOC_HOST="${RUNTIME_VOC_HOST:-$EVAL_DIR_HOST/runtime_vocab_compact_v1_20k.json}"
export RUNTIME_VOC_CT="${RUNTIME_VOC_CT:-$EVAL_DIR_CT/runtime_vocab_compact_v1_20k.json}"

lrz_tok_show_paths() {
  cat <<EOF
REPO=$REPO
IMAGE_CPU=$IMAGE_CPU
CPU_MOUNTS=$CPU_MOUNTS
ART=$ART
DB=$DB
SPLITS=$SPLITS
CODES_PARQUET=$CODES_PARQUET
MEDTOK_VOC_BASE=$MEDTOK_VOC_BASE
MEDTOK_CODE2EMBEDS=$MEDTOK_CODE2EMBEDS
EVAL_DIR_HOST=$EVAL_DIR_HOST
EVAL_DIR_CT=$EVAL_DIR_CT
MEDTOK_VOC_FINAL_HOST=$MEDTOK_VOC_FINAL_HOST
MEDTOK_VOC_FINAL_CT=$MEDTOK_VOC_FINAL_CT
SPARSE_VOC_HOST=$SPARSE_VOC_HOST
SPARSE_VOC_CT=$SPARSE_VOC_CT
AUDIT_JSON_HOST=$AUDIT_JSON_HOST
AUDIT_JSON_CT=$AUDIT_JSON_CT
RUNTIME_VOC_HOST=$RUNTIME_VOC_HOST
RUNTIME_VOC_CT=$RUNTIME_VOC_CT
EOF
}

lrz_tok_cpu() {
  local cpus="${1:-8}"
  local mem="${2:-32G}"
  local time_limit="${3:-02:00:00}"
  shift 3 || true
  local cmd="$*"
  [[ -n "$cmd" ]] || lrz_die "lrz_tok_cpu requires a command string"

  mkdir -p "$EVAL_DIR_HOST"

  srun --immediate="$LRZ_CPU_IMMEDIATE" \
    -p "$LRZ_CPU_PARTITION" \
    --qos="$LRZ_CPU_QOS" \
    --cpus-per-task="$cpus" \
    --mem="$mem" \
    --time="$time_limit" \
    --container-image="$IMAGE_CPU" \
    --container-mounts="$CPU_MOUNTS" \
    bash -lc "set -euo pipefail; mkdir -p \"$EVAL_DIR_CT\"; export PYTHONPATH=/workspace/ehr-hier:/deps\${PYTHONPATH:+:\$PYTHONPATH}; cd /workspace/ehr-hier; $cmd"
}
