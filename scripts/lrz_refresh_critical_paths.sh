#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "${SCRIPT_DIR%/}/lib/lrz_secure_runtime.sh"

ROOT_HOME="${ROOT_HOME:-$HOME/mimic_project}"
ROOT_DSS="${ROOT_DSS:-/dss/dssfs04/lwp-dss-0002/pn76ko/pn76ko-dss-0000/proc_mining_dfg/go75meh2}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT="${OUT:-$ROOT_HOME/analysis/lrz_fs_inventory/$STAMP}"

mkdir -p "$OUT"

find "$ROOT_HOME" -printf '%y\t%s\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' | sort > "$OUT/home_mimic_project.find.tsv"
find "$ROOT_DSS" -printf '%y\t%s\t%TY-%Tm-%Td %TH:%TM:%TS\t%p\n' | sort > "$OUT/dss_go75meh2.find.tsv"
du -h --max-depth=4 "$ROOT_HOME" > "$OUT/home_mimic_project.du.txt"
du -h --max-depth=4 "$ROOT_DSS" > "$OUT/dss_go75meh2.du.txt"

LATEST_ART="$(find "$ROOT_DSS/etl" -maxdepth 1 -type d -name 'pipeline_artifacts_*' | sort | tail -n 1)"
LATEST_DB="$(find "$ROOT_DSS/etl" -path '*/meds_reader_db_mimiciv_*/mimiciv.db' | sort | tail -n 1)"
LATEST_SPLITS="$(find "$ROOT_DSS/etl" -path '*/subject_splits.parquet' | sort | tail -n 1)"
LATEST_CODES="$(find "$ROOT_DSS/etl" -path '*/metadata/codes.parquet' | sort | tail -n 1)"
LATEST_MEDTOK_BASE="$(find "$ROOT_DSS/etl" -type d -name 'medtok_compressed_v1' | sort | tail -n 1)"

if [[ -f "$ROOT_DSS/artifacts/medtok/code2embeddings.json" ]]; then
  MEDTOK_C2E_HOST="$ROOT_DSS/artifacts/medtok/code2embeddings.json"
elif [[ -f /dss/artifacts/medtok/code2embeddings.json ]]; then
  MEDTOK_C2E_HOST="/dss/artifacts/medtok/code2embeddings.json"
else
  MEDTOK_C2E_HOST=""
fi

cat > "$OUT/critical_paths.sh" <<EOF
export REPO="$ROOT_HOME"
export REPO_CT="/workspace/ehr-hier"

export DSS_HOST="$ROOT_DSS"
export DEPS_HOST="$ROOT_DSS/containers/runtime_pydeps"
export IMAGE_CPU='docker://ghcr.io#andersenaaron1-bot/ehr-pipeline-cpu:0.3.1-pipeline-cpu-min'
export CPU_MOUNTS="\$REPO:\$REPO_CT,\$DSS_HOST:/dss,\$DEPS_HOST:/deps"

export ART_HOST="${LATEST_ART:-}"
export DB_HOST="${LATEST_DB:-}"
export SPLITS_HOST="${LATEST_SPLITS:-}"
export CODES_PARQUET_HOST="${LATEST_CODES:-}"
export MEDTOK_VOC_BASE_HOST="${LATEST_MEDTOK_BASE:-}"
export MEDTOK_CODE2EMBEDS_HOST="${MEDTOK_C2E_HOST:-}"
export EVAL_DIR_HOST="\$DSS_HOST/etl/tokenization_v1_eval"

export ART="/dss/etl/$(basename "${LATEST_ART:-pipeline_artifacts_20260226_033711}")"
export DB="/dss/etl/$(basename "$(dirname "${LATEST_DB:-$ROOT_DSS/etl/meds_reader_db_mimiciv_20260226_033711/mimiciv.db}")")/mimiciv.db"
export SPLITS="/dss/etl/$(printf '%s' "${LATEST_SPLITS:-$ROOT_DSS/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/subject_splits.parquet}" | sed "s#^$ROOT_DSS/etl/##")"
export CODES_PARQUET="/dss/etl/$(printf '%s' "${LATEST_CODES:-$ROOT_DSS/etl/mimiciv_20260224_0114_fresh_img023/out_plain/MEDS_cohort/metadata/codes.parquet}" | sed "s#^$ROOT_DSS/etl/##")"
export MEDTOK_VOC_BASE="/dss/etl/$(printf '%s' "${LATEST_MEDTOK_BASE:-$ROOT_DSS/etl/pipeline_artifacts_20260226_033711/medtok_compressed_v1}" | sed "s#^$ROOT_DSS/etl/##")"
export MEDTOK_CODE2EMBEDS="/dss/artifacts/medtok/code2embeddings.json"
export EVAL_DIR="/dss/etl/tokenization_v1_eval"
export MEDTOK_VOC_FINAL="/dss/etl/tokenization_v1_eval/medtok_compressed_v1_exact"
export CROSSWALK_JSON="/dss/etl/tokenization_v1_eval/medtok_crosswalk_v1.json"
export DECISION_CSV="/dss/etl/tokenization_v1_eval/decision_table_train_20k.csv"
export SPARSE_VOC="/dss/etl/tokenization_v1_eval/token_vocab_sparse_v1.json"
export AUDIT_JSON="/dss/etl/tokenization_v1_eval/tokenization_audit_train_1000.json"
export RUNTIME_VOC="/dss/etl/tokenization_v1_eval/runtime_vocab_compact_v1_20k.json"
EOF

ln -sfn "$OUT" "$ROOT_HOME/analysis/lrz_fs_inventory/latest"

printf 'Snapshot: %s\n' "$OUT"
printf 'Env file:  %s\n' "$OUT/critical_paths.sh"
printf '\n'
sed -n '1,120p' "$OUT/critical_paths.sh"
