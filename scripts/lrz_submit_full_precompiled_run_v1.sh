#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/lib/lrz_secure_runtime.sh
source "${SCRIPT_DIR%/}/lib/lrz_secure_runtime.sh"

CRITICAL_PATHS_SH="${CRITICAL_PATHS_SH:-$HOME/mimic_project/analysis/lrz_fs_inventory/latest/critical_paths.sh}"
[[ -f "$CRITICAL_PATHS_SH" ]] || lrz_die "critical_paths.sh not found: $CRITICAL_PATHS_SH"
# shellcheck source=/dev/null
source "$CRITICAL_PATHS_SH"

lrz_require_cmd sbatch
lrz_require_cmd squeue

REPO_CT="${REPO_CT:-/workspace/ehr-hier}"
IMAGE_CPU="${IMAGE_CPU:-docker://ghcr.io#andersenaaron1-bot/ehr-pipeline-cpu:0.3.1-pipeline-cpu-min}"
IMAGE_LOCAL="${IMAGE_LOCAL:-$DSS_HOST/containers/pytorch-2.5.1-cuda12.4.sqsh}"
IMAGE_REMOTE="${IMAGE_REMOTE:-docker://pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
IMAGE_GPU="${IMAGE_GPU:-$(lrz_pick_container_image "$IMAGE_LOCAL" "$IMAGE_REMOTE")}"

PRECOMP_BASENAME="${PRECOMP_BASENAME:-precompiled_transformer_v1_full_orfix_v1}"
RUN_BASENAME="${RUN_BASENAME:-transformer_run_v1_full_semantic_boost_v2_orfix_v1}"

PRECOMP_HOST="${PRECOMP_HOST:-$DSS_HOST/etl/$PRECOMP_BASENAME}"
PRECOMP_CT="${PRECOMP_CT:-/dss/etl/$PRECOMP_BASENAME}"
RUN_HOST="${RUN_HOST:-$DSS_HOST/etl/$RUN_BASENAME}"
RUN_CT="${RUN_CT:-/dss/etl/$RUN_BASENAME}"

PRECOMP_HOST_BASENAME="$(basename "$PRECOMP_HOST")"
RUN_HOST_BASENAME="$(basename "$RUN_HOST")"
[[ "$(basename "$PRECOMP_CT")" == "$PRECOMP_HOST_BASENAME" ]] || lrz_die "PRECOMP_CT basename ($(basename "$PRECOMP_CT")) does not match PRECOMP_HOST basename ($PRECOMP_HOST_BASENAME). Unset stale PRECOMP_CT or set both roots consistently."
[[ "$(basename "$RUN_CT")" == "$RUN_HOST_BASENAME" ]] || lrz_die "RUN_CT basename ($(basename "$RUN_CT")) does not match RUN_HOST basename ($RUN_HOST_BASENAME). Unset stale RUN_CT or set both roots consistently."

TUNING_MAX_SUBJECTS="${TUNING_MAX_SUBJECTS:-1024}"
TUNING_SAMPLE_SEED="${TUNING_SAMPLE_SEED:-1337}"
TRAJECTORY_MODE="${TRAJECTORY_MODE:-full_subject}"
POST_DISCHARGE_CUTOFF_DAYS="${POST_DISCHARGE_CUTOFF_DAYS:-31.0}"

TRAIN_PRECOMP_CPUS="${TRAIN_PRECOMP_CPUS:-32}"
TRAIN_PRECOMP_MEM="${TRAIN_PRECOMP_MEM:-128G}"
TRAIN_PRECOMP_TIME="${TRAIN_PRECOMP_TIME:-12:00:00}"
TRAIN_PRECOMP_SHARDS="${TRAIN_PRECOMP_SHARDS:-256}"
TRAIN_PRECOMP_CHUNKSIZE="${TRAIN_PRECOMP_CHUNKSIZE:-1}"
TRAIN_PRECOMP_PROGRESS_EVERY="${TRAIN_PRECOMP_PROGRESS_EVERY:-500}"

TUNING_PRECOMP_CPUS="${TUNING_PRECOMP_CPUS:-16}"
TUNING_PRECOMP_MEM="${TUNING_PRECOMP_MEM:-64G}"
TUNING_PRECOMP_TIME="${TUNING_PRECOMP_TIME:-03:00:00}"
TUNING_PRECOMP_SHARDS="${TUNING_PRECOMP_SHARDS:-64}"
TUNING_PRECOMP_CHUNKSIZE="${TUNING_PRECOMP_CHUNKSIZE:-1}"
TUNING_PRECOMP_PROGRESS_EVERY="${TUNING_PRECOMP_PROGRESS_EVERY:-100}"

TRAIN_PARTITION="${TRAIN_PARTITION:-lrz-hgx-h100-94x4}"
TRAIN_GRES="${TRAIN_GRES:-gpu:1}"
TRAIN_CPUS="${TRAIN_CPUS:-8}"
TRAIN_MEM="${TRAIN_MEM:-48G}"
TRAIN_TIME="${TRAIN_TIME:-40:00:00}"
TRAIN_DATALOADER_WORKERS="${TRAIN_DATALOADER_WORKERS:-6}"
TRAIN_PREFETCH_FACTOR="${TRAIN_PREFETCH_FACTOR:-4}"

BATCH_SIZE="${BATCH_SIZE:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-8}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
EPOCHS="${EPOCHS:-30}"
MAX_STEPS="${MAX_STEPS:-100000}"
SAVE_EVERY_STEPS="${SAVE_EVERY_STEPS:-2500}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-2500}"
TOKEN_FAMILY_WEIGHT_PRESET="${TOKEN_FAMILY_WEIGHT_PRESET:-semantic_boost_v2}"

DMODEL="${DMODEL:-256}"
NUM_HEADS="${NUM_HEADS:-4}"
DFF="${DFF:-512}"
NUM_LOCAL_LAYERS="${NUM_LOCAL_LAYERS:-2}"
NUM_GLOBAL_LAYERS="${NUM_GLOBAL_LAYERS:-2}"
NUM_CHUNK_LAYERS="${NUM_CHUNK_LAYERS:-1}"
MAX_WINDOWS="${MAX_WINDOWS:-24}"
MAX_CHUNKS_PER_WINDOW="${MAX_CHUNKS_PER_WINDOW:-4}"
MAX_LEN_PER_WINDOW="${MAX_LEN_PER_WINDOW:-96}"

mkdir -p "$PRECOMP_HOST/logs" "$RUN_HOST/logs" "$RUN_HOST/submit_meta"

TRAIN_PRECOMP_JOB="$RUN_HOST/submit_meta/precomp_train_full.sbatch"
TUNING_PRECOMP_JOB="$RUN_HOST/submit_meta/precomp_tuning_1024.sbatch"
TRAIN_JOB="$RUN_HOST/submit_meta/train_full_semboost_v2.sbatch"

lrz_log "Preparing LRZ submission chain"
lrz_log "PRECOMP_HOST=$PRECOMP_HOST"
lrz_log "PRECOMP_CT=$PRECOMP_CT"
lrz_log "RUN_HOST=$RUN_HOST"
lrz_log "RUN_CT=$RUN_CT"
lrz_log "TRAIN_PARTITION=$TRAIN_PARTITION"
lrz_log "IMAGE_CPU=$IMAGE_CPU"
lrz_log "IMAGE_GPU=$IMAGE_GPU"

cat >"$TRAIN_PRECOMP_JOB" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=precomp_train_full
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=$TRAIN_PRECOMP_CPUS
#SBATCH --mem=$TRAIN_PRECOMP_MEM
#SBATCH --time=$TRAIN_PRECOMP_TIME
#SBATCH --output=$PRECOMP_HOST/logs/precomp_train_%j.out

set -euo pipefail
source "$CRITICAL_PATHS_SH"
srun --ntasks=1 \\
  --container-image="$IMAGE_CPU" \\
  --container-mounts="$CPU_MOUNTS" \\
  bash -lc "set -euo pipefail; mkdir -p '$PRECOMP_CT'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/compile_timelines_v1.py --meds_reader_db '$DB' --splits_parquet '$SPLITS' --split train --output_dir '$PRECOMP_CT/train_full' --num_workers '$TRAIN_PRECOMP_CPUS' --num_output_shards '$TRAIN_PRECOMP_SHARDS' --chunksize '$TRAIN_PRECOMP_CHUNKSIZE' --skip_existing --progress_every '$TRAIN_PRECOMP_PROGRESS_EVERY' --trajectory_mode '$TRAJECTORY_MODE' --post_discharge_cutoff_days '$POST_DISCHARGE_CUTOFF_DAYS' --medtok_vocab_dir '$MEDTOK_VOC_FINAL' --medtok_crosswalk_json '$CROSSWALK_JSON' --codes_parquet_parent_lookup '$CODES_PARQUET' --sparse_vocab_json '$SPARSE_VOC' --code2id_pt '$ART/code2id.pt' --stats_pt '$ART/stats.pt' --cvae_ckpt '$ART/cvae_ckpt.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' 2>&1 | tee '$PRECOMP_CT/compile_train_full.log'"
EOF

cat >"$TUNING_PRECOMP_JOB" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=precomp_tuning_1024
#SBATCH --partition=lrz-cpu
#SBATCH --qos=cpu
#SBATCH --cpus-per-task=$TUNING_PRECOMP_CPUS
#SBATCH --mem=$TUNING_PRECOMP_MEM
#SBATCH --time=$TUNING_PRECOMP_TIME
#SBATCH --output=$PRECOMP_HOST/logs/precomp_tuning_%j.out

set -euo pipefail
source "$CRITICAL_PATHS_SH"
srun --ntasks=1 \\
  --container-image="$IMAGE_CPU" \\
  --container-mounts="$CPU_MOUNTS" \\
  bash -lc "set -euo pipefail; mkdir -p '$PRECOMP_CT'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/compile_timelines_v1.py --meds_reader_db '$DB' --splits_parquet '$SPLITS' --split tuning --max_subjects '$TUNING_MAX_SUBJECTS' --sample_seed '$TUNING_SAMPLE_SEED' --output_dir '$PRECOMP_CT/tuning_1024' --num_workers '$TUNING_PRECOMP_CPUS' --num_output_shards '$TUNING_PRECOMP_SHARDS' --chunksize '$TUNING_PRECOMP_CHUNKSIZE' --skip_existing --progress_every '$TUNING_PRECOMP_PROGRESS_EVERY' --trajectory_mode '$TRAJECTORY_MODE' --post_discharge_cutoff_days '$POST_DISCHARGE_CUTOFF_DAYS' --medtok_vocab_dir '$MEDTOK_VOC_FINAL' --medtok_crosswalk_json '$CROSSWALK_JSON' --codes_parquet_parent_lookup '$CODES_PARQUET' --sparse_vocab_json '$SPARSE_VOC' --code2id_pt '$ART/code2id.pt' --stats_pt '$ART/stats.pt' --cvae_ckpt '$ART/cvae_ckpt.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' 2>&1 | tee '$PRECOMP_CT/compile_tuning_1024.log'"
EOF

cat >"$TRAIN_JOB" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=train_full_semboost_v2
#SBATCH --partition=$TRAIN_PARTITION
#SBATCH --gres=$TRAIN_GRES
#SBATCH --cpus-per-task=$TRAIN_CPUS
#SBATCH --mem=$TRAIN_MEM
#SBATCH --time=$TRAIN_TIME
#SBATCH --output=$RUN_HOST/logs/train_%j.out

set -euo pipefail
source "$CRITICAL_PATHS_SH"
srun --ntasks=1 \\
  --container-image="$IMAGE_GPU" \\
  --container-mounts="$CPU_MOUNTS" \\
  bash -lc "set -euo pipefail; mkdir -p '$RUN_CT'; export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'; export PYTHONPATH='$REPO_CT:/deps'\${PYTHONPATH:+':'\$PYTHONPATH}; cd '$REPO_CT'; python scripts/train_transformer_v1.py --precompiled_train_root '$PRECOMP_CT/train_full' --precompiled_eval_root '$PRECOMP_CT/tuning_1024' --splits_parquet '$SPLITS' --train_split train --eval_split tuning --trajectory_mode '$TRAJECTORY_MODE' --post_discharge_cutoff_days '$POST_DISCHARGE_CUTOFF_DAYS' --medtok_vocab_dir '$MEDTOK_VOC_FINAL' --medtok_crosswalk_json '$CROSSWALK_JSON' --codes_parquet_parent_lookup '$CODES_PARQUET' --sparse_vocab_json '$SPARSE_VOC' --runtime_vocab_json '$RUNTIME_VOC' --code2id_pt '$ART/code2id.pt' --stats_pt '$ART/stats.pt' --cvae_ckpt '$ART/cvae_ckpt.pt' --tokenizer_ckpt '$ART/value_tokenizer.pt' --output_dir '$RUN_CT' --batch_size '$BATCH_SIZE' --eval_batch_size '$EVAL_BATCH_SIZE' --num_workers '$TRAIN_DATALOADER_WORKERS' --prefetch_factor '$TRAIN_PREFETCH_FACTOR' --grad_accum_steps '$GRAD_ACCUM_STEPS' --epochs '$EPOCHS' --max_steps '$MAX_STEPS' --save_every_steps '$SAVE_EVERY_STEPS' --eval_every_steps '$EVAL_EVERY_STEPS' --token_family_weight_preset '$TOKEN_FAMILY_WEIGHT_PRESET' --d_model '$DMODEL' --num_heads '$NUM_HEADS' --d_ff '$DFF' --num_local_layers '$NUM_LOCAL_LAYERS' --num_global_layers '$NUM_GLOBAL_LAYERS' --num_chunk_layers '$NUM_CHUNK_LAYERS' --max_windows '$MAX_WINDOWS' --max_chunks_per_window '$MAX_CHUNKS_PER_WINDOW' --max_len_per_window '$MAX_LEN_PER_WINDOW' --device cuda 2>&1 | tee '$RUN_CT/train.log'"
EOF

chmod 700 "$TRAIN_PRECOMP_JOB" "$TUNING_PRECOMP_JOB" "$TRAIN_JOB"

TRAIN_PRECOMP_JID="$(sbatch --parsable --export=ALL "$TRAIN_PRECOMP_JOB")"
TUNING_PRECOMP_JID="$(sbatch --parsable --export=ALL "$TUNING_PRECOMP_JOB")"
TRAIN_JID="$(sbatch --parsable --export=ALL --dependency=afterok:${TRAIN_PRECOMP_JID}:${TUNING_PRECOMP_JID} "$TRAIN_JOB")"

printf 'TRAIN_PRECOMP_JID=%s\n' "$TRAIN_PRECOMP_JID"
printf 'TUNING_PRECOMP_JID=%s\n' "$TUNING_PRECOMP_JID"
printf 'TRAIN_JID=%s\n' "$TRAIN_JID"
printf 'PRECOMP_HOST=%s\n' "$PRECOMP_HOST"
printf 'RUN_HOST=%s\n' "$RUN_HOST"

squeue -j "$TRAIN_PRECOMP_JID,$TUNING_PRECOMP_JID,$TRAIN_JID"
