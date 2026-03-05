#!/bin/bash
#SBATCH --job-name=z_vllm_mistral_local_drop_m01_m14_all_embeds
#SBATCH --partition=workq
#SBATCH --account=brics.a5u
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-41

set -euo pipefail

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
BASE=${PROJECT}/aisi_economy_index/store/AISI_demo/stage_3/dev
IMG=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif

SCRIPT=${PROJECT}/nbs/AISI_demo/Stage_3_negative_selection/dev/sbatch/run_mistral_drops_vllm_local.py

mkdir -p "${PROJECT}/logs"

export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_MODULES_CACHE=${PROJECT}/.hf_modules
export XDG_CACHE_HOME=${PROJECT}/.cache
mkdir -p "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"

# model
export MODEL_ID="/projects/a5u/adu_dev/aisi-economy-index/models/mistral_awq"
export QUANTIZATION="awq_marlin"      # if it crashes at init, set to "" for a boot test

# safer first-run knobs (bump later)
export MAX_MODEL_LEN=4096
export GPU_MEM_UTIL=0.82
export CHUNK_SIZE=16
export GEN_MAX_TOKENS=192
export RETRY_MAX_TOKENS=128
export TEMPERATURE=0.2
export TOP_P=0.95
export MIN_KEEP=2
export MAX_KEEP=3
export TP_SIZE=1

export START=0
export STOP=1000000000
export TEST_ONLY=1
export TEST_N=1000

# map array -> embed, month
EMBEDS=(bge_large e5_large gte_large)

TASK_ID=${SLURM_ARRAY_TASK_ID}
EMBED_IDX=$(( TASK_ID / 14 ))
MONTH_NUM=$(( TASK_ID % 14 + 1 ))

EMBED=${EMBEDS[$EMBED_IDX]}
MONTH=$(printf "adzuna_month%02d" "${MONTH_NUM}")

NPZ_DIR=${BASE}/llm_negative_selection/${EMBED}
NPZ_PATH=${NPZ_DIR}/${MONTH}.npz

export EMBED="${EMBED}"
export NPZ_PATH="${NPZ_PATH}"
export MONTH_TAG="${MONTH}"

# sanity
test -f "$SCRIPT"
test -f "$IMG"
test -f "$NPZ_PATH"
test -f "${MODEL_ID}/config.json"
ls -1 "${MODEL_ID}"/*.safetensors >/dev/null
test -f "${MODEL_ID}/quant_config.json"

echo "[TASK]  $TASK_ID -> embed=$EMBED month=$MONTH"
echo "[IMG]   $IMG"
echo "[PY]    $SCRIPT"
echo "[NPZ]   $NPZ_PATH"
echo "[MODEL] $MODEL_ID"
echo "[QUANT] ${QUANTIZATION:-none}"

srun --cpu-bind=none apptainer exec --nv \
  --bind /projects:/projects \
  --bind /lus:/lus \
  "$IMG" \
  python -u "$SCRIPT"
