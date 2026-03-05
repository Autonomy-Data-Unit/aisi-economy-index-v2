#!/bin/bash
#SBATCH --job-name=z_vllm_drop_m01_m14_bge
#SBATCH --partition=workq
#SBATCH --account=brics.a5u
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=1-14

set -euo pipefail

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
BASE=${PROJECT}/aisi_economy_index/store/AISI_demo/stage_3/dev

IMG=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif
SCRIPT=${PROJECT}/nbs/AISI_demo/Stage_3_negative_selection/dev/sbatch/run_deepseek_drops_vllm_test.py

NPZ_DIR=${BASE}/llm_negative_selection/bge_large
MONTH=$(printf "adzuna_month%02d" "${SLURM_ARRAY_TASK_ID}")
NPZ_PATH=${NPZ_DIR}/${MONTH}.npz

mkdir -p "${PROJECT}/logs"

# -----------------------------
# Offline cache + telemetry off
# -----------------------------
export EMBED=bge_large
export HF_HOME=/projects/public/brics/cache
export HF_HUB_CACHE=/projects/public/brics/cache
export TRANSFORMERS_CACHE=/projects/public/brics/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

# Writable caches
export HF_MODULES_CACHE=${PROJECT}/.hf_modules
mkdir -p "$HF_MODULES_CACHE"
export XDG_CACHE_HOME=${PROJECT}/.cache
mkdir -p "$XDG_CACHE_HOME"

# -----------------------------
# Model + input
# -----------------------------
export MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

export NPZ_PATH="${NPZ_PATH}"
export MONTH_TAG="${MONTH}"

# -----------------------------
# Range + knobs
# -----------------------------
export START=0
export STOP=1000000000   # irrelevant when TEST_ONLY=1
export TEST_ONLY=1
export TEST_N=100

export CHUNK_SIZE=32
export GEN_MAX_TOKENS=256
export RETRY_MAX_TOKENS=160
export TEMPERATURE=0.3
export TOP_P=0.95

export MIN_KEEP=2
export MAX_KEEP=3

export TP_SIZE=1
unset QUANTIZATION

# -----------------------------
# Sanity checks
# -----------------------------
test -f "$SCRIPT"
test -f "$NPZ_PATH"
test -f "$IMG"

echo "[MONTH]  $MONTH"
echo "[IMG]    $IMG"
echo "[SCRIPT] $SCRIPT"
echo "[NPZ]    $NPZ_PATH"
echo "[MODEL]  $MODEL_ID"
echo "[HF_MODULES_CACHE] $HF_MODULES_CACHE"
echo "[XDG_CACHE_HOME]   $XDG_CACHE_HOME"

srun --cpu-bind=none apptainer exec --nv \
  --bind /projects:/projects \
  --bind /lus:/lus \
  "$IMG" \
  python "$SCRIPT"
