#!/bin/bash
#SBATCH --job-name=z_vllm_drop_m14_bge_prod
#SBATCH --partition=workq
#SBATCH --account=brics.a5u
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/nbs/AISI_demo/Stage_3_negative_selection/prod/sbatch/logs/%x_%A_%a.out
#SBATCH --error=//projects/a5u/adu_dev/aisi-economy-index/nbs/AISI_demo/Stage_3_negative_selection/prod/sbatch/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00   # Month 14 is ~8.4k rows; 2hrs is plenty for 20 split jobs
#SBATCH --array=0-19

set -euo pipefail

# --- PATHS ---
PROJECT=/projects/a5u/adu_dev/aisi-economy-index
BASE=${PROJECT}/aisi_economy_index/store/AISI_demo/stage_3/prod 
METADATA="${BASE}/llm_negative_selection/metadata.json"
SCRIPT=${PROJECT}/nbs/AISI_demo/Stage_3_negative_selection/prod/sbatch/run_deepseek_drops_vllm_test.py
IMG=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif

# --- SELECTION ---
EMBED="bge_large"
MONTH="adzuna_month14"
NPZ_PATH=${BASE}/llm_negative_selection/${EMBED}/${MONTH}.npz

# --- DYNAMIC RANGE CALCULATION ---
# Reads the JSON dictionary directly using Python (standard on the login/compute node)
TOTAL_ROWS=$(python3 -c "import json; print(json.load(open('$METADATA'))['$MONTH'])")
NUM_TASKS=20
ROWS_PER_TASK=$(( (TOTAL_ROWS + NUM_TASKS - 1) / NUM_TASKS ))

export START=$(( SLURM_ARRAY_TASK_ID * ROWS_PER_TASK ))
export STOP=$(( START + ROWS_PER_TASK ))

# Safety cap for the final task
if [ "$STOP" -gt "$TOTAL_ROWS" ]; then STOP=$TOTAL_ROWS; fi

# --- EXPORTS ---
export TEST_ONLY=0
export EMBED="${EMBED}"
export NPZ_PATH="${NPZ_PATH}"
export MONTH_TAG="${MONTH}"

# Model & Inference Knobs
export MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
export CHUNK_SIZE=32
export GEN_MAX_TOKENS=256
export RETRY_MAX_TOKENS=160
export TEMPERATURE=0.3
export TOP_P=0.95
export MIN_KEEP=2
export MAX_KEEP=3
export TP_SIZE=1

# --- ENVIRONMENT & CACHE ---
export HF_HOME=/projects/public/brics/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_MODULES_CACHE=${PROJECT}/.hf_modules
export XDG_CACHE_HOME=${PROJECT}/.cache

mkdir -p "${PROJECT}/logs" "$HF_MODULES_CACHE" "$XDG_CACHE_HOME"

# --- SANITY CHECK ---
if [ ! -f "$NPZ_PATH" ]; then echo "FATAL: $NPZ_PATH not found"; exit 1; fi

echo "[TASK $SLURM_ARRAY_TASK_ID] Processing rows $START to $STOP (Total Month Rows: $TOTAL_ROWS)"

# --- EXECUTION ---
srun --cpu-bind=none apptainer exec --nv \
  --bind /projects:/projects \
  --bind /lus:/lus \
  "$IMG" \
  python "$SCRIPT"