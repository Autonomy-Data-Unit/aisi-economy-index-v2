#!/bin/bash
#SBATCH --job-name=z_vllm_drop_m14_bge_prod
#SBATCH --partition=workq
#SBATCH --account=brics.a5u
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
# NOTE: Removed --array as we are targeting a single specific run

set -euo pipefail

# --- PATH UPDATES (Targeting PROD) ---
PROJECT=/projects/a5u/adu_dev/aisi-economy-index
# Changed /dev to /prod
BASE=${PROJECT}/aisi_economy_index/store/AISI_demo/stage_3/prod 
# Changed /dev to /prod in script path
SCRIPT=${PROJECT}/nbs/AISI_demo/Stage_3_negative_selection/prod/sbatch/run_deepseek_drops_vllm_test.py

IMG=/projects/public/brics/containers/e4s/e4s-cuda90-aarch64-25.11.sif
mkdir -p "${PROJECT}/logs"

# -----------------------------
# Offline cache + telemetry
# -----------------------------
export HF_HOME=/projects/public/brics/cache
export HF_HUB_CACHE=/projects/public/brics/cache
export TRANSFORMERS_CACHE=/projects/public/brics/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export HF_MODULES_CACHE=${PROJECT}/.hf_modules
mkdir -p "$HF_MODULES_CACHE"
export XDG_CACHE_HOME=${PROJECT}/.cache
mkdir -p "$XDG_CACHE_HOME"

# -----------------------------
# Model Configuration
# -----------------------------
export MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

# -----------------------------
# Specific Selection: BGE + Month 14
# -----------------------------
# Hardcoded to your specific requirements
EMBED="bge_large"
MONTH="adzuna_month14"

NPZ_DIR=${BASE}/llm_negative_selection/${EMBED}
NPZ_PATH=${NPZ_DIR}/${MONTH}.npz

export EMBED="${EMBED}"
export NPZ_PATH="${NPZ_PATH}"
export MONTH_TAG="${MONTH}"

# -----------------------------
# Range + Knobs
# -----------------------------

export TEST_ONLY=0        # Set to 0 to process the FULL dataset
export TEST_N=0           # Ignored when TEST_ONLY=0
export START=0
export STOP=1000000000    # High limit to ensure we catch all records

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
test -f "$SCRIPT" || { echo "Script not found at $SCRIPT"; exit 1; }
test -f "$IMG"    || { echo "Image not found at $IMG"; exit 1; }
test -f "$NPZ_PATH" || { echo "NPZ file not found at $NPZ_PATH"; exit 1; }

echo "[RUNNING PROD] embed=$EMBED month=$MONTH"
echo "[NPZ]    $NPZ_PATH"
echo "[SCRIPT] $SCRIPT"

srun --cpu-bind=none apptainer exec --nv \
  --bind /projects:/projects \
  --bind /lus:/lus \
  "$IMG" \
  python "$SCRIPT"