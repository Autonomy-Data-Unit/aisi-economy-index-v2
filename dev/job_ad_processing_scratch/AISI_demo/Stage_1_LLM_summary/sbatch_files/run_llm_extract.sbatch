#!/bin/bash
#SBATCH --job-name=llama_extract_all
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=160G
#SBATCH --time=12:00:00
#SBATCH --array=0-99

set -euo pipefail

# args (passed at submission time from Jupyter/controller)
MONTH="$1"      # e.g. adzuna_month02
N_ROWS="$2"     # e.g. 2692671
SHARD="$3"      # e.g. ceil(N_ROWS/100)
NPZ_PATH="$4"   # full path to the .npz

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
VENV=$PROJECT/.venv_embed
SCRIPT=$PROJECT/nbs/isambard/2026_01/adzuna_22_all_months/run_llm_extract.py

module purge
module load cudatoolkit/24.11_12.6

cd "$PROJECT"
source "$VENV/bin/activate"

export HF_HOME="$PROJECT/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false

export TMPDIR="$PROJECT/tmp"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
export JOBLIB_TEMP_FOLDER="$TMPDIR"

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256

export TORCH_THREADS=$SLURM_CPUS_PER_TASK
export FLASH_ATTN=1
export TORCH_COMPILE=0
export INFERENCE_MODE=1

# pass context to python
export MONTH="$MONTH"
export N_ROWS="$N_ROWS"
export SHARD="$SHARD"
export NPZ_PATH="$NPZ_PATH"

export START=$(( SLURM_ARRAY_TASK_ID * SHARD ))
export STOP=$(( (SLURM_ARRAY_TASK_ID + 1) * SHARD ))

# skip empty tasks
if [ "$START" -ge "$N_ROWS" ]; then
  echo "[SKIP] month=$MONTH task=$SLURM_ARRAY_TASK_ID START=$START >= N_ROWS=$N_ROWS"
  exit 0
fi

# clamp last chunk
if [ "$STOP" -gt "$N_ROWS" ]; then
  export STOP="$N_ROWS"
fi

export BATCH_SIZE=64
export GEN_MAX_TOKENS=220

echo "[RUN] month=$MONTH task=$SLURM_ARRAY_TASK_ID START=$START STOP=$STOP shard=$SHARD rows=$N_ROWS threads=$TORCH_THREADS"

srun python "$SCRIPT"
