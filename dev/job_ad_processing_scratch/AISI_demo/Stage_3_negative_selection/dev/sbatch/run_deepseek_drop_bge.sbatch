#!/bin/bash
#SBATCH --job-name=z_deepseek_drop_test_bge
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --array=0-0

set -euo pipefail

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
BASE=${PROJECT}/aisi_economy_index/store/AISI_demo/stage_3/dev
VENV=${PROJECT}/.venv_embed

# DeepSeek script
SCRIPT=${PROJECT}/nbs/AISI_demo/Stage_3_negative_selection/dev/sbatch/run_deepseek_drops_fix_test.py

# Your NPZ inputs (still bge_large in this example)
NPZ_DIR=${BASE}/llm_negative_selection/bge_large

NPZ_PATHS=(
  "${NPZ_DIR}/adzuna_month01.npz"
  "${NPZ_DIR}/adzuna_month02.npz"
  "${NPZ_DIR}/adzuna_month03.npz"
  "${NPZ_DIR}/adzuna_month04.npz"
  "${NPZ_DIR}/adzuna_month05.npz"
  "${NPZ_DIR}/adzuna_month06.npz"
  "${NPZ_DIR}/adzuna_month07.npz"
  "${NPZ_DIR}/adzuna_month08.npz"
  "${NPZ_DIR}/adzuna_month09.npz"
  "${NPZ_DIR}/adzuna_month10.npz"
  "${NPZ_DIR}/adzuna_month11.npz"
  "${NPZ_DIR}/adzuna_month12.npz"
  "${NPZ_DIR}/adzuna_month13.npz"
  "${NPZ_DIR}/adzuna_month14.npz"
)

# sanity prints
echo "PROJECT=${PROJECT}"
echo "BASE=${BASE}"
echo "SCRIPT=${SCRIPT}"
echo "NPZ_DIR=${NPZ_DIR}"
ls -lh "${NPZ_DIR}" | head

module purge
module load cudatoolkit/24.11_12.6

cd "$PROJECT"
source "$VENV/bin/activate"

python -m pip install -q --no-cache-dir "pydantic>=2,<3" "accelerate>=0.30" "huggingface_hub>=0.23" >/dev/null 2>&1 || true

export EMBED=bge_large
echo "EMBED=${EMBED}"

# Use BriCS shared cache (offline) for DeepSeek
export HF_HOME=/projects/public/brics/cache
export HF_HUB_CACHE=/projects/public/brics/cache
export TRANSFORMERS_CACHE=/projects/public/brics/cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# DeepSeek model id (shared cache)
export MODEL_ID="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

export NPZ_PATH="${NPZ_PATHS[$SLURM_ARRAY_TASK_ID]}"
export MONTH_TAG="$(basename "${NPZ_PATH}" .npz)"

# range vars
export START=0
export STOP=1000

# 32B model: start conservative to avoid OOM
export BATCH_SIZE=16
export TEST_N=100


echo "[TASK] ${SLURM_ARRAY_TASK_ID} -> ${MONTH_TAG}"
echo "[NPZ]  ${NPZ_PATH}"
echo "[MODEL_ID] ${MODEL_ID}"
echo "[BATCH_SIZE] ${BATCH_SIZE}"

srun python "$SCRIPT"
