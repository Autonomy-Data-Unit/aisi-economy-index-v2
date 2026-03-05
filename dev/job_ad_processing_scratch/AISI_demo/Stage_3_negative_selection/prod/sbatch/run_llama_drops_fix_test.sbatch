#!/bin/bash
#SBATCH --job-name=z_llama_drop_new_test
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0

set -euo pipefail

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
BASE=${PROJECT}/aisi_economy_index/store/AISI_demo/stage_3/dev
VENV=${PROJECT}/.venv_embed
SCRIPT=${PROJECT}/nbs/AISI_demo/Stage3_negative_selection/dev/sbatch/run_llama_drops_fix_test.py
NPZ_DIR=${BASE}/llm_negative_selection

NPZ_PATHS=(
  "${NPZ_DIR}/adzuna_month01.npz"

)
# "${NPZ_DIR}/adzuna_month03_test01.npz"
# "${NPZ_DIR}/adzuna_month04_test01.npz"
# "${NPZ_DIR}/adzuna_month05_test01.npz"
# "${NPZ_DIR}/adzuna_month06_test01.npz"
# "${NPZ_DIR}/adzuna_month07_test01.npz"
# "${NPZ_DIR}/adzuna_month08_test01.npz"
# "${NPZ_DIR}/adzuna_month09_test01.npz"
# "${NPZ_DIR}/adzuna_month10_test01.npz"
# "${NPZ_DIR}/adzuna_month11_test01.npz"
# "${NPZ_DIR}/adzuna_month12_test01.npz"
# "${NPZ_DIR}/adzuna_month13_test01.npz"
# "${NPZ_DIR}/adzuna_month14_test01.npz"

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

python -m pip install -q --no-cache-dir "pydantic>=2,<3" "accelerate>=0.30" >/dev/null 2>&1 || true

export HF_HOME="$PROJECT/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

export NPZ_PATH="${NPZ_PATHS[$SLURM_ARRAY_TASK_ID]}"
export MONTH_TAG="$(basename "${NPZ_PATH}" .npz)"

# range vars not really needed for test NPZs, but keep to satisfy script
export START=0
export STOP=1000

echo "[TASK] ${SLURM_ARRAY_TASK_ID} -> ${MONTH_TAG}"
echo "[NPZ]  ${NPZ_PATH}"

srun python "$SCRIPT"
