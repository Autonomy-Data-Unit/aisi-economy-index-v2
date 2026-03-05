#!/bin/bash
#SBATCH --job-name=emb_onet_e5_only
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/logs/%x_%j.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=64G
#SBATCH --time=02:00:00

set -euo pipefail

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
STORE=$PROJECT/aisi_economy_index/store/AISI_demo/stage_2_embeddings_and_cosines/dev/embeddings
VENV=$PROJECT/.venv_embed

# Updated to point to your E5-adapted script
# Ensure you have saved the E5 version of the python code to this path
SCRIPT_E5=$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/embed_onet_e5_large.py

module purge
module load cudatoolkit/24.11_12.6

cd "$PROJECT"
source "$VENV/bin/activate"

mkdir -p "$PROJECT/logs" "$STORE/.hf_cache" \
  "$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/logs"

export PYTHONUNBUFFERED=1
export HF_HOME="$STORE/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

export BATCH_SIZE=64
export CHUNK_SIZE=1000

# GPU Sanity Check
python - <<'PY'
import torch
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY

echo "=============================="
echo "Running E5 embeddings (O*NET)"
echo "=============================="

# Updated model constant
export MODEL_NAME="intfloat/e5-large-v2"
srun python -u "$SCRIPT_E5"

echo "=============================="
echo "O*NET E5 embeddings finished"
echo "=============================="