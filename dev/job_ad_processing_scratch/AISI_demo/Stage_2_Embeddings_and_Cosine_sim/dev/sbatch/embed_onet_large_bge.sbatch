#!/bin/bash
#SBATCH --job-name=emb_onet_bge_only
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

# Only BGE script path retained
SCRIPT_BGE=$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/embed_onet_bge_large.py

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
echo "Running BGE embeddings (O*NET)"
echo "=============================="

export MODEL_NAME="BAAI/bge-large-en-v1.5"
srun python -u "$SCRIPT_BGE"

echo "=============================="
echo "O*NET BGE embeddings finished"
echo "=============================="