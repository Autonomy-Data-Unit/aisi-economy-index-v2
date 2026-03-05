#!/bin/bash
#SBATCH --job-name=cos_e5_only
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-27

set -euo pipefail
set -x

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
STORE=$PROJECT/aisi_economy_index/store/AISI_demo/stage_2_embeddings_and_cosines
ROOT=$STORE/dev
VENV=$PROJECT/.venv_embed

# ===============================
# EMBEDDINGS (TARGET) - E5 PATHS
# ===============================
EMB_E5=$ROOT/embeddings/run_embed_target/e5_large

# ===============================
# EMBEDDINGS (ONET) - E5 FILES
# ===============================
ONET_E5_ROLE=$ROOT/embeddings/run_embed_onet_e5/embeddings_role_onet_e5_large_v2.npy
ONET_E5_TASK=$ROOT/embeddings/run_embed_onet_e5/embeddings_tasks_onet_e5_large_v2.npy

TITLES_NPZ=$ROOT/embeddings/aspectt_vectors.npz

# ===============================
# OUTPUT - E5 DIRECTORY
# ===============================
OUT_COS_E5=$ROOT/cosines/e5_large

# ===============================
# SHARD LOGIC
# ===============================
N_SHARDS=2
TASK_ID=${SLURM_ARRAY_TASK_ID}
MONTH_IDX=$(( TASK_ID / N_SHARDS ))
SHARD_ID=$(( TASK_ID % N_SHARDS ))
MONTH=$(printf "adzuna_month%02d" $((MONTH_IDX+1)))
SHARD_TAG=$(printf "%02d" "$SHARD_ID")

module purge
module load cudatoolkit/24.11_12.6
cd "$PROJECT"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$STORE/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

mkdir -p "$OUT_COS_E5" "$STORE/.hf_cache" "$PROJECT/logs"

COS_SCRIPT=$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/cosine_month_onet_topk.py

# ===============================
# INPUT FILES
# ===============================
E5_IN="$EMB_E5/$MONTH/${MONTH}_shard${SHARD_TAG}_of02.npz"

python - <<PY
from pathlib import Path
import torch
print("TASK_ID:", "${TASK_ID}")
print("MONTH:", "${MONTH}", "SHARD:", "${SHARD_ID}")
print("E5_IN:", "${E5_IN}", "exists=", Path("${E5_IN}").exists())
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
assert Path("${E5_IN}").exists(), f"E5 Shard not found: ${E5_IN}"
assert Path("${ONET_E5_ROLE}").exists(), "ONET E5 Role file missing"
PY

# ===============================
# E5 COSINE EXECUTION
# ===============================
srun python -u "$COS_SCRIPT" \
  --month-emb-npz "$E5_IN" \
  --onet-role-npy "$ONET_E5_ROLE" \
  --onet-task-npy "$ONET_E5_TASK" \
  --titles-npz "$TITLES_NPZ" \
  --out-dir "$OUT_COS_E5" \
  --month "${MONTH}_shard${SHARD_TAG}" \
  --topk 5 \
  --chunk-rows 500 \
  --job-batch 500 \
  --dtype float16

echo "end: $(date -Is)"