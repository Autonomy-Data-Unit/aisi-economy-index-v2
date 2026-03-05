#!/bin/bash
#SBATCH --job-name=emb_bge_gte_small
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-27

set -euo pipefail
set -x

echo "start: $(date -Is)"
echo "host:  $(hostname)"

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
STORE=$PROJECT/aisi_economy_index/store/AISI_demo/stage_2_embeddings_and_cosines
ROOT=$STORE/dev

IN_DIR=$ROOT/llama_summary_1000s
OUT_BGE=$ROOT/embeddings/run_embed_target/bge_large
OUT_GTE=$ROOT/embeddings/run_embed_target/gte_large

VENV=$PROJECT/.venv_embed

N_SHARDS=2
TASK_ID=${SLURM_ARRAY_TASK_ID}

MONTH_IDX=$(( TASK_ID / N_SHARDS ))
SHARD_ID=$(( TASK_ID % N_SHARDS ))
MONTH=$(printf "adzuna_month%02d" $((MONTH_IDX+1)))
SHARD_TAG=$(printf "%02d" "$SHARD_ID")

BGE_SCRIPT=$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/embed_target_large_bge.py
GTE_SCRIPT=$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/embed_target_large_gte.py

IN_NPZ=$IN_DIR/${MONTH}.npz
BGE_OUT_DIR=$OUT_BGE/${MONTH}
GTE_OUT_DIR=$OUT_GTE/${MONTH}

BGE_OUT_FILE=$BGE_OUT_DIR/${MONTH}_shard${SHARD_TAG}_of$(printf "%02d" ${N_SHARDS}).npz
GTE_OUT_FILE=$GTE_OUT_DIR/${MONTH}_shard${SHARD_TAG}_of$(printf "%02d" ${N_SHARDS}).npz

mkdir -p "$BGE_OUT_DIR" "$GTE_OUT_DIR" "$STORE/.hf_cache" "$PROJECT/logs" \
  "$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/logs"

module purge
module load cudatoolkit/24.11_12.6

cd "$PROJECT"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$STORE/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1

python - <<PY
from pathlib import Path
import torch
print("TASK_ID:", "${TASK_ID}")
print("MONTH:", "${MONTH}")
print("SHARD:", "${SHARD_ID}", "of", "${N_SHARDS}")
print("INPUT:", "${IN_NPZ}", "exists=", Path("${IN_NPZ}").exists())
print("BGE_SCRIPT:", "${BGE_SCRIPT}", "exists=", Path("${BGE_SCRIPT}").exists())
print("GTE_SCRIPT:", "${GTE_SCRIPT}",  "exists=", Path("${GTE_SCRIPT}").exists())
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
assert Path("${IN_NPZ}").exists()
assert Path("${BGE_SCRIPT}").exists()
assert Path("${GTE_SCRIPT}").exists()
PY

# show model constant inside scripts (avoids env-var confusion)
echo "BGE model line:"
grep -n 'MODEL_NAME' "$BGE_SCRIPT" | head -n 3 || true
echo "GTE model line:"
grep -n 'MODEL_NAME' "$GTE_SCRIPT" | head -n 3 || true

# force regeneration (avoid stale identical outputs)
rm -f "$BGE_OUT_FILE" "$GTE_OUT_FILE"

# ---------- BGE ----------
srun python -u "$BGE_SCRIPT" \
  --in-npz "$IN_NPZ" \
  --out-dir "$BGE_OUT_DIR" \
  --month "$MONTH" \
  --shard-id "$SHARD_ID" \
  --n-shards "$N_SHARDS" \
  --batch-size 512

# ---------- GTE ----------
srun python -u "$GTE_SCRIPT" \
  --in-npz "$IN_NPZ" \
  --out-dir "$GTE_OUT_DIR" \
  --month "$MONTH" \
  --shard-id "$SHARD_ID" \
  --n-shards "$N_SHARDS" \
  --batch-size 512

echo "end: $(date -Is)"
