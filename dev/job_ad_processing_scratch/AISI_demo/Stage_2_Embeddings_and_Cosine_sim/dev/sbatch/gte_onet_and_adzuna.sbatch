#!/bin/bash
#SBATCH --job-name=emb_target_gte_large
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --array=0-27

set -euo pipefail
set -x

PROJECT=/projects/a5u/adu_dev/aisi-economy-index
ROOT=$PROJECT/aisi_economy_index/store/AISI_demo/stage_2_embeddings_and_cosines/dev
VENV=$PROJECT/.venv_embed

IN_DIR=$ROOT/llama_summary_1000s
OUT_DIR=$ROOT/embeddings/run_embed_target/gte_large

SCRIPT_GTE=$PROJECT/nbs/AISI_demo/Stage_2_Embeddings_and_Cosine_sim/dev/sbatch/embed_target_large_gte.py

N_SHARDS=2
TASK_ID=${SLURM_ARRAY_TASK_ID}

MONTH_IDX=$(( TASK_ID / N_SHARDS ))
SHARD_ID=$(( TASK_ID % N_SHARDS ))

MONTH=$(printf "adzuna_month%02d" $((MONTH_IDX+1)))
SHARD_TAG=$(printf "%02d" "$SHARD_ID")

IN_NPZ=$IN_DIR/${MONTH}.npz
OUT_MONTH_DIR=$OUT_DIR/${MONTH}
OUT_FILE=$OUT_MONTH_DIR/${MONTH}_shard${SHARD_TAG}_of$(printf "%02d" ${N_SHARDS}).npz

mkdir -p "$OUT_MONTH_DIR" "$ROOT/.hf_cache" "$PROJECT/logs"

module purge
module load cudatoolkit/24.11_12.6

cd "$PROJECT"
source "$VENV/bin/activate"

export PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/.hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_TELEMETRY=1
export MODEL_NAME="Alibaba-NLP/gte-large-en-v1.5"

python - <<PY
from pathlib import Path
import torch
print("MONTH:", "${MONTH}")
print("SHARD:", "${SHARD_ID}", "of", "${N_SHARDS}")
print("INPUT:", "${IN_NPZ}", Path("${IN_NPZ}").exists())
print("OUTPUT:", "${OUT_FILE}")
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))
assert Path("${IN_NPZ}").exists()
PY

# força regeneração
rm -f "$OUT_FILE"

srun python -u "$SCRIPT_GTE" \
  --in-npz "$IN_NPZ" \
  --out-dir "$OUT_MONTH_DIR" \
  --month "$MONTH" \
  --shard-id "$SHARD_ID" \
  --n-shards "$N_SHARDS" \
  --batch-size 512

echo "DONE TARGET GTE ${MONTH} shard ${SHARD_ID}"
