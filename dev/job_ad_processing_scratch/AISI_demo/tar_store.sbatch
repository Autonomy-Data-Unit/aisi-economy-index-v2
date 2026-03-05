#!/bin/bash
#SBATCH --job-name=z_tar_store
#SBATCH --partition=workq
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.out
#SBATCH --error=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --time=24:00:00

set -euo pipefail

SRC=/lus/lfs1aip2/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store
OUT=/lus/lfs1aip2/projects/a5u/adu_dev/aisi-economy-index/aisi_economy_index/store_20260227.tar.gz

mkdir -p "$(dirname "$OUT")"

echo "Start UTC: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
echo "Host: $(hostname)"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Source: $SRC"
echo "Output: $OUT"

command -v pigz >/dev/null 2>&1 || { echo "pigz not found"; exit 1; }
test ! -f "$OUT" || { echo "Refusing to overwrite existing $OUT"; exit 1; }

du -sb "$SRC" | awk '{print "Input bytes:", $1}'

tar cf - -C "$(dirname "$SRC")" "$(basename "$SRC")" \
  | pigz -p "$SLURM_CPUS_PER_TASK" \
  > "$OUT"

echo "Done UTC: $(date -u +"%Y-%m-%d %H:%M:%S UTC")"
ls -lh "$OUT"