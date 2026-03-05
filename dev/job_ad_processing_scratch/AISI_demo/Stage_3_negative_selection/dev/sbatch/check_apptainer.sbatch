#!/bin/bash
#SBATCH --job-name=z_apptainer_check
#SBATCH --partition=workq
#SBATCH --account=brics.a5u
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=00:10:00
#SBATCH --output=/projects/a5u/adu_dev/aisi-economy-index/logs/%x_%A.out
set -euo pipefail

module purge
module load brics/userenv/2.7 brics/default/1.0
module load apptainer 2>/dev/null || module load singularity 2>/dev/null

echo "[which apptainer]"
which apptainer || true
which singularity || true
nvidia-smi

# placeholder image path, we’ll set this once we confirm what images you have
IMG=/projects/a5u/adu_dev/aisi-economy-index/containers/vllm.sif
ls -lh "$IMG" || true
