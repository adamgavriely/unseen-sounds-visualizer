#!/bin/bash
#SBATCH --job-name=r10score
#SBATCH --output=logs/r10score_%j.out
#SBATCH --error=logs/r10score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#
# Detector round 10 scoring (reads caches only). STEPS: check | fit | heldout (heldout only after the setup audit is read).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-fit}; do
  python benchmark/detector_round10.py $st
done
echo "DONE round10 score"
