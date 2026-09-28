#!/bin/bash
#SBATCH --job-name=r5score
#SBATCH --output=logs/r5score_%j.out
#SBATCH --error=logs/r5score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Detector round 5, CPU part: gates 2-3 (windows, labels, order), then the 280 (gate 1, four cells, the pick),
# then the 415 for the pick only. Reads the caches; runs no model.
#   sbatch slurm/job_round5_score.sh            (STEPS="check fit heldout" by default)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-check fit heldout}; do
  python benchmark/detector_round5.py $st
done
echo "DONE round5 score"
