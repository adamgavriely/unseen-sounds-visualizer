#!/bin/bash
#SBATCH --job-name=r6score
#SBATCH --output=logs/r6score_%j.out
#SBATCH --error=logs/r6score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Detector round 6, CPU part: c4 (caches and labels), then the 280 (gate, bars v and g, D1 + D2, the pick), then the 415
# for the pick only. Reads the caches; runs no model.
#   sbatch slurm/job_round6_score.sh            (STEPS="check fit heldout" by default)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-check fit heldout}; do
  python benchmark/detector_round6.py $st
done
echo "DONE round6 score"
