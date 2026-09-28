#!/bin/bash
#SBATCH --job-name=r7score
#SBATCH --output=logs/r7score_%j.out
#SBATCH --error=logs/r7score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Detector round 7, CPU part: identity gate, then the 280 (baseline gate, S1, S2, the pick), then the 415 for the pick
# only. Reads the caches; runs no model.
#   sbatch slurm/job_round7_score.sh            (STEPS="identity fit heldout" by default)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-identity fit heldout}; do
  python benchmark/detector_round7.py $st
done
echo "DONE round7 score"
