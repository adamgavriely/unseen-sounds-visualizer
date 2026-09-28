#!/bin/bash
#SBATCH --job-name=r11score
#SBATCH --output=logs/r11score_%j.out
#SBATCH --error=logs/r11score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Detector round 11 (docs/prereg_round11_masker.md): gate 0, gate R0, the cells on the 280 (and, only when the hold
# is lifted, the picks on the 415). Reads caches only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-fit}; do
  python benchmark/detector_round11.py $st
done
echo "DONE round11 score"
