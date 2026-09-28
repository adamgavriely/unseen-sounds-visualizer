#!/bin/bash
#SBATCH --job-name=r9score
#SBATCH --output=logs/r9score_%j.out
#SBATCH --error=logs/r9score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Detector round 9 (docs/prereg_round9_contrast.md): J1 / J2 on the 280, then the picks on the 415. Reads caches only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-fit heldout}; do
  python benchmark/detector_round9.py $st
done
echo "DONE round9 score"
