#!/bin/bash
#SBATCH --job-name=r12score
#SBATCH --output=logs/r12score_%j.out
#SBATCH --error=logs/r12score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#
# Detector round 12 (docs/prereg_round12_v2.md): v2 cost, reads caches only. STEPS: step1 | rescore | new | picks | heldout
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-step1}; do
  python benchmark/detector_round12.py $st
done
echo "DONE round12 score"
