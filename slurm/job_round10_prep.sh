#!/bin/bash
#SBATCH --job-name=r10prep
#SBATCH --output=logs/r10prep_%j.out
#SBATCH --error=logs/r10prep_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#
# Detector round 10 (docs/prereg_round10_rescue.md), CPU: label-free band candidates, 16-kHz wavs, the 3 perturbed
# versions for candidate clips, and the GPU work lists. Every step skips what exists.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
df -h "$HOME" | tail -1
for s in ${SETS:-calib heldout}; do
  python benchmark/detector_round10.py prep --set $s
done
df -h "$HOME" | tail -1
echo "DONE round10 prep"
