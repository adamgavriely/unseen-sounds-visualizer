#!/bin/bash
#SBATCH --job-name=r8score
#SBATCH --output=logs/r8score_%j.out
#SBATCH --error=logs/r8score_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#
# Detector round 8 (docs/prereg_round8_ideas.md), CPU part. Reads the caches; runs no model.
#   STEPS="views" sbatch slurm/job_round8_score.sh        (band-limited wavs for both sets)
#   STEPS="gate" | "fit" | "check" | "heldout"            (CELLS="I3 I4 ..." limits fit to those cells)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
for st in ${STEPS:-gate}; do
  if [ "$st" = "views" ]; then
    python benchmark/detector_round8.py views --set calib
    python benchmark/detector_round8.py views --set heldout
  elif [ "$st" = "fit" ] && [ -n "${CELLS:-}" ]; then
    python benchmark/detector_round8.py fit --cells $CELLS
  else
    python benchmark/detector_round8.py $st
  fi
done
echo "DONE round8 score $STEPS"
