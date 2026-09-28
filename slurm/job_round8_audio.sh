#!/bin/bash
#SBATCH --job-name=r8audio
#SBATCH --output=logs/r8audio_%j.out
#SBATCH --error=logs/r8audio_%j.err
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Detector round 8, GPU audio caches (every step skips what exists, so a timeout is a resubmit):
#   PART=beats  : BEATs 1-s windows + BEATs on the LP/HP views, both sets
#   PART=flex   : FlexSED (same runner, 215 families) on the views; SETS / VIEWS choose which (default all four)
# The views (data/work/round8_views/<set>/<lp|hp>/<id>__<v>.wav) are made first by job_round8_score.sh STEPS=views.
# The view wavs carry the suffix __lp / __hp so flexsed_run's wav copy (data/work/gold_wav_flat/<stem>.wav) can never
# reuse an original clip's wav.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if [ "${PART:-beats}" = "beats" ]; then
  export HF_HUB_OFFLINE=1
  python benchmark/detector_round8.py beats --set calib
  python benchmark/detector_round8.py beats --set heldout
else
  for s in ${SETS:-calib heldout}; do
    for v in ${VIEWS:-lp hp}; do
      python benchmark/gold/flexsed_run.py --clip-dir "$SLURM_SUBMIT_DIR/data/work/round8_views/$s/$v" \
          --out "$SLURM_SUBMIT_DIR/data/work/round8_flexsed_${s}_${v}" --batch ${BATCH:-24}
    done
  done
fi
echo "DONE round8 audio ${PART:-beats}"
