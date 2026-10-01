#!/bin/bash
#SBATCH --job-name=grpp
#SBATCH --output=logs/grpp_%j.out
#SBATCH --error=logs/grpp_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 51 GRP-P (docs/prereg_round13_detector_push.md). From ~/MscProj_tg.
# Step 1 (held-out 415 calibration) always; step 2 (merged DEV) only if step 1 passes the bar (score415 exit 0; exit 3 = below).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS="SHIP8+MD3"
python benchmark/gold/grpp_screen.py pairs415
python benchmark/gold/grpp_screen.py ask415
set +e
python benchmark/gold/grpp_screen.py score415
rc=$?
set -e
if [ "$rc" -eq 0 ]; then
  echo "STEP 1 PASS -> step 2 (merged DEV)"
  python benchmark/gold/grpp_screen.py pairsdev
  python benchmark/gold/grpp_screen.py askdev
  python benchmark/gold/grpp_screen.py scoredev
elif [ "$rc" -eq 3 ]; then
  echo "STEP 1 below the bar -> STOP (no step 2)"
else
  echo "score415 failed with exit $rc"; exit "$rc"
fi
echo "DONE grpp"
