#!/bin/bash
#SBATCH --job-name=sign2b
#SBATCH --output=logs/sign2b_%j.out
#SBATCH --error=logs/sign2b_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 54b SIGN-2 (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg.
# Step 0 sanity (10 DEV stretches): exit 3 = >= 9 share one answer pair -> STOP, steps 1-2 not run.
# Step 1 gate-gold in ~/MscProj (reported); step 2 saved SHIP8+MD3 merged-DEV pictures in ~/MscProj_tg (decides).
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS="SHIP8+MD3"
cd "$HOME/MscProj_tg"
python benchmark/gold/sign2_screen.py sanity || { echo "SANITY STOP or error (exit $?)"; exit 0; }
(cd "$HOME/MscProj" && python benchmark/gold/sign2_gate.py run && python benchmark/gold/sign2_gate.py score)
cd "$HOME/MscProj_tg"
python benchmark/gold/sign2_screen.py run && python benchmark/gold/sign2_screen.py score
