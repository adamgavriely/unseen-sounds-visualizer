#!/bin/bash
#SBATCH --job-name=sign
#SBATCH --output=logs/sign_%j.out
#SBATCH --error=logs/sign_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 54 SIGN (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg.
# Step 1: gate-gold screen in ~/MscProj (caches live there). Step 2: saved SHIP8+MD3 merged-DEV pictures in ~/MscProj_tg.
# Step 1 does not gate step 2.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
(cd "$HOME/MscProj" && python benchmark/gold/sign_gate.py run && python benchmark/gold/sign_gate.py score)
cd "$HOME/MscProj_tg"
export TG_ARMS="SHIP8+MD3"
python benchmark/gold/sign_screen.py run && python benchmark/gold/sign_screen.py score
