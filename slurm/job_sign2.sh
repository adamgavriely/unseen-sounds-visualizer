#!/bin/bash
#SBATCH --job-name=sign2
#SBATCH --output=logs/sign2_%j.out
#SBATCH --error=logs/sign2_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 54 SIGN step 2 only (re-run after job_sign.sh's step 2 failed on an import). Submit from ~/MscProj_tg.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS="SHIP8+MD3"
cd "$HOME/MscProj_tg"
python benchmark/gold/sign_screen.py run && python benchmark/gold/sign_screen.py score
