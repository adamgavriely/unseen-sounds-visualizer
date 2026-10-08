#!/bin/bash
#SBATCH --job-name=v4fix
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:55:00
#SBATCH --output=logs/v4fix_%j.out
# Step 10: fixed Qwen V4 answers. SPLITS="dev dev2" or "test test2" (TEST: cache only, never scored here).
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd ~/MscProj_r13
python ~/MscProj_tg/benchmark/gold/coverage/v4fix.py run $SPLITS
