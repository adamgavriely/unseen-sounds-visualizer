#!/bin/bash
#SBATCH --job-name=obvious
#SBATCH --output=logs/obvious_%j.out
#SBATCH --error=logs/obvious_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
# Step 3b (benchmark/gold/coverage/PREREG_step3b_omni_verify.md). Submit from ~/MscProj_tg; resumes from its cache.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj; source ~/venvs/qomni_av/bin/activate
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FFMPEG="$HOME/miniconda3/envs/msproj/bin/ffmpeg"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/coverage/obvious_check.py run
echo OBVIOUS_DONE
