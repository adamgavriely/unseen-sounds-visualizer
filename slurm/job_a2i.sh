#!/bin/bash
#SBATCH --job-name=a2i
#SBATCH --output=logs/a2i_%j.out
#SBATCH --error=logs/a2i_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Display-only audio-to-image examples (AudioToken) for the inspector's Audio-to-image tab.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/gold/a2i_audiotoken.py
