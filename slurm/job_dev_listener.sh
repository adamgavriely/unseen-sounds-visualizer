#!/bin/bash
#SBATCH --job-name=devlis
#SBATCH --output=logs/devlis_%j.out
#SBATCH --error=logs/devlis_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Round 13: Qwen3-Omni listener scores on the DEV candidate pool (benchmark/gold/dev_listener.py score). DEV only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/gold/dev_listener.py score
