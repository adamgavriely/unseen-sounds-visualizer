#!/bin/bash
#SBATCH --job-name=lispic
#SBATCH --output=logs/lpic_%j.out
#SBATCH --error=logs/lpic_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Report-only listener job for the pictures (docs/freeze_picture_setup_2026-09-25.md).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/gold/listener_pictures.py
