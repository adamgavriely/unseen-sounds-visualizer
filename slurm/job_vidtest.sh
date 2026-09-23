#!/bin/bash
#SBATCH --job-name=vidtest
#SBATCH --output=logs/vid_%j.out
#SBATCH --error=logs/vid_%j.err
#SBATCH --partition=A100-4h,L40s-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=03:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/gold/video_vs_images.py --half ${HALF:-dev}
