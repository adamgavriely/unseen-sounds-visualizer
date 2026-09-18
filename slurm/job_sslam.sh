#!/bin/bash
#SBATCH --job-name=sslam
#SBATCH --output=logs/sslam_%j.out
#SBATCH --error=logs/sslam_%j.err
#SBATCH --partition=L4-4h,L4-12h,A100-4h,RTX6000-4h,L40s-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate sota
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python benchmark/sslam_cache.py
echo DONE
