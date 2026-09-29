#!/bin/bash
#SBATCH --job-name=lisp3
#SBATCH --output=logs/lisp3_%j.out
#SBATCH --error=logs/lisp3_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=6:00:00
#
# P3 (BEATs weak band) through the stricter listeners, DEV + TEST (benchmark/gold/listener_p3.py). No gold.
# Usage (after `python benchmark/gold/listener_p3.py pool`):
#   sbatch slurm/job_listener_p3.sh qwen
#   sbatch -p H200-12h,A100-4h --mem 96G --time 4:00:00 slurm/job_listener_p3.sh afn
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/gold/listener_p3.py "${1:?qwen|afn}"
