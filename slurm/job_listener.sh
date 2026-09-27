#!/bin/bash
#SBATCH --job-name=listener
#SBATCH --output=logs/lis_%j.out
#SBATCH --error=logs/lis_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Amendment 25: Qwen3-Omni-30B-A3B-Instruct yes/no scores on the candidate pool (benchmark/listener_round.py score).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/listener_round.py score --set "${SET:-calib}" --limit "${LIMIT:-0}"
