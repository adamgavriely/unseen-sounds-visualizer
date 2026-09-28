#!/bin/bash
#SBATCH --job-name=r9dev
#SBATCH --output=logs/r9dev_%j.out
#SBATCH --error=logs/r9dev_%j.err
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#
# Round 9 amendment 1 B (DEV only): BEATs frame scores on the DEV clips' audio.wav, then the J2 DEV check
# (benchmark/gold/j2_dev_check.py -> benchmark/gold/j2_dev_check.json).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/j2_dev_check.py beats
python benchmark/gold/j2_dev_check.py score
echo "DONE round9 dev"
