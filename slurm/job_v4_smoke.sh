#!/bin/bash
#SBATCH --job-name=v4_smoke
#SBATCH --output=logs/v4_smoke_%j.out
#SBATCH --error=logs/v4_smoke_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,L40s-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:30:00
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python scripts/v4_smoke.py --parts sam3 granite
echo DONE
