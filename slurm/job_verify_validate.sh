#!/bin/bash
#SBATCH --job-name=vfy_val
#SBATCH --output=logs/vfyval_%j.out
#SBATCH --error=logs/vfyval_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Validate the picture check (scripts/verify_validate.py) before any redraw: draw the deliberately wrong set
# (Qwen-Image-2512), then check the 82 shipped pictures + the wrong set twice (Qwen3.8-27B + EasyOCR).
#   PHASE=all sbatch slurm/job_verify_validate.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/verify_validate.py --phase "${PHASE:-all}"
echo "DONE verify_validate ${PHASE:-all}"
