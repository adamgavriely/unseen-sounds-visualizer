#!/bin/bash
#SBATCH --job-name=sota_smoke
#SBATCH --output=logs/sota_smoke_%j.out
#SBATCH --error=logs/sota_smoke_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#
# Smoke test of the SOTA candidates (scripts/sota_smoke.py) in the 'sota' env on a big GPU:
# FLAM detector, Qwen3.8-27B visibility question, Z-Image-Turbo picture. Weights must be
# in the HF cache (login-node download).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate sota
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/sota_smoke.py --parts ${PARTS:-detector vlm images}
echo DONE
