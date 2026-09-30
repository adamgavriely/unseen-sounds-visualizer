#!/bin/bash
#SBATCH --job-name=subj
#SBATCH --output=logs/subj_%j.out
#SBATCH --error=logs/subj_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 31 SUBJ: scene-subject silence, VLM answers (text only).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/subj_screen.py ask
python benchmark/gold/subj_screen.py score
echo "DONE subj"
