#!/bin/bash
#SBATCH --job-name=nameall
#SBATCH --output=logs/nameall_%j.out
#SBATCH --error=logs/nameall_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 66 NAME-ALL steps 0-1 on gate-gold (docs/prereg_round13_detector_push.md). Submit from ~/MscProj. Resumable per clip.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/MscProj"
python benchmark/gold/nameall.py run 2>&1 | grep -v "Loading weights"
python benchmark/gold/nameall.py score
echo "NAMEALL_DONE rc=$?"
