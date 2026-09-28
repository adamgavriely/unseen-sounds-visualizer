#!/bin/bash
#SBATCH --job-name=r5cache
#SBATCH --output=logs/r5cache_%j.out
#SBATCH --error=logs/r5cache_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#
# Detector round 5 (docs/prereg_v4.md, 2026-09-28): EAT-large and Dasheng-base AudioSet scores on BEATs' 2-s / 0.25-s
# windows, the 280 (calib) and the 415 (heldout), into eat_cache/ and dasheng_cache/ (new folders; nothing overwritten).
# Weights were fetched on the login node, so the node runs offline.
#   sbatch slurm/job_round5_cache.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for m in eat dasheng; do
  for s in calib heldout; do
    python benchmark/detector_round5.py cache --model $m --set $s
  done
done
echo "DONE round5 cache"
