#!/bin/bash
#SBATCH --job-name=avsync
#SBATCH --output=logs/sync_%j.out
#SBATCH --error=logs/sync_%j.err
#SBATCH --partition=A100-4h,L40s-4h,H200-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Audio-visual synchrony (Adam's idea): does the detected object's region change AT the instant the
# sound starts, relative to the background? Three "is the source on screen" experiments all landed
# on the break-even line because a concept detector answers presence, not source. Synchrony is the
# one signal that could tell a real source from a bystander, and nobody has measured it.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/av_synchrony.py --half "${HALF:-dev}"
