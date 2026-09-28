#!/bin/bash
#SBATCH --job-name=lisvar
#SBATCH --output=logs/lisvar_%j.out
#SBATCH --error=logs/lisvar_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#
# Round 13 amendment A: listener variants V1-V4 on DEV + TEST candidates (benchmark/gold/listener_variants.py score). No gold.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/gold/listener_variants.py score
