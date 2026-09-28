#!/bin/bash
#SBATCH --job-name=horn_tr
#SBATCH --output=logs/horntr_%j.out
#SBATCH --error=logs/horntr_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#
# Horn wordings on the sliceB Honk clip + the VLM-written looks / look-alikes preview (scripts/horn_trials.py, 28 Sept).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/horn_trials.py
echo "DONE horn_trials"
