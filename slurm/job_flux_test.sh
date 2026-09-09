#!/bin/bash
#SBATCH --job-name=fluxtest
#SBATCH --output=logs/fluxtest_%j.out
#SBATCH --error=logs/fluxtest_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python scripts/flux_test.py
ls -la data/output/flux_test/
