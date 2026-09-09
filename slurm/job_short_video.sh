#!/bin/bash
#SBATCH --job-name=shortvid
#SBATCH --output=logs/shortvid_%j.out
#SBATCH --error=logs/shortvid_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
python -c "import diffusers; print('diffusers', diffusers.__version__)"
python scripts/short_video.py
ls -la data/output/short_video/
