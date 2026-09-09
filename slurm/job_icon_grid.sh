#!/bin/bash
#SBATCH --job-name=icongrid
#SBATCH --output=logs/icongrid_%j.out
#SBATCH --error=logs/icongrid_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#
# Time-boxed: settle the generator/prompt question by looking at a contact sheet
# instead of rewriting one prompt at a time. Every cell is labelled with the model and
# the exact prompt that produced it. NOTE: docs/PLAN.md lists icon rendering as out of
# scope; this runs once, on request, and the remaining work is the write-up.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
python scripts/icon_grid.py --subjects Bird "Fire engine" Dog
ls -la data/output/icon_grid/SHEET_*.png 2>/dev/null
