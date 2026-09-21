#!/bin/bash
#SBATCH --job-name=screen_m2
#SBATCH --output=logs/screen_m2_%j.out
#SBATCH --error=logs/screen_m2_%j.err
#SBATCH --partition=A100-4h,H200-4h,L40s-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
# VLM screen of the wave-2 mixed candidates (scripts/screen_mixed2.py): the v4b3 gate without pictures.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
python scripts/screen_mixed2.py data/input/benchmark/unsorted 'm2_*.mp4'
echo DONE
