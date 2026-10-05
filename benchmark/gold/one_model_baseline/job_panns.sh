#!/bin/bash
#SBATCH --job-name=omb_panns
#SBATCH --partition=A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/dsi/adamg/MscProj_tg/benchmark/gold/one_model_baseline/job_panns_%j.log
set -euo pipefail
cd ~/MscProj_tg
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/gold/one_model_baseline/cache_panns.py
