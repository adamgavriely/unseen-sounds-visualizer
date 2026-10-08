#!/bin/bash
#SBATCH --job-name=s5mix
#SBATCH --partition=generic
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/s5_mix_%j.out
# Step 5: training mixtures (resumes per shard). Submit from ~/MscProj_tg.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/gold/coverage/build_mixtures.py 20000 1000
