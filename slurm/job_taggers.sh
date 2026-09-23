#!/bin/bash
#SBATCH --job-name=taggers
#SBATCH --output=logs/tag_%j.out
#SBATCH --error=logs/tag_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/gold/extra_taggers.py --model ${MODEL:?} --shard ${SHARD:-0} --of ${OF:-1}
