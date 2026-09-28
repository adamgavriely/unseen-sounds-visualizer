#!/bin/bash
#SBATCH --job-name=psedheld
#SBATCH --output=logs/psedh_%j.out
#SBATCH --error=logs/psedh_%j.err
#SBATCH --partition=H200-4h,A100-4h,generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# PretrainedSED (BEATs strong) frame scores on the held-out 415 AudioSet-Strong clips (cache only, no scores printed).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate psed
export PYTHONUNBUFFERED=1
python -m benchmark.audioset_detector_eval --cache psed --set heldout
