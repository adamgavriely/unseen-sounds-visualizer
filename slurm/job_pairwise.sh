#!/bin/bash
#SBATCH --job-name=pairwise
#SBATCH --output=logs/pairwise_%j.out
#SBATCH --error=logs/pairwise_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# GP-4 point 4: the pairwise check (benchmark/gold/pairwise.py), GLM-4.6V-Flash.
#   sbatch slurm/job_pairwise.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="${PY:-$HOME/venvs/judge/bin/python}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" benchmark/gold/pairwise.py ${ARGS:-calibrate}
echo "DONE"
