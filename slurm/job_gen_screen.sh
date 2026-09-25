#!/bin/bash
#SBATCH --job-name=gen_screen
#SBATCH --output=logs/genscreen_%j.out
#SBATCH --error=logs/genscreen_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# GP-4 generator screening (benchmark/gold/gen_screen.py) in ~/venvs/gen. ARGS passed through.
#   ARGS="--model q21 --gate" sbatch slurm/job_gen_screen.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
PY="${PY:-$HOME/venvs/gen/bin/python}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" benchmark/gold/gen_screen.py ${ARGS:?}
echo "DONE"
