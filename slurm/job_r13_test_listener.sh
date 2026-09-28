#!/bin/bash
#SBATCH --job-name=testlis
#SBATCH --output=logs/testlis_%j.out
#SBATCH --error=logs/testlis_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# Round 13 TEST PREPARATION: Qwen3-Omni listener scores on the gold-free TEST superset pool
# (benchmark/gold/test_listener.py; pool built on the login node). No gold, no scoring. Submit from ~/MscProj_r13. Resumable.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/test_listener.py score
echo "DONE testlis"
