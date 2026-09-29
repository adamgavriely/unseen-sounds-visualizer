#!/bin/bash
#SBATCH --job-name=lisafn
#SBATCH --output=logs/lisafn_%j.out
#SBATCH --error=logs/lisafn_%j.err
#SBATCH --partition=H200-12h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#
# Round 14 amendment C: Audio Flamingo Next as a second listener (V4 open inventory + yes/no) on the amendment-A candidates,
# DEV + TEST (benchmark/gold/listener_afnext.py). No gold. Usage: sbatch slurm/job_listener_afnext.sh [smoke|score|both]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/gold/listener_afnext.py "${1:-score}"
