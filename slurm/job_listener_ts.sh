#!/bin/bash
#SBATCH --job-name=lists
#SBATCH --output=logs/lists_%j.out
#SBATCH --error=logs/lists_%j.err
#SBATCH --partition=H200-12h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#
# Round 14 amendment J4: audio-LLM onset timestamps on the weak / rescued runs, DEV + TEST, one model per job
# (benchmark/gold/listener_timestamps.py run --model {qwen|afn}). No gold. Usage: sbatch slurm/job_listener_ts.sh qwen|afn
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/gold/listener_timestamps.py run --model "$1"
