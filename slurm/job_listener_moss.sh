#!/bin/bash
#SBATCH --job-name=lismoss
#SBATCH --output=logs/lismoss_%j.out
#SBATCH --error=logs/lismoss_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#
# Round 34 MOSS: MOSS-Audio-8B-Thinking as a listener (V4) on the P2/PV items already answered by Qwen3-Omni and Audio
# Flamingo Next, and on the P1 sanity items (benchmark/gold/listener_moss.py). Generation in ~/venv_moss (msproj +
# transformers 4.57.1, official code ~/MOSS-Audio), matching in msproj (the stack that scored Qwen/AF). No gold.
# Usage (from ~/MscProj_tg): sbatch slurm/job_listener_moss.sh "dev" [SMOKE_N]   (splits: dev dev2 p1; SMOKE_N > 0: *_smoke.json)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SMOKE="${2:-0}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for split in ${1:-dev}; do
    "$HOME/venv_moss/bin/python" benchmark/gold/listener_moss.py gen "$split" --smoke "$SMOKE"
    ( source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
      python benchmark/gold/listener_moss.py match "$split" --smoke "$SMOKE" )
done
echo "DONE lismoss"
