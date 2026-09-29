#!/bin/bash
#SBATCH --job-name=r14k2
#SBATCH --output=logs/r14k2_%j.out
#SBATCH --error=logs/r14k2_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 14 amendment K2 supplement (DEV): listener answers for the 27 runs K2 newly offers (Qwen3-Omni variants + AF Next,
# benchmark/gold/listener_k2.py, their code imported unchanged), then arms TO1F7F8+K2x and TO1F7F8+K2K3x (round13_dev.py).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
if ! python -c "import json,sys; d=json.load(open('benchmark/gold/dev_listener_k2_afn.json')); sys.exit(0 if all('accept' in x for x in d['items']) else 1)" 2>/dev/null; then
  python benchmark/gold/listener_k2.py pool
  python benchmark/gold/listener_k2.py score
fi
python benchmark/gold/round13_dev.py stage4 --arms TO1F7F8+K2x TO1F7F8+K2K3x
python benchmark/gold/round13_dev.py stage5 --arms TO1F7F8+K2x TO1F7F8+K2K3x
python benchmark/gold/round13_dev.py score
echo "DONE r14k2"
