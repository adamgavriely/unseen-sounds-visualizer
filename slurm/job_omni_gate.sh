#!/bin/bash
#SBATCH --job-name=omnigate
#SBATCH --output=logs/omnigate_%j.out
#SBATCH --error=logs/omnigate_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 15 amendment N (docs/prereg_round13_detector_push.md): Qwen3-Omni audio-visual gate vote, DEV screen.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/omni_gate.py run
python benchmark/gold/omni_gate.py score
echo "DONE omnigate"
