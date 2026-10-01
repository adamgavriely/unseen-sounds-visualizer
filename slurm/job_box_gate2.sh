#!/bin/bash
#SBATCH --job-name=boxgate2
#SBATCH --output=logs/boxgate2_%j.out
#SBATCH --error=logs/boxgate2_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 38 BOX-2 (docs/prereg_round13_detector_push.md): re-prompt unparsed box replies once.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/box_gate.py reprompt
python benchmark/gold/box_gate.py score2
echo "DONE boxgate2"
