#!/bin/bash
#SBATCH --job-name=gground
#SBATCH --output=logs/gground_%j.out
#SBATCH --error=logs/gground_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#
# Grounded visibility gate, DEV only (benchmark/gold/gate_grounding_dev.py): Qwen3.8-27B noun + boxes on every
# stretch the scored gate called seen, then OWLv2 and SAM 3 detections; then the CPU scoring and overlays.
#   sbatch slurm/job_gate_grounding_dev.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/gate_grounding_dev.py collect
python benchmark/gold/gate_grounding_dev.py score
python benchmark/gold/gate_grounding_dev.py overlays
echo "DONE gate_grounding_dev"
