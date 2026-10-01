#!/bin/bash
#SBATCH --job-name=relgate
#SBATCH --output=logs/relgate_%j.out
#SBATCH --error=logs/relgate_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#
# Round 39 RELABEL-GATE (docs/prereg_round13_detector_push.md): live shipped gate on the relabelled pictures scan left open.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/relabel_gate_screen.py gate
python benchmark/gold/relabel_gate_screen.py score
echo "DONE relgate"
