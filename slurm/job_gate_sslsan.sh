#!/bin/bash
#SBATCH --job-name=gatesslsan
#SBATCH --output=logs/gatesslsan_%j.out
#SBATCH --error=logs/gatesslsan_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Round 33 SSL-SaN (docs/prereg_round13_detector_push.md): audio-visual localisation as a gate vote, all 139 gold clips.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/gate_sslsan.py run --device cuda
python benchmark/gold/gate_sslsan.py score
echo "DONE gatesslsan"
