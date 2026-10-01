#!/bin/bash
#SBATCH --job-name=human2gate
#SBATCH --output=logs/human2gate_%j.out
#SBATCH --error=logs/human2gate_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 43b HUMAN-2 (docs/prereg_round13_detector_push.md): strict-format open question + majority of stretches. DEV screen.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum benchmark/gold/annotations/gold_AG.json
python benchmark/gold/human2_gate.py run
python benchmark/gold/human2_gate.py score
echo "DONE human2gate"
