#!/bin/bash
#SBATCH --job-name=hboxgate
#SBATCH --output=logs/hboxgate_%j.out
#SBATCH --error=logs/hboxgate_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 50 HUMAN-BOX (docs/prereg_round13_detector_push.md): ground HUMAN-2's named phrase, crop, ask the crop. DEV gate-gold screen.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
md5sum benchmark/gold/annotations/gold_AG.json
python benchmark/gold/humanbox_gate.py run
python benchmark/gold/humanbox_gate.py score
echo "DONE hboxgate"
