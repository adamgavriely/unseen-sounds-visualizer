#!/bin/bash
#SBATCH --job-name=bandlist
#SBATCH --output=logs/bandlist_%j.out
#SBATCH --error=logs/bandlist_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#
# Round 49 BANDLIST (docs/prereg_round13_detector_push.md): shipped gate on the band-run candidates, then score. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/bandlist_screen.py gate
python benchmark/gold/bandlist_screen.py score
echo "DONE bandlist"
