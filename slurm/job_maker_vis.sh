#!/bin/bash
#SBATCH --job-name=makervis
#SBATCH --output=logs/makervis_%j.out
#SBATCH --error=logs/makervis_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#
# Round 39 MAKER-VIS (docs/prereg_round13_detector_push.md): gate VLM asked whether the depiction's MAKER is visible; from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/maker_vis_screen.py makers
python benchmark/gold/maker_vis_screen.py run
python benchmark/gold/maker_vis_screen.py score
echo "DONE makervis"
