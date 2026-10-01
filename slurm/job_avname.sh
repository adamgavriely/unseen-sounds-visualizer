#!/bin/bash
#SBATCH --job-name=avname
#SBATCH --output=logs/avname_%j.out
#SBATCH --error=logs/avname_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#
# Round 44 AVNAME (docs/prereg_round13_detector_push.md): Qwen3-Omni names every placed SHIP8 picture from audio + frames. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/avname_screen.py ask
python benchmark/gold/avname_screen.py score
echo "DONE avname"
