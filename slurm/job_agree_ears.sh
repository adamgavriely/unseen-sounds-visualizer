#!/bin/bash
#SBATCH --job-name=agreeears
#SBATCH --output=logs/agreeears_%j.out
#SBATCH --error=logs/agreeears_%j.err
#SBATCH --partition=H200-4h,A100-4h,L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#
# Round 45 AGREE-EARS (docs/prereg_round13_detector_push.md): Audio Flamingo Next whole-clip lists (+ yes/no at the onset cut) on
# merged DEV, merged TEST and the 415 held-out, then the CPU score (held-out first, DEV, TEST, pooled). From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/agree_ears_screen.py listen
python benchmark/gold/agree_ears_screen.py score
echo "DONE agreeears"
