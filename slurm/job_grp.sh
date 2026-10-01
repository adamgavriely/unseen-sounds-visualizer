#!/bin/bash
#SBATCH --job-name=grp
#SBATCH --output=logs/grp_%j.out
#SBATCH --error=logs/grp_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=02:00:00
#
# Round 47 GROUP (docs/prereg_round13_detector_push.md). From ~/MscProj_tg. SET=dev|test
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
python benchmark/gold/grp_screen.py ask "$SET"
python benchmark/gold/grp_screen.py score "$SET"
echo "DONE grp $SET"
