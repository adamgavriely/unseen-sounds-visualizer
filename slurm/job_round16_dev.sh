#!/bin/bash
#SBATCH --job-name=r16dev
#SBATCH --output=logs/r16dev_%j.out
#SBATCH --error=logs/r16dev_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 16 night arms (docs/prereg_round13_detector_push.md) on the old DEV 49 (round13_dev in ~/MscProj_r13). The
# tagger DEV part runs in ~/MscProj_tg afterwards; merged scoring with benchmark/gold/merged_dev.py.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
ARMS="${ARMS:-TO1F7F8+N1 TO1F7F8+N2 TO1F7F8+N4}"
python benchmark/gold/round13_dev.py stage4 --arms $ARMS
python benchmark/gold/round13_dev.py stage5 --arms $ARMS
python benchmark/gold/round13_dev.py score
echo "DONE r16dev"
