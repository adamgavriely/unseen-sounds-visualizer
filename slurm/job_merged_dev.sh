#!/bin/bash
#SBATCH --job-name=mdev
#SBATCH --output=logs/mdev_%j.out
#SBATCH --error=logs/mdev_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Merged-DEV selection (docs/prereg_round13_detector_push.md): the candidate arms on the tagger DEV part (tagger_prep,
# TG_ARMS), gates, then benchmark/gold/merged_dev.py on DEV + tagger DEV. TEST parts are not scored. Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CAND="TO1+F7 TO1+F7F8+FIX TO1F7F8+K3 A1 LR-V12+1+F1F4F3+F7F8 LR-V12+1+F1F4F3+F7F8@AG4 R13-1"
export TG_ARMS="$CAND"
P=benchmark/gold/tagger_prep.py
python benchmark/gold/tagger_coverage.py
python $P --split dev2 stage4 --arms B0r B1 "TO1+F7F8" $CAND
python $P --split dev2 stage5 --arms B0r B1 "TO1+F7F8" $CAND
python $P --split dev2 gates --arms B0r B1 "TO1+F7F8" $CAND
python benchmark/gold/merged_dev.py --arms B0r B1 "TO1+F7F8" $CAND
echo "DONE mdev"
