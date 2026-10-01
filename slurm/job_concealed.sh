#!/bin/bash
#SBATCH --job-name=conceal
#SBATCH --output=logs/conceal_%j.out
#SBATCH --error=logs/conceal_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 64 CONCEALED-ACTION (docs/prereg_round13_detector_push.md): D = SHIP8+MD3+WW5 vs +CA (ship table {Bell}) and +CAR
# (report-only table) on old DEV (~/MscProj_r13) and DEV2 (~/MscProj_tg), merged DEV. Submit from ~/MscProj_tg.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
A="SHIP8+MD3+WW5 SHIP8+MD3+WW5+CA SHIP8+MD3+WW5+CAR"
export TG_ARMS="$A"
cd "$HOME/MscProj_r13"
python benchmark/gold/round13_dev.py stage4 --arms $A
python benchmark/gold/round13_dev.py stage5 --arms $A
python benchmark/gold/round13_dev.py score
cd "$HOME/MscProj_tg"
python benchmark/gold/tagger_prep.py --split dev2 stage4 --arms B0r "TO1+F7F8" $A
python benchmark/gold/tagger_prep.py --split dev2 stage5 --arms B0r "TO1+F7F8" $A
python benchmark/gold/tagger_prep.py --split dev2 gates --arms B0r "TO1+F7F8" $A
python benchmark/gold/merged_dev.py --arms B0r "TO1+F7F8" $A
echo "DONE conceal"
