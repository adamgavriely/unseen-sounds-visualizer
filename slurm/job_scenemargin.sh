#!/bin/bash
#SBATCH --job-name=scmarg
#SBATCH --output=logs/scmarg_%j.out
#SBATCH --error=logs/scmarg_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 60 SCENE-MARGIN (docs/prereg_round13_detector_push.md): video map, step 0 sanity (exit on STOP), step 1 415 gate (exit on
# STOP), then SHIP8+MD3 / +WW / +WW5 on old DEV (~/MscProj_r13) and DEV2 (~/MscProj_tg), merged DEV, floor rows, diff.
# Submit from ~/MscProj_tg.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
F=benchmark/gold/scenemargin.py
A="SHIP8+MD3 SHIP8+MD3+WW SHIP8+MD3+WW5"
export TG_ARMS="$A"
cd "$HOME/MscProj_tg"
python $F videos
python $F sanity
python $F gate
cd "$HOME/MscProj_r13"
python benchmark/gold/round13_dev.py stage4 --arms $A
python benchmark/gold/round13_dev.py stage5 --arms $A
python benchmark/gold/round13_dev.py score
cd "$HOME/MscProj_tg"
python benchmark/gold/tagger_prep.py --split dev2 stage4 --arms B0r "TO1+F7F8" $A
python benchmark/gold/tagger_prep.py --split dev2 stage5 --arms B0r "TO1+F7F8" $A
python benchmark/gold/tagger_prep.py --split dev2 gates --arms B0r "TO1+F7F8" $A
python benchmark/gold/merged_dev.py --arms B0r "TO1+F7F8" $A
for arm in $A; do for fl in none 0.4; do TG_ARMS="$arm" python benchmark/gold/floor_check_arm.py $fl "$arm"; done; done
python $F diff
echo "DONE scmarg"
