#!/bin/bash
#SBATCH --job-name=scenelogit
#SBATCH --output=logs/scenelogit_%j.out
#SBATCH --error=logs/scenelogit_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 60L SCENE-LOGIT (docs/prereg_round13_detector_push.md): B, D, SL on old DEV (~/MscProj_r13) and DEV2 (~/MscProj_tg),
# merged DEV, pass vs B, diff. Recipe = slurm/job_scenemargin.sh steps 2. Submit from ~/MscProj_tg.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
A="SHIP8+MD3 SHIP8+MD3+WW5 SHIP8+MD3+WW5+SL"
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
python benchmark/gold/scenelogit.py
echo "DONE scenelogit"
