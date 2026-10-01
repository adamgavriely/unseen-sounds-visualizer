#!/bin/bash
#SBATCH --job-name=logit_scene
#SBATCH --output=logs/logit_scene_%j.out
#SBATCH --error=logs/logit_scene_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:00:00
#
# Round 62 secondary SCENE-RECHECK of D (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/MscProj_tg"
python benchmark/gold/logit_gate.py scene 2>&1 | grep -v "Warning\|Loading weights"
echo LOGIT_SCENE_DONE
