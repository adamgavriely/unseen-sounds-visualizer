#!/bin/bash
#SBATCH --job-name=hbox_logit
#SBATCH --output=logs/hbox_logit_%j.out
#SBATCH --error=logs/hbox_logit_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#
# Round 50L HUMAN-BOX-LOGIT steps 0-1 on gate-gold (docs/prereg_round13_detector_push.md). Submit from ~/MscProj.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/MscProj"
python benchmark/gold/humanbox_logit.py run 2>&1 | grep -v "Loading weights"
python benchmark/gold/humanbox_logit.py score
echo "HBOX_LOGIT_DONE rc=$?"
