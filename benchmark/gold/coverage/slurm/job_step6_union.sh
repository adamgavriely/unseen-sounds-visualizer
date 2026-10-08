#!/bin/bash
#SBATCH --job-name=s6un
#SBATCH --output=logs/step6_union_%j.out
#SBATCH --error=logs/step6_union_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Step 6 B1 union arms (benchmark/gold/coverage/PREREG_step6_detector_retry_and_extra_ear.md), DEV + dev2 parts. Submit from ~/MscProj_tg with
# MODE=s4 | shard (SHARD=i/n) | fin. Same layout as the earlier merged-DEV arm jobs (DEV part from ~/MscProj_r13).
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
W="$HOME/MscProj_tg/benchmark/gold/coverage/step6_union_arms.py"
A="SHIP8+MD3+WW5+SL+UNO3 SHIP8+MD3+WW5+SL+UNO5 SHIP8+MD3+WW5+SL+UNF3 SHIP8+MD3+WW5+SL+UNF5"
case "$MODE" in
  s4)
    cd "$HOME/MscProj_r13"; python "$W" dev stage4 --arms $A
    cd "$HOME/MscProj_tg"; python "$W" dev2 stage4 --split dev2 --arms $A ;;
  shard)
    export R13_SHARD="$SHARD"
    cd "$HOME/MscProj_r13"; python "$W" dev stage5 --arms $A
    cd "$HOME/MscProj_tg"; python "$W" dev2 stage5 --split dev2 --arms $A ;;
  fin)
    cd "$HOME/MscProj_r13"; python "$W" dev stage5 --arms $A
    cd "$HOME/MscProj_tg"; python "$W" dev2 stage5 --split dev2 --arms $A ;;
esac
echo "STEP6_UNION_DONE $MODE ${SHARD:-}"
