#!/bin/bash
#SBATCH --job-name=na_dev
#SBATCH --output=logs/nameall_dev_%j.out
#SBATCH --error=logs/nameall_dev_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 66 NAME-ALL step 2 (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg with MODE=s4 | shard (SHARD=i/n)
# | score. s4: stage 4 of the arm on both parts; shard: stage 5 of the arm for every n-th clip (identical code / config, own
# ask-memo copy); score: complete check, round13 score, DEV2 gates, merged_dev (B0r, TO1+F7F8, D', arm), nameall_dev.py.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="SHIP8+MD3+WW5+SL"; A="SHIP8+MD3+WW5+SL+NA"
export TG_ARMS="$D $A"
case "$MODE" in
  s4)
    cd "$HOME/MscProj_r13"; python benchmark/gold/round13_dev.py stage4 --arms $A
    cd "$HOME/MscProj_tg"; python benchmark/gold/tagger_prep.py --split dev2 stage4 --arms $A ;;
  shard)
    export R13_SHARD="$SHARD"
    cd "$HOME/MscProj_r13"; python benchmark/gold/round13_dev.py stage5 --arms $A
    cd "$HOME/MscProj_tg"; python benchmark/gold/tagger_prep.py --split dev2 stage5 --arms $A ;;
  score)
    cd "$HOME/MscProj_r13"; python benchmark/gold/round13_dev.py stage5 --arms $A    # completes nothing if all shards finished
    python benchmark/gold/round13_dev.py score
    cd "$HOME/MscProj_tg"
    python benchmark/gold/tagger_prep.py --split dev2 stage5 --arms $A
    python benchmark/gold/tagger_prep.py --split dev2 gates --arms B0r "TO1+F7F8" $D $A
    python benchmark/gold/merged_dev.py --arms B0r "TO1+F7F8" $D $A
    python benchmark/gold/nameall_dev.py ;;
esac
echo "NAMEALL_DEV_DONE $MODE ${SHARD:-}"
