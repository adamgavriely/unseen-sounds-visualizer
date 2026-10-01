#!/bin/bash
#SBATCH --job-name=ctxarm
#SBATCH --output=logs/ctxarm_%j.out
#SBATCH --error=logs/ctxarm_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 59 CONTEXT (docs/prereg_round13_detector_push.md): step-1 gate (exit on STOP), override caches + replay, then arms
# SHIP8+MD3+CTX (rule a) and SHIP8+MD3+CTXB (variant b) beside SHIP8+MD3 on old DEV (~/MscProj_r13) and DEV2 (~/MscProj_tg),
# merged DEV, floor rows, changed pictures with Omni's context answers. Submit from ~/MscProj_tg after job_context_q.sh.
set -euo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
F=benchmark/gold/context.py
A="SHIP8+MD3 SHIP8+MD3+CTX SHIP8+MD3+CTXB"
export TG_ARMS="$A"
for d in "$HOME/MscProj_r13" "$HOME/MscProj_tg"; do grep -q "_ctx_bypass" "$d/src/stage4_audio_event_detection/__init__.py"; done
cd "$HOME/MscProj_tg"
python $F gate
python $F build
cd "$HOME/MscProj_r13"
python $F r13 stage4 --arms $A
python $F r13 stage5 --arms $A
python $F r13 score
cd "$HOME/MscProj_tg"
python $F tg stage4 --split dev2 --arms B0r "TO1+F7F8" $A
python $F tg stage5 --split dev2 --arms B0r "TO1+F7F8" $A
python $F tg gates --split dev2 --arms B0r "TO1+F7F8" $A
python $F merged --arms B0r "TO1+F7F8" $A
for arm in $A; do for fl in none 0.4; do python $F floor $fl "$arm"; done; done
python $F diff
echo "DONE ctxarm"
