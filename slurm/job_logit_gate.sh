#!/bin/bash
#SBATCH --job-name=logit_gate
#SBATCH --output=logs/logit_gate_%j.out
#SBATCH --error=logs/logit_gate_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 62 LOGIT-GATE (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg.
# sanity (harness check, exit 4 = STOP) -> gate-gold d -> step 0/1 (exit 0 only on step-1 PASS) -> DEV arm D -> score.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 ARM="SHIP8+MD3+WW5" TG_ARMS="SHIP8+MD3+WW5"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/MscProj"
python benchmark/gold/logit_gate.py sanity || { echo "LOGIT_GATE STOP: harness"; exit 0; }
python benchmark/gold/logit_gate.py gold || { echo "GOLD RUN failed"; exit 1; }
python benchmark/gold/logit_gate.py score_gold; rc=$?
cp benchmark/gold/logit_gate_gold.json "$HOME/MscProj_tg/benchmark/gold/" 2>/dev/null
if [ "$rc" != "0" ]; then echo "LOGIT_GATE step 1 not passed (rc $rc): step 2 not run"; echo LOGIT_GATE_DONE; exit 0; fi
cd "$HOME/MscProj_tg"
python benchmark/gold/logit_gate.py dev || { echo "DEV RUN failed"; exit 1; }
python benchmark/gold/logit_gate.py score_dev 2>&1 | grep -v Warning
echo LOGIT_GATE_DONE
