#!/bin/bash
#SBATCH --job-name=explain_thr
#SBATCH --output=logs/explain_thr_%j.out
#SBATCH --error=logs/explain_thr_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 61c v-think SCENE-EXPLAIN second worker (reverse order, run only) (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg.
# Harness check -> step 0 sanity (10 specs incl. laundromat Train; exit 3 = >= 9 share one Q1 pair -> STOP) -> run 44 -> score.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS="SHIP8+MD3" EXPLAIN_VARIANT=think EXPLAIN_REVERSE=1
cd "$HOME/MscProj_tg"
SF=benchmark/gold/explain_sanity_think.json      # wait for the first worker's step 0; run only on GO
until [ -f "$SF" ]; do sleep 60; done
python -c "import json,sys; sys.exit(3 if json.load(open('$SF'))['stop'] else 0)" || { echo "SANITY STOP"; echo EXPLAIN_DONE; exit 0; }
python benchmark/gold/explain_screen_vr.py run
echo EXPLAIN_DONE
