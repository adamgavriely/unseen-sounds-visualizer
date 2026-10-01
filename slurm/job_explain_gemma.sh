#!/bin/bash
#SBATCH --job-name=explain_gm
#SBATCH --output=logs/explain_gm_%j.out
#SBATCH --error=logs/explain_gm_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 61c v-gemma SCENE-EXPLAIN (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg.
# Specs in msproj -> sanity + run with gemma-4-31B-it in the judge venv (Round 24 path) -> score in msproj.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS="SHIP8+MD3" EXPLAIN_VARIANT=gemma
cd "$HOME/MscProj_tg"
JP="$HOME/venvs/judge/bin/python"
[ -f benchmark/gold/explain_specs.json ] || python benchmark/gold/explain_screen_v.py specs || { echo "HARNESS CHECK failed"; exit 1; }
"$JP" benchmark/gold/explain_screen_v.py sanity || { echo "SANITY STOP or error (exit $?)"; echo EXPLAIN_DONE; exit 0; }
"$JP" benchmark/gold/explain_screen_v.py run && python benchmark/gold/explain_screen_v.py score; python benchmark/gold/explain_screen_v.py score_replace
echo EXPLAIN_DONE
