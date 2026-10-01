#!/bin/bash
#SBATCH --job-name=explain_d
#SBATCH --output=logs/explain_d_%j.out
#SBATCH --error=logs/explain_d_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 61d SCENE-EXPLAIN (docs/prereg_round13_detector_push.md). Submit from ~/MscProj_tg. Spec lists already written
# (explain_d_specs_{B,D}.json). Main asks (msproj) -> gemma L1n (judge venv) -> score B/D x stack/replace (main), gemma (report).
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd "$HOME/MscProj_tg"
EXPLAIN_VARIANT=main python benchmark/gold/explain_d.py ask || { echo "MAIN ASK failed"; }
EXPLAIN_VARIANT=gemma "$HOME/venvs/judge/bin/python" benchmark/gold/explain_d.py ask || echo "GEMMA ASK failed"
for V in main gemma; do for A in SHIP8+MD3+WW5 SHIP8+MD3; do for F in replace stack; do
  EXPLAIN_VARIANT=$V ARM=$A TG_ARMS=$A FORM=$F python benchmark/gold/explain_d.py score 2>&1 | grep -v Warning
done; done; done
echo EXPLAIN_D_DONE
