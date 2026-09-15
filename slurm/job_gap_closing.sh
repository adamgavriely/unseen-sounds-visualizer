#!/bin/bash
#SBATCH --job-name=gap_close
#SBATCH --output=logs/gap_close_%j.out
#SBATCH --error=logs/gap_close_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#
# Gap-closing kill-criterion pilot (benchmark/gap_closing_pilot.py, docs/prereg_gap_closing.md):
# 60 dev clips x 3 runs of Qwen2.5-Omni (sound / mute / mute at shifted frames), then eval.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
python - <<'PY'
import sys, config
config.DEVICE = "cuda"
sys.argv = ["gap_closing_pilot"]
from benchmark import gap_closing_pilot as G
G.main()
PY
python -m benchmark.gap_closing_pilot --eval
echo DONE
