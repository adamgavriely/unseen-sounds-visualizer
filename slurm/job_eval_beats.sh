#!/bin/bash
#SBATCH --job-name=eval_beats
#SBATCH --output=logs/eval_beats_%j.out
#SBATCH --error=logs/eval_beats_%j.err
#SBATCH --partition=L4-4h,L40s-4h,A100-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Threshold sweep for the BEATs detector against Adam's 274 hand tags.
#
# DISPLAY_THRESHOLD = 0.30 was set from eight demo clips (every real sound at 0.30+,
# every phantom at 0.28-). This scores the gate over every tagged clip at a grid of
# thresholds and writes benchmark/eval_results_owlv2_beats.json. The cache is per
# detector, so the PANNs numbers are untouched and the two can be compared.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python - <<'PY'
import config
config.DEVICE = "cuda"
config.VIDEO_BACKEND = "owlv2"
config.AED_MODEL = "beats"
import sys
sys.argv = ["evaluate"]
from benchmark import evaluate
evaluate.main()
PY
echo "DONE -> benchmark/eval_results_owlv2_beats.json"
