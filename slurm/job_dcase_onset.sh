#!/bin/bash
#SBATCH --job-name=dcase_onset
#SBATCH --output=logs/dcase_onset_%j.out
#SBATCH --error=logs/dcase_onset_%j.err
#SBATCH --partition=L4-4h,L40s-4h,A100-4h,L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#
# Onset error against DCASE 2025 Task 3 gold starts, three stamping methods
# (benchmark/eval_dcase_onset.py). Detector only; no VLM.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
N="${N:-300}"
python - "$N" <<'PY'
import config, sys
config.DEVICE = "cuda"
sys.argv = ["eval_dcase_onset", "--n", sys.argv[1]]
from benchmark import eval_dcase_onset as E
E.main()
PY
echo DONE
