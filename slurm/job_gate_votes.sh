#!/bin/bash
#SBATCH --job-name=gate_votes
#SBATCH --output=logs/gate_votes_%j.out
#SBATCH --error=logs/gate_votes_%j.err
#SBATCH --partition=L4-12h,L40s-12h,A100-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# Cache the gate's raw visibility votes for one split (benchmark/gate_dev_sweep.py
# --cache). SPLIT=dev (174 clips) or SPLIT=test (100 clips); run both in parallel,
# then sweep on dev only (CPU). Resumable: one JSON per clip.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
SPLIT="${SPLIT:-dev}"
python - "$SPLIT" <<'PY'
import sys, config
config.DEVICE = "cuda"
sys.argv = ["gate_dev_sweep", "--cache", "--split", sys.argv[1]]
from benchmark import gate_dev_sweep as G
G.main()
PY
echo DONE
