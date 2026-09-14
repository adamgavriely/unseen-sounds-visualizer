#!/bin/bash
#SBATCH --job-name=persist_rank
#SBATCH --output=logs/persist_rank_%j.out
#SBATCH --error=logs/persist_rank_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# BEATs per-window scores cached for dev, test and DCASE (persistence + rank rule)
# (benchmark/persist_rank.py). Needs the gate-vote caches; ~30 min on an L4.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
python - <<'PY'
import sys, config
config.DEVICE = "cuda"
sys.argv = ["persist_rank", "--cache", "dev", "--cache", "test", "--cache", "dcase"]
from benchmark import persist_rank as K
K.main()
PY
echo DONE
