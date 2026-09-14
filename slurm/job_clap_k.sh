#!/bin/bash
#SBATCH --job-name=clap_k
#SBATCH --output=logs/clap_k_%j.out
#SBATCH --error=logs/clap_k_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# CLAP second opinion: calibrate k on DCASE gold, then rank every dev/test detection
# (benchmark/calibrate_clap_k.py). Needs the gate-vote caches; ~30 min on an L4.
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
sys.argv = ["calibrate_clap_k", "--dcase", "--ranks", "dev", "--ranks", "test"]
from benchmark import calibrate_clap_k as K
K.main()
PY
echo DONE
