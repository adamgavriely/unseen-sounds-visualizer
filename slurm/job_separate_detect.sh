#!/bin/bash
#SBATCH --job-name=sep_detect
#SBATCH --output=logs/sep_detect_%j.out
#SBATCH --error=logs/sep_detect_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Separate first, detect second (benchmark/separate_detect.py): Demucs htdemucs "other"
# stem -> BEATs windows cached for DCASE and dev, then the declared evaluation. Small
# assessment only; no test-set contact. ~1-2 h on an L4.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
python -c "import demucs; print('demucs', demucs.__version__)"
python - <<'PY'
import sys, config
config.DEVICE = "cuda"
sys.argv = ["separate_detect", "--cache", "dcase", "--cache", "dev", "--eval"]
from benchmark import separate_detect as S
S.main()
PY
echo DONE
