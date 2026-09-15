#!/bin/bash
#SBATCH --job-name=audio_llm_smoke
#SBATCH --output=logs/audio_llm_smoke_%j.out
#SBATCH --error=logs/audio_llm_smoke_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Audio-LLM second opinion smoke test on 20 dev sounds (Qwen2-Audio)
# (benchmark/audio_llm_smoke.py). Needs the gate-vote caches; ~30 min on an L4.
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
sys.argv = ["audio_llm_smoke", "--full", "dev", "test"]
from benchmark import audio_llm_smoke as K
K.main()
PY
echo DONE
