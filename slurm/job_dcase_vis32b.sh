#!/bin/bash
#SBATCH --job-name=dcase_vis32b
#SBATCH --output=logs/dcase_vis32b_%j.out
#SBATCH --error=logs/dcase_vis32b_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#
# The visibility question against DCASE 2025 Task 3's onscreen/offscreen gold
# (benchmark/eval_dcase_visibility.py). No detector in the loop.
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
# the same 258 events, the same three-vote question, a 32B model instead of the 7B:
# does a larger VLM move the 35% on-screen recall? Declared bar (Fable, 2026-09-15):
# on-screen recall >= 50% AND off-screen events wrongly silenced up by <= 2 points.
config.VLM_MODEL = "Qwen/Qwen2.5-VL-32B-Instruct"
sys.argv = ["eval_dcase_visibility", "--n", sys.argv[1], "--out",
            "benchmark/eval_dcase_visibility_32b.json"]
from benchmark import eval_dcase_visibility as E
E.main()
PY
echo DONE
