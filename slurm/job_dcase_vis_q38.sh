#!/bin/bash
#SBATCH --job-name=dcase_q38
#SBATCH --output=logs/dcase_q38_%j.out
#SBATCH --error=logs/dcase_q38_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# The visibility question against DCASE 2025 gold with Qwen3.8-27B, two arms
# (docs/prereg_qwen38_visibility.md): THINK=off (control) or THINK=on (primary).
#   THINK=off sbatch slurm/job_dcase_vis_q38.sh ; THINK=on sbatch slurm/job_dcase_vis_q38.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
N="${N:-300}"; THINK="${THINK:-off}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - "$N" "$THINK" <<'PY'
import config, sys
config.DEVICE = "cuda"
config.VLM_MODEL = "Qwen/Qwen3.8-27B"
config.VLM_THINKING = (sys.argv[2] == "on")
sys.argv = ["eval_dcase_visibility", "--n", sys.argv[1], "--out",
            f"benchmark/eval_dcase_visibility_q38_{'think' if config.VLM_THINKING else 'direct'}.json"]
from benchmark import eval_dcase_visibility as E
E.main()
PY
echo DONE
