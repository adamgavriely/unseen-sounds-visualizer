#!/bin/bash
#SBATCH --job-name=av_ref
#SBATCH --output=logs/av_ref_%j.out
#SBATCH --error=logs/av_ref_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#
# With-sound / without-sound reference pilot (benchmark/av_reference_pilot.py,
# docs/prereg_av_reference.md). Dev split only. Tries MiniCPM-o 2.6 first (not a Qwen
# vision encoder); if it cannot load or answer on a 3-clip smoke, falls back to
# Qwen2.5-Omni-7B. Models must already be in the HF cache (login node download).
#   usage: sbatch slurm/job_av_reference.sh          BACKEND=omni sbatch ... to force
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

run_py () {   # $1 backend, rest = args
python - "$@" <<'PY'
import sys, config
config.DEVICE = "cuda"
args = sys.argv[1:]
sys.argv = ["av_reference_pilot", "--backend"] + args
from benchmark import av_reference_pilot as A
A.main()
PY
}

BACKEND="${BACKEND:-}"
if [ -z "$BACKEND" ]; then
    echo "=== smoke: minicpm ==="
    if run_py minicpm --limit 3 && [ "$(ls benchmark/av_reference_pilot/minicpm/*.json 2>/dev/null | wc -l)" -ge 2 ]; then
        BACKEND=minicpm
    else
        echo "=== minicpm failed; smoke: omni ==="
        if run_py omni --limit 3 && [ "$(ls benchmark/av_reference_pilot/omni/*.json 2>/dev/null | wc -l)" -ge 2 ]; then
            BACKEND=omni
        else
            echo "both backends failed"; exit 1
        fi
    fi
fi
echo "=== full dev run: $BACKEND ==="
run_py "$BACKEND"
echo "=== eval: $BACKEND ==="
run_py "$BACKEND" --eval
echo DONE
