#!/bin/bash
#SBATCH --job-name=votes_q38
#SBATCH --output=logs/votes_q38_%j.out
#SBATCH --error=logs/votes_q38_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# The gate's raw visibility votes with the v4 VLM (Qwen3.8-27B, thinking off) on one split,
# into their own cache (benchmark/gate_votes_q38/<split>), so the silence rule can be set
# once on the DEV split for v4 exactly as it was for v3 (benchmark/gate_dev_sweep.py; the
# designed tuning procedure, never on test). Resumable: one JSON per clip.
#   SPLIT=dev sbatch slurm/job_gate_votes_q38.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
SPLIT="${SPLIT:-dev}"
python - "$SPLIT" <<'PY'
import sys, config
config.DEVICE = "cuda"
config.use_v4("5")
from benchmark import gate_dev_sweep as G
from pathlib import Path
G.CACHE_DIR = Path(str(G.CACHE_DIR) + "_q38")
G.SETTING = G.SETTING.with_name("gate_setting_q38.json")
sys.argv = ["gate_dev_sweep", "--cache", "--split", sys.argv[1]]
G.main()
PY
echo DONE
