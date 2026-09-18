#!/bin/bash
#SBATCH --job-name=sweep_q38
#SBATCH --output=logs/sweep_q38_%j.out
#SBATCH --error=logs/sweep_q38_%j.err
#SBATCH --partition=cpu192G-48h,generic
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
# The v4 gate's silence rule, set once on the DEV split from the Qwen3.8 votes with the
# rubric-enforced asymmetry (miss -4, redundant -2). CPU only.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
python - <<'PY'
import sys
from pathlib import Path
from benchmark import gate_dev_sweep as G
G.CACHE_DIR = Path(str(G.CACHE_DIR) + "_q38")
G.SETTING = G.SETTING.with_name("gate_setting_q38.json")
G.COST_REDUNDANT = 2.0
sys.argv = ["gate_dev_sweep", "--sweep"]
G.main()
PY
echo DONE
