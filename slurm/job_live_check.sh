#!/bin/bash
#SBATCH --job-name=livechk
#SBATCH --output=logs/livechk_%j.out
#SBATCH --error=logs/livechk_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:00:00
#
# On-the-spot listener inputs (src/listener_prep.py) for one DEV clip; its answers must equal the shipcheck run's.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
python - <<PY
from pathlib import Path
import json
from src.listener_prep import ensure_listener_inputs
v = Path("data/input/shipcheck_src/as_explosion_XJ8lc3I6.mp4")
s = ensure_listener_inputs(v)
print("split", s)
g = Path("benchmark/gold")
for k in ("", "_v", "_afn"):
    a = [x for x in json.load(open(g / f"shipcheck_listener{k}.json"))["items"] if x["clip"] == v.stem]
    b = json.load(open(g / f"{s}_listener{k}.json"))["items"]
    key = lambda x: (x["pool"], x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2))
    A = {key(x): (x.get("accept"), x.get("score")) for x in a}; B = {key(x): (x.get("accept"), x.get("score")) for x in b}
    print(k or "yn", "items", len(A), len(B), "same keys", A.keys() == B.keys(), "same answers", A == B)
PY
echo "DONE livechk"
