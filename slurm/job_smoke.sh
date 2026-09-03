#!/bin/bash
#SBATCH --job-name=smoke
#SBATCH --output=logs/smoke_%j.out
#SBATCH --error=logs/smoke_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
# Smoke test: prove GPU + env + data are all in place before spending real GPU hours.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj

echo "== node/GPU =="
hostname; nvidia-smi --query-gpu=name,memory.total --format=csv

echo "== torch =="
python -c "import torch;print('torch',torch.__version__,'cuda',torch.cuda.is_available(),torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

echo "== ffmpeg =="
ffmpeg -version | head -1

echo "== benchmark data =="
python - <<'PY'
from pathlib import Path
b = Path("data/input/benchmark")
tot = 0
for f in ["unseen_ambient", "seen_ambient", "mixed", "no_ambient", "unsorted"]:
    n = len([p for p in (b / f).glob("*") if p.suffix.lower() in (".mp4", ".webm", ".ogv")])
    tot += n
    print(f"  {f:16} {n}")
print(f"  {'TOTAL':16} {tot}")
assert tot > 0, "no benchmark clips found -- run slurm/sync_data.sh from your PC"
PY

echo "== pipeline end-to-end on one clip (GPU) =="
python - <<'PY'
import config; config.DEVICE = "cuda"
from pathlib import Path
from src import pipeline
clip = next(iter(sorted(Path("data/input/benchmark/unseen_ambient").glob("*.mp4"))), None) \
       or Path("data/input/clip.mp4")
r = pipeline.run(clip)
print("OK ->", r.output_path)
PY

echo "SMOKE TEST PASSED"
