#!/bin/bash
#SBATCH --job-name=eval_vlm
#SBATCH --output=logs/eval_vlm_%j.out
#SBATCH --error=logs/eval_vlm_%j.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamgavriely@gmail.com
# THE HEADLINE EXPERIMENT: re-score the benchmark with the Qwen2.5-VL gate and
# compare against the CLIP gate on identical clips and ground truth.
# Signals are cached per clip, so a requeue resumes instead of restarting.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 1. CLIP-gate baseline (cheap; rescores from the cache if it was synced)
echo "===== CLIP gate (baseline) ====="
VIDEO_BACKEND=clip python - <<'PY'
import config; config.VIDEO_BACKEND = "clip"; config.DEVICE = "cuda"
import importlib, benchmark.evaluate as ev; importlib.reload(ev); ev.main()
PY

# 2. VLM gate
echo "===== Qwen2.5-VL gate ====="
python - <<'PY'
import config; config.VIDEO_BACKEND = "vlm"; config.DEVICE = "cuda"
import importlib, benchmark.evaluate as ev; importlib.reload(ev); ev.main()
PY

# 3. side-by-side table
echo "===== COMPARISON ====="
python - <<'PY'
import json
from pathlib import Path
rows = []
for name, f in [("CLIP gate (v2-a)", "benchmark/eval_results.json"),
                ("Qwen2.5-VL gate (v2-b)", "benchmark/eval_results_vlm.json")]:
    p = Path(f)
    if not p.exists():
        continue
    m = json.loads(p.read_text())["main"]
    rows.append((name, m))
print(f"{'gate':26}{'n':>5}{'acc':>9}{'prec':>9}{'rec':>9}{'F1':>9}")
for name, m in rows:
    print(f"{name:26}{m['n_clips']:>5}{m['accuracy']:>8.1%}{m['precision']:>9.1%}"
          f"{m['recall']:>9.1%}{m['f1']:>9.1%}")
if len(rows) == 2:
    d = rows[1][1]["f1"] - rows[0][1]["f1"]
    print(f"\nVLM gate F1 delta: {d:+.1%}")
PY
echo "DONE -> benchmark/eval_results.json + eval_results_vlm.json"
