#!/bin/bash
#SBATCH --job-name=diffusion
#SBATCH --output=logs/diffusion_%j.out
#SBATCH --error=logs/diffusion_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamgavriely@gmail.com
# v2-b: run the FULL pipeline with the VLM gate + SDXL generation over benchmark
# clips, producing augmented videos for the qualitative comparison vs retrieval.
# Skips clips already rendered, so a requeued job continues where it stopped.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# how many clips per category to render (override: sbatch --export=N_PER_CAT=10 ...)
N_PER_CAT="${N_PER_CAT:-8}"

python - <<PY
import config
config.DEVICE = "cuda"
config.VIDEO_BACKEND = "vlm"        # best gate
config.GEN_BACKEND = "diffusion"    # SDXL instead of Openverse retrieval
config.GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
config.RENDER_MODE = "full"

from pathlib import Path
from src import pipeline

bench = Path("data/input/benchmark")
out = config.OUTPUT_DIR
out.mkdir(parents=True, exist_ok=True)
n_per = int("${N_PER_CAT}")

clips = []
for cat in ["unseen_ambient", "mixed", "seen_ambient", "no_ambient"]:
    ps = sorted(p for p in (bench / cat).glob("*")
                if p.suffix.lower() in (".mp4", ".webm", ".ogv"))
    clips += [(cat, p) for p in ps[:n_per]]
print(f"rendering {len(clips)} clips ({n_per}/category)", flush=True)

for i, (cat, p) in enumerate(clips, 1):
    dest = out / f"{p.stem}_augmented.mp4"
    if dest.exists():
        print(f"[{i}/{len(clips)}] skip {p.name} (done)", flush=True)
        continue
    print(f"[{i}/{len(clips)}] {cat}/{p.name}", flush=True)
    try:
        pipeline.run(p)
    except Exception as e:
        print(f"   ! failed: {type(e).__name__}: {e}", flush=True)
print("all done ->", out, flush=True)
PY
