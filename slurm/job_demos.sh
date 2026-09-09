#!/bin/bash
#SBATCH --job-name=demos
#SBATCH --output=logs/demos_%j.out
#SBATCH --error=logs/demos_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# DEMO RENDERS -- the artefact an advisor actually reacts to.
#
# Renders the clips chosen by benchmark/select_demo.py, which ranks by how VISIBLY
# SELECTIVE the augmentation is: a panel that is on from the first frame to the last
# looks like an image was appended to the video, not like a system deciding moment by
# moment. Clips are picked for mid-range coverage and several transitions.
#
# Rendered as the PROPOSED system with retrieval, because that is the configuration the
# ablation showed to be best (Section "Ablation" in the notes) -- and because the
# protocol runs overwrote data/output with whichever system rendered last, which was
# the caption baseline in minimal mode.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

N="${N:-8}"
python -m benchmark.select_demo --from-protocol --write "$N"

python - <<'PY'
import json, sys
from pathlib import Path
import config
config.DEVICE = "cuda"
config.VIDEO_BACKEND = "owlv2"
config.GEN_BACKEND = "retrieve"     # beats SDXL, and needs no GPU (see the ablation)
config.GATE_ENABLED = True          # the proposed system
config.RENDER_MODE = "full"
from src import pipeline

root = Path(".").resolve()
chosen = json.loads((root / "benchmark" / "demo_set.json").read_text(encoding="utf-8"))
out = root / "data" / "output" / "demos"
out.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR = out            # a Path: pipeline does OUTPUT_DIR / "<name>.mp4"
ok = 0
for i, rel in enumerate(chosen, 1):
    clip = root / "data" / "input" / "benchmark" / rel
    if not clip.exists():
        print(f"  ! missing {rel}", flush=True); continue
    print(f"[{i}/{len(chosen)}] {clip.name}", flush=True)
    try:
        pipeline.run(clip, work_root=root / "data" / "work" / "demos")
        ok += 1
    except Exception as e:
        print(f"  ! {clip.name}: {type(e).__name__}: {e}", flush=True)
print(f"rendered {ok}/{len(chosen)} demos", flush=True)
if ok == 0:
    sys.exit("no demos rendered")
PY

ls -la data/output/demos/*.mp4 2>/dev/null | tail -12
echo "DONE -> data/output/demos/"
