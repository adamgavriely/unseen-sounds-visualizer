#!/bin/bash
#SBATCH --job-name=demos
#SBATCH --output=logs/demos_%j.out
#SBATCH --error=logs/demos_%j.err
# Queue on every partition whose GPUs this torch build actually supports, so the job
# lands wherever frees first. On a busy day most GPU partitions show "mixed-" -- the
# trailing dash means DRAINING, and a job queued there pends indefinitely. B200 is
# excluded on purpose: it is sm_100 and this torch supports up to sm_90, so a job there
# starts, fails on the first kernel, and wastes the slot.
#SBATCH --partition=L4-4h,L4-12h,L40s-4h,A100-4h
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
# Rendered as the PROPOSED system with the shipping pictogram generator -- and because the
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
config.GEN_BACKEND = "diffusion"    # the shipping generator: SDXL-Turbo pictograms
config.GATE_ENABLED = True          # the proposed system
config.TRANSCRIBE = True            # the gate asks whether people react to each sound
config.RENDER_MODE = "full"
config.SHOW_PROMPT = True           # demos are for diagnosis: show the prompt under each picture
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
