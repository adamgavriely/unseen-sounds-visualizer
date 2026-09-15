#!/bin/bash
#SBATCH --job-name=demos_v3
#SBATCH --output=logs/demos_v3_%j.out
#SBATCH --error=logs/demos_v3_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# DEMOS ON THE SHIPPING CONFIGURATION (v3, 2026-09-15).
#
# The demos under data/output/demos predate the BEATs detector, so they show a system the
# report no longer describes. This renders the three clips of the report's Figure 2 --
# a correct silence, a wrong silence, a correct picture -- so the demos, the figure and
# the executive summary tell one story (Fable: option A). Two passes per clip:
#
#   data/output/demos_v3/         clean, the deliverable
#   data/output/demos_v3_debug/   the phrase and every raw detection with the gate's
#                                 verdict under each picture, so an empty panel reads as
#                                 "detector saw traffic, gate declined", not "nothing ran"
#
# Same config as the protocol's "proposed" system (cuda, OWLv2, FLUX, gate on, speech
# context on, full render). Nothing here is tuned; the clips were fixed by the figure.
#
#   usage:  sbatch slurm/job_demos_v3.sh

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'" ) || true
    export HF_TOKEN
fi

for MODE in clean debug; do
DEMO_MODE="$MODE" python - <<'PY'
import json, os, sys
from pathlib import Path
import config
config.DEVICE = "cuda"
config.VIDEO_BACKEND = "owlv2"
config.GEN_BACKEND = "diffusion"
config.GATE_ENABLED = True
config.DEPICTION_REASONING = True
config.VLM_VISIBILITY = True
config.SPEECH_CONTEXT = True
config.TRANSCRIBE = True
config.RENDER_MODE = "full"
debug = os.environ["DEMO_MODE"] == "debug"
config.SHOW_PROMPT = debug
config.SHOW_DEBUG_SOUNDS = debug
from src import pipeline
from benchmark.gate_dev_sweep import _find_clip   # clips sit in per-scenario folders

root = Path(".").resolve()
chosen = json.loads((root / "benchmark" / "demo_set_v3.json").read_text(encoding="utf-8"))
out = root / "data" / "output" / ("demos_v3_debug" if debug else "demos_v3")
out.mkdir(parents=True, exist_ok=True)
config.OUTPUT_DIR = out
ok = 0
for i, rel in enumerate(chosen, 1):
    clip = _find_clip(rel)
    if clip is None:
        print(f"  ! missing {rel}", flush=True); continue
    print(f"[{i}/{len(chosen)}] {clip.name} ({os.environ['DEMO_MODE']})", flush=True)
    try:
        pipeline.run(clip, work_root=root / "data" / "work" / out.name)
        ok += 1
    except Exception as e:
        print(f"  ! {clip.name}: {type(e).__name__}: {e}", flush=True)
print(f"rendered {ok}/{len(chosen)} demos ({os.environ['DEMO_MODE']})", flush=True)
if ok == 0:
    sys.exit("no demos rendered")
PY
done

ls -la data/output/demos_v3*/*.mp4 2>/dev/null
echo "DONE"
