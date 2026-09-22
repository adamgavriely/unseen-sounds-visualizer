#!/bin/bash
#SBATCH --job-name=protocol
#SBATCH --output=logs/protocol_%j.out
#SBATCH --error=logs/protocol_%j.err
#SBATCH --partition=L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamgavriely@gmail.com
#
# THE MAIN EXPERIMENT (proposal sec 6.1 + sec 7).
#
# For every labelled benchmark clip, render augmentations under each of the three
# systems, then score how much of the missing audio information each one actually
# conveys, using VLM-describe -> LLM-reference -> independent LLM judge.
#
#   proposed       cross-modal gate: depict only sounds whose source is NOT visible
#   blind_a2i      direct audio-to-image: depict every sound, ignore the video
#   audio_caption  text only, standing in for enriched subtitles
#
# Runs in TWO PASSES so the describing VLM (~16 GB) and the judging LLM (~15 GB) are
# never resident together -- they do not co-fit on a 24 GB card, and the large-memory
# partitions are often draining. Pass 1 caches descriptions, pass 2 scores them.
#
# Results append after every clip, so a requeued job resumes instead of restarting.
#
# Parameters, all via --export:
#   LIMIT=100     clips to run (stratified evenly across the four human tags)
#   GEN=diffusion|retrieve      how the augmentation image is produced
#   TAG=sdxl      keep this run's descriptions and results in their own files
#   PHASE=all|describe|judge    default all
#   JUDGE=<hf id> override the judge model (second judge for agreement)
#   DESC_TAG=...  judge a different run's cached descriptions

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
# Unbuffered, or a crash mid-model-load loses every print and leaves an empty log --
# which is exactly how the 11 GB GTX 1080 Ti failure on 'generic' presented itself.
export PYTHONUNBUFFERED=1
# FLUX is gated: read the token out of ~/.bashrc (sourcing it under set -e killed a job).
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'" ) || true
    export HF_TOKEN
fi

GEN="${GEN:-diffusion}"
LIMIT="${LIMIT:-}"
CLIP_DIR="${CLIP_DIR:-}"
PSED_BAR="${PSED_BAR:-}"
REF_TAG="${REF_TAG:-}"       # v4ab2: override the PSED bar (benchmark/psed_f1_bar.json)
TAG="${TAG:-}"
PHASE="${PHASE:-all}"
JUDGE="${JUDGE:-}"
DESC_TAG="${DESC_TAG:-}"
RESCORE="${RESCORE:-}"
INDEP="${INDEP:-}"
GROUNDED="${GROUNDED:-}"
SKIP_RENDER="${SKIP_RENDER:-}"
WORK_TAG="${WORK_TAG:-}"
SYSTEMS="${SYSTEMS:-}"   # e.g. "proposed" or "blind_a2i audio_caption": one system per job when sharding
V4="${V4:-}"          # v4 stages to apply (docs/prereg_v4.md), e.g. "4", "45", "23456"
echo "[cfg] phase=$PHASE limit=${LIMIT:-all} gen=$GEN tag=${TAG:-<main>} judge=${JUDGE:-<config>}"

# The judge pass is text-only and fits on a small card; the describe pass is not.
# Refuse to start on a card too small rather than dying silently mid-load.
# SMALL_OK=1: a render with no VLM and no diffusion (audio_caption with GEN=placeholder) fits any card
if [ "$PHASE" != "judge" ] && [ -z "${SMALL_OK:-}" ]; then
python - <<'PY'
import torch, sys
if torch.cuda.is_available():
    gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"[check] GPU {torch.cuda.get_device_name(0)}  {gb:.0f} GB", flush=True)
    if gb < 20:
        sys.exit(f"GPU too small: Qwen2.5-VL 7B needs ~16 GB, this card has {gb:.0f} GB. "
                 "Use --partition=L4-4h (23 GB), A100-4h or L40s-4h.")
PY
fi

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

run_phase () {   # $1 = describe | judge
python - "$1" <<PY
import sys
phase = sys.argv[1]
import config
config.DEVICE = "cuda"
config.VIDEO_BACKEND = "owlv2"     # strongest visibility backend (see sec:findings)
config.GEN_BACKEND = "${GEN}"
if "${V4}":
    print("[v4]", config.use_v4("${V4}"), flush=True)
if "${PSED_BAR}":                      # v4ab2: PSED's own operating point (benchmark/psed_f1_bar.json)
    config.PSED_BAR = float("${PSED_BAR}"); print("[v4] PSED_BAR", config.PSED_BAR, flush=True)
# GEN_MODEL deliberately NOT overridden: this job used to pin SDXL-base here, which
# would have evaluated a generator the pipeline no longer ships. config.py holds the
# shipping one (PixArt-Sigma), and the point of the run is to score what ships.
config.TRANSCRIBE = True           # the reference-builder needs the transcript
argv = ["run_protocol", "--phase", phase]
if "${LIMIT}":    argv += ["--limit", "${LIMIT}"]
if "${CLIP_DIR}": argv += ["--clip-dir", "${CLIP_DIR}"]
if "${SYSTEMS}":  argv += ["--systems"] + "${SYSTEMS}".split()
if "${TAG}":      argv += ["--tag", "${TAG}"]
if "${JUDGE}":    argv += ["--judge", "${JUDGE}"]
if "${DESC_TAG}": argv += ["--desc-tag", "${DESC_TAG}"]
if "${RESCORE}" and phase == "judge": argv += ["--rescore"]
if "${GROUNDED}" and phase == "judge": argv += ["--grounded"]
if "${REF_TAG}" and phase == "judge":  argv += ["--ref-tag", "${REF_TAG}"]
if "${INDEP}" and phase == "judge":    argv += ["--independent"]
if "${SKIP_RENDER}" and phase == "describe": argv += ["--skip-render"]
if "${WORK_TAG}" and phase == "describe":    argv += ["--work-tag", "${WORK_TAG}"]
sys.argv = argv
print("[argv]", " ".join(argv), flush=True)
from benchmark.run_protocol import main
main()
PY
}

# Separate processes, so the describer's memory is definitely released before the
# judge is loaded -- unload_vlm() frees the weights but not always the allocator.
# One large model resident per process: SDXL renders, then the VLM describes, then
# the judge scores. SDXL and the VLM together overflow a 24 GB card.
case "$PHASE" in
  render)   run_phase render ;;
  describe) run_phase describe ;;
  judge)    run_phase judge ;;
  reference) run_phase reference ;;
  all)      run_phase describe; run_phase judge ;;
esac

echo "DONE -> benchmark/protocol_results${TAG:+_$TAG}.json"
