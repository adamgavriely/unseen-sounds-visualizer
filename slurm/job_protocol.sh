#!/bin/bash
#SBATCH --job-name=protocol
#SBATCH --output=logs/protocol_%j.out
#SBATCH --error=logs/protocol_%j.err
#SBATCH --partition=H200-12h
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
# Results append to benchmark/protocol_results.json after every clip, so a requeued
# job resumes instead of restarting.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Generation backend for the augmentation images. SDXL is the proposal's candidate;
# set GEN=retrieve to fall back to Openverse retrieval if the GPU is busy.
GEN="${GEN:-diffusion}"
LIMIT="${LIMIT:-}"          # e.g. sbatch --export=LIMIT=30 for a quick pass

python - <<PY
import config
config.DEVICE = "cuda"
config.VIDEO_BACKEND = "owlv2"     # strongest visibility backend (see sec:findings)
config.GEN_BACKEND = "${GEN}"
config.GEN_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
import sys
sys.argv = ["run_protocol"] + (["--limit", "${LIMIT}"] if "${LIMIT}" else [])
from benchmark.run_protocol import main
main()
PY

echo "DONE -> benchmark/protocol_results.json"
