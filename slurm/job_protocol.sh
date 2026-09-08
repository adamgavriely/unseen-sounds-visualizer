#!/bin/bash
#SBATCH --job-name=protocol
#SBATCH --output=logs/protocol_%j.out
#SBATCH --error=logs/protocol_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
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
# Results append after every clip (descriptions to protocol_descriptions.json, scores
# to protocol_results.json), so a requeued job resumes instead of restarting.

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
sys.argv = (["run_protocol", "--phase", "describe"]
            + (["--limit", "${LIMIT}"] if "${LIMIT}" else []))
from benchmark.run_protocol import main
main()
PY

# pass 2: judge only. Separate process so the VLM's memory is definitely released.
python - <<PY
import config
config.DEVICE = "cuda"
import sys
sys.argv = ["run_protocol", "--phase", "judge"]
from benchmark.run_protocol import main
main()
PY

echo "DONE -> benchmark/protocol_results.json"
