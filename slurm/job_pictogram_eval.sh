#!/bin/bash
#SBATCH --job-name=picteval
#SBATCH --output=logs/picteval_%j.out
#SBATCH --error=logs/picteval_%j.err
#SBATCH --partition=L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#
# Does the pictogram generator beat the retrieval baseline it replaced?
#
# The reported numbers were measured with Openverse retrieval, but retrieval was ruled
# out: a retrieved image only exists if somebody photographed that sound source and
# licensed it, which is a ceiling for arbitrary future sounds. The system now generates
# pictograms with SDXL-Turbo instead. This re-runs the identical 100 clips so the
# headline describes the configuration the system actually ships with, and so the
# quality cost (or gain) of that decision is measured rather than assumed.
#
# Rendering is its own phase: the generator and the describing VLM do not co-fit on a
# 24 GB card, which is how the first attempt at the main run died.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
LIMIT="${LIMIT:-100}"

echo "=== render (generator resident, no VLM) ==="
GEN=diffusion TAG=pictogram LIMIT="$LIMIT" PHASE=render bash slurm/job_protocol.sh
echo "=== describe (VLM resident, no generator) ==="
GEN=diffusion TAG=pictogram LIMIT="$LIMIT" PHASE=describe SKIP_RENDER=1 bash slurm/job_protocol.sh
echo "=== judge ==="
GEN=diffusion TAG=pictogram LIMIT="$LIMIT" PHASE=judge bash slurm/job_protocol.sh
echo "=== judge, human-grounded reference ==="
GEN=diffusion TAG=pictogram_grounded DESC_TAG=pictogram PHASE=judge GROUNDED=1 \
    bash slurm/job_protocol.sh

echo "=== pictogram generation vs the retrieval baseline it replaced ==="
python scripts/compare_runs.py protocol_results_retrieve.json protocol_results_pictogram.json || true
echo "=== pictogram generation vs the original SDXL-base run ==="
python scripts/compare_runs.py protocol_results.json protocol_results_pictogram.json || true
