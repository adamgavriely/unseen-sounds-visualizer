#!/bin/bash
#SBATCH --job-name=main_run
#SBATCH --output=logs/main_%j.out
#SBATCH --error=logs/main_%j.err
#SBATCH --partition=L4-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamgavriely@gmail.com
#
# THE EXPERIMENT, unattended and from a clean state:
#
#   P  pilot          12 clips x 3 systems with retrieval -- cheap, and its only job is
#                     to show the protocol works before the expensive run starts
#   G  gate           refuse to continue if the pilot shows a protocol that is not
#                     measuring anything (scripts/check_pilot.py)
#   T3 main run       100 clips x 3 systems with SDXL, the table the thesis turns on
#
# The pilot writes to its own tagged caches, so the two runs never share a description
# cache or a results file -- they use different image sources and mixing them would
# quietly corrupt the headline result.
#
# Every stage is stamped and every cache is written per clip, so this script is
# idempotent: after a time-limit kill it resumes instead of restarting, which is what
# lets the chain use a continuation job.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

STAMP=benchmark/.chain
mkdir -p "$STAMP" logs
LIMIT="${LIMIT:-100}"
PILOT="${PILOT:-12}"

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# ---------------------------------------------------------------- P: pilot
if [ ! -f "$STAMP/pilot_done" ]; then
    echo "=== P: pilot, $PILOT clips x 3 systems (retrieval) ==="
    GEN=retrieve LIMIT="$PILOT" TAG=pilot bash slurm/job_protocol.sh
    touch "$STAMP/pilot_done"
fi

# ---------------------------------------------------------------- G: gate
# Do not spend ten GPU-hours on a protocol that is not measuring anything. The checks
# a person would make by eye on the pilot are made here instead, and a failure stops
# the chain -- the dependent jobs carry kill-on-invalid-dep, so they are cancelled
# rather than left pending for days.
if [ ! -f "$STAMP/gate_passed" ]; then
    echo "=== G: gate -- is the pilot sound enough to scale up? ==="
    python scripts/check_pilot.py protocol_results_pilot.json
    touch "$STAMP/gate_passed"
fi

# ---------------------------------------------------------------- T3: main run
echo "=== T3a: render $LIMIT clips x 3 systems with SDXL (no VLM resident) ==="
GEN=diffusion LIMIT="$LIMIT" PHASE=render bash slurm/job_protocol.sh

echo "=== T3b: describe the rendered augmentations (no SDXL resident) ==="
GEN=diffusion LIMIT="$LIMIT" PHASE=describe SKIP_RENDER=1 bash slurm/job_protocol.sh

echo "=== T3c: judge ==="
GEN=diffusion LIMIT="$LIMIT" PHASE=judge bash slurm/job_protocol.sh

# A second scoring of the SAME descriptions against the human-corrected reference. The
# model-derived reference reports a sound as missing even when the annotator recorded
# its source as plainly on screen, which scores correct silence at 0 -- on 128 of the
# 274 labelled clips. Judging is cheap and needs no vision, so both numbers come out of
# the one expensive describe pass and the report can state each with its assumption.
echo "=== T3d: re-score against the human-grounded reference ==="
GEN=diffusion LIMIT="$LIMIT" PHASE=judge GROUNDED=1 TAG=grounded DESC_TAG=""     bash slurm/job_protocol.sh

touch "$STAMP/main_done"
echo "DONE -> benchmark/protocol_results.json (+ _grounded.json)"
