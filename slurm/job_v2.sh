#!/bin/bash
#SBATCH --job-name=protocol_v2
#SBATCH --output=logs/v2_%j.out
#SBATCH --error=logs/v2_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamgavriely@gmail.com
#
# THE MAIN EXPERIMENT, RE-RUN ON THE SHIPPING CONFIGURATION (2026-09-11).
#
# The numbers in the notes were produced by a pipeline that no longer exists: SDXL
# pictograms, a 30-concept visibility list, one dimmed slot per sound for the whole
# clip, and the transcript inside the depiction prompt. What ships now is PixArt-Sigma,
# a per-sound VLM visibility question, event-framed depictions made specific by the
# place, ontology-based deduplication, and speech as gate context. A thesis cannot
# report one and demonstrate the other.
#
# Everything is tagged "v2" so the earlier results stay intact for the comparison. The
# pilot and gate are skipped: the protocol itself has not changed and was validated on
# the first run. Same four passes as job_main.sh T3, same stamp-and-resume behaviour,
# so a 12 h kill resumes rather than restarts -- resubmit the same script.
#
#   usage:  sbatch slurm/job_v2.sh            (or LIMIT=20 sbatch ... for a smoke test)

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs benchmark/.chain
LIMIT="${LIMIT:-100}"
STAMP=benchmark/.chain/v2
mkdir -p "$STAMP"

step () {   # $1 stamp name, rest = command
    local name="$1"; shift
    if [ -f "$STAMP/$name" ]; then echo "=== $name: done earlier, skipping ==="; return; fi
    echo "=== $name ==="
    "$@"
    touch "$STAMP/$name"
}

step render   env GEN=diffusion LIMIT="$LIMIT" TAG=v2 PHASE=render bash slurm/job_protocol.sh
step describe env GEN=diffusion LIMIT="$LIMIT" TAG=v2 PHASE=describe SKIP_RENDER=1 bash slurm/job_protocol.sh
step judge    env GEN=diffusion LIMIT="$LIMIT" TAG=v2 PHASE=judge bash slurm/job_protocol.sh
step grounded env GEN=diffusion LIMIT="$LIMIT" TAG=v2_grounded DESC_TAG=v2 PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh

echo "DONE -> benchmark/protocol_results_v2.json (+ _v2_grounded.json)"
