#!/bin/bash
#SBATCH --job-name=abl_gen
#SBATCH --output=logs/abl_gen_%j.out
#SBATCH --error=logs/abl_gen_%j.err
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#
# ABLATION: does GENERATING the augmentation image beat RETRIEVING one?
# (docs/PLAN.md, Day 7)
#
# The proposal names diffusion (SDXL) as the image source, but that choice has never
# been tested against the much cheaper alternative of retrieving a stock image for the
# same sound label. If retrieval ties, the thesis should say so plainly: it removes a
# GPU dependency from the whole system, which matters for an accessibility tool meant
# to run on ordinary hardware.
#
# The main run uses SDXL. This job runs the identical clips with retrieval, tagged so
# it gets its own description cache -- otherwise the cache, keyed by (clip, system),
# would hand the retrieval arm the diffusion arm's descriptions and the ablation would
# silently compare a run against itself.

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || \
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda activate msproj
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1

LIMIT="${LIMIT:-100}"
GEN=retrieve TAG=retrieve LIMIT="$LIMIT" bash slurm/job_protocol.sh

python scripts/compare_runs.py protocol_results.json protocol_results_retrieve.json
