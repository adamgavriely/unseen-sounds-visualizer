#!/bin/bash
#SBATCH --job-name=rejudge_gemma
#SBATCH --output=logs/rejudge_gemma_%j.out
#SBATCH --error=logs/rejudge_gemma_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# The v4 judge (Gemma-4-31B-it, docs/prereg_v4.md) over every cached description set, once
# the weights are on disk (hf download google/gemma-4-31B-it on the login node, ~62 GB).
# Judge-only passes: no rendering, no describing. Output tags get the suffix _gemma.
#   TAGS="v3_q38 v4b v4a v4ab" sbatch slurm/job_rejudge_gemma.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
TAGS="${TAGS:-v3_q38 v4b}"; LIMIT="${LIMIT:-100}"
for T in $TAGS; do
  env GEN=diffusion LIMIT="$LIMIT" TAG="${T}_gemma" DESC_TAG="$T" JUDGE=google/gemma-4-31B-it PHASE=judge bash slurm/job_protocol.sh
  env GEN=diffusion LIMIT="$LIMIT" TAG="${T}_gemma_grounded" DESC_TAG="$T" JUDGE=google/gemma-4-31B-it PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh
done
echo DONE
