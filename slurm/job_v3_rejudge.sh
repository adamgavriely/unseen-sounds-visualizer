#!/bin/bash
#SBATCH --job-name=v3_rejudge
#SBATCH --output=logs/v3_rejudge_%j.out
#SBATCH --error=logs/v3_rejudge_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# v3's rendered panels, re-described by Qwen3.8-27B and re-judged by Gemma-4-31B-it: the
# v3 row of the v4 table under the v4 evaluation pair (docs/prereg_v4.md). No rendering.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs benchmark/.chain/v3_q38; STAMP=benchmark/.chain/v3_q38
LIMIT="${LIMIT:-100}"; JUDGE="${JUDGE:-google/gemma-4-31B-it}"
step () { local name="$1"; shift; if [ -f "$STAMP/$name" ]; then echo "=== $name: done ==="; return; fi; echo "=== $name ==="; "$@"; touch "$STAMP/$name"; }
step describe env V4=5 GEN=diffusion LIMIT="$LIMIT" TAG=v3_q38 WORK_TAG=v3 PHASE=describe SKIP_RENDER=1 bash slurm/job_protocol.sh
step judge    env V4=5 GEN=diffusion LIMIT="$LIMIT" TAG=v3_q38 JUDGE="$JUDGE" PHASE=judge bash slurm/job_protocol.sh
step grounded env V4=5 GEN=diffusion LIMIT="$LIMIT" TAG=v3_q38_grounded DESC_TAG=v3_q38 JUDGE="$JUDGE" PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh
echo DONE
