#!/bin/bash
#SBATCH --job-name=gold_judge
#SBATCH --output=logs/gold_judge_%j.out
#SBATCH --error=logs/gold_judge_%j.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# The secondary metric on the gold set (amendment 5): describe the rendered panels of every clip in
# data/input/gold139/all (all three systems, one tag), judge with Mistral-7B (rubric-enforced), then
# the grounded judge (human tag from the gold ticks, benchmark/run_protocol.clips_to_run).
#   TAG=v4b4 V4=59 sbatch slurm/job_gold_judge.sh      (after every render shard of the tag is done)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs benchmark/.chain
TAG="${TAG:-v4b4}"; V4="${V4:-59}"; JUDGE="${JUDGE:-google/gemma-4-31B-it}"   # Gemma: the only judge that passed both trust checks
CLIP_DIR="${CLIP_DIR:-data/input/gold139/all}"
DESC_STAGES="$V4"; [[ "$DESC_STAGES" == *5* ]] || DESC_STAGES="${DESC_STAGES}5"
STAMP=benchmark/.chain/gold_$TAG; mkdir -p "$STAMP"
step () { local name="$1"; shift; if [ -f "$STAMP/$name" ]; then echo "=== $name: done earlier ==="; return; fi; echo "=== $name ==="; "$@"; touch "$STAMP/$name"; }
step describe env V4="$DESC_STAGES" GEN=diffusion CLIP_DIR="$CLIP_DIR" TAG="$TAG" PHASE=describe SKIP_RENDER=1 bash slurm/job_protocol.sh
step judge    env V4="$DESC_STAGES" GEN=diffusion CLIP_DIR="$CLIP_DIR" TAG="$TAG" JUDGE="$JUDGE" PHASE=judge bash slurm/job_protocol.sh
step grounded env V4="$DESC_STAGES" GEN=diffusion CLIP_DIR="$CLIP_DIR" TAG="${TAG}_grounded" DESC_TAG="$TAG" JUDGE="$JUDGE" PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
python scripts/rubric_enforce.py "${TAG}_grounded"
echo "DONE -> benchmark/protocol_results_${TAG}_grounded_rubric.json"
