#!/bin/bash
#SBATCH --job-name=protocol_v4
#SBATCH --output=logs/v4_%j.out
#SBATCH --error=logs/v4_%j.err
#SBATCH --partition=H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=adamgavriely@gmail.com
#
# THE MAIN EXPERIMENT UNDER THE v4 CONFIGURATION (docs/prereg_v4.md), one swap at a time.
#
#   STAGES=4      sbatch slurm/job_v4.sh      -> v4a  (FLAM)                        TAG=v4a
#   STAGES=45     sbatch slurm/job_v4.sh      -> v4b  (+ Qwen3.8-27B)               TAG=v4b
#   STAGES=456    sbatch slurm/job_v4.sh      -> v4c  (+ Qwen-Image-2512)           TAG=v4c
#   STAGES=23456  sbatch slurm/job_v4.sh      -> v4   (+ Granite, SAM 3)            TAG=v4
#   LIMIT=20 ...                              -> the 20-clip check first
#
# The evaluation pair is the same for every v4 row: describer Qwen3.8-27B (stage 5 applied
# in the describe phase whatever STAGES says), judge Gemma-4-31B-it. Same stamp-and-resume
# behaviour as job_v3.sh: a 12 h kill resumes, resubmit the same command.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs benchmark/.chain
STAGES="${STAGES:-4}"
LIMIT="${LIMIT:-100}"
JUDGE="${JUDGE:-google/gemma-4-31B-it}"
case "$STAGES" in
  4) TAG=v4a ;; 45) TAG=v4b ;; 456) TAG=v4c ;; 23456) TAG=v4 ;; *) TAG="v4_s$STAGES" ;;
esac
[ "$LIMIT" != "100" ] && TAG="${TAG}_n$LIMIT"
STAMP=benchmark/.chain/$TAG
mkdir -p "$STAMP"
echo "[v4] stages=$STAGES tag=$TAG limit=$LIMIT judge=$JUDGE"

step () {   # $1 stamp name, rest = command
    local name="$1"; shift
    if [ -f "$STAMP/$name" ]; then echo "=== $name: done earlier, skipping ==="; return; fi
    echo "=== $name ==="
    "$@"
    touch "$STAMP/$name"
}

source "$HOME/miniconda3/etc/profile.d/conda.sh"
# FLAM pre-pass (env sota): raw scores for every benchmark clip, read by stage 4
if [[ "$STAGES" == *4* ]]; then
    step flam_cache bash -c 'conda activate sota && HF_HUB_OFFLINE=1 python -m benchmark.flam_v2 --cache-benchmark'
fi
DESC_STAGES="${STAGES}"; [[ "$DESC_STAGES" == *5* ]] || DESC_STAGES="${DESC_STAGES}5"
step render   env V4="$STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="$TAG" PHASE=render bash slurm/job_protocol.sh
step describe env V4="$DESC_STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="$TAG" PHASE=describe SKIP_RENDER=1 bash slurm/job_protocol.sh
step judge    env V4="$DESC_STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="$TAG" JUDGE="$JUDGE" PHASE=judge bash slurm/job_protocol.sh
step grounded env V4="$DESC_STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="${TAG}_grounded" DESC_TAG="$TAG" JUDGE="$JUDGE" PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh

echo "DONE -> benchmark/protocol_results_${TAG}.json (+ _${TAG}_grounded.json)"
