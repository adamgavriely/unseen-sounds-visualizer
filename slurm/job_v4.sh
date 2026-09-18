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
#   STAGES=5      sbatch slurm/job_v4.sh      -> v4b  (Qwen3.8-27B gate; BEATs stays)   TAG=v4b
#   STAGES=56     sbatch slurm/job_v4.sh      -> v4c  (+ Qwen-Image-2512)               TAG=v4c
#   STAGES=2356   sbatch slurm/job_v4.sh      -> v4   (+ Granite, SAM 3)                TAG=v4
#   STAGES=4 / 45 -> v4a / v4ab, only once a stage-4 swap passes its pre-registered bars
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
# stage 4 has no passing swap yet (FLAM x2 and PretrainedSED failed their bars, docs/prereg_*.md),
# so the cumulative rows are 5 -> 56 -> 2356 with BEATs; "4" rows exist for a future pass
case "$STAGES" in
  4) TAG=v4a ;; 45) TAG=v4ab ;; 5) TAG=v4b ;; 56) TAG=v4c ;; 2356) TAG=v4 ;; 23456) TAG=v4_all ;; *) TAG="v4_s$STAGES" ;;
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
# PretrainedSED pre-pass (env psed): raw scores for every benchmark clip, read by stage 4
# (benchmark/psed_eval.py --cache already fills data/work/psed_cache; this is idempotent)
if [[ "$STAGES" == *4* ]]; then
    step psed_cache bash -c 'source "$HOME/miniconda3/etc/profile.d/conda.sh"; export PATH="$HOME/miniconda3/envs/msproj/bin:$PATH"; conda activate psed && HF_HUB_OFFLINE=1 python -m benchmark.psed_eval --cache'
fi
DESC_STAGES="${STAGES}"; [[ "$DESC_STAGES" == *5* ]] || DESC_STAGES="${DESC_STAGES}5"
step render   env V4="$STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="$TAG" PHASE=render bash slurm/job_protocol.sh
step describe env V4="$DESC_STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="$TAG" PHASE=describe SKIP_RENDER=1 bash slurm/job_protocol.sh
step judge    env V4="$DESC_STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="$TAG" JUDGE="$JUDGE" PHASE=judge bash slurm/job_protocol.sh
step grounded env V4="$DESC_STAGES" GEN=diffusion LIMIT="$LIMIT" TAG="${TAG}_grounded" DESC_TAG="$TAG" JUDGE="$JUDGE" PHASE=judge GROUNDED=1 bash slurm/job_protocol.sh

echo "DONE -> benchmark/protocol_results_${TAG}.json (+ _${TAG}_grounded.json)"
