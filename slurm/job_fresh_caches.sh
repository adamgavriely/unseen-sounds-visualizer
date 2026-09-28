#!/bin/bash
#SBATCH --job-name=freshcache
#SBATCH --output=logs/freshc_%j.out
#SBATCH --error=logs/freshc_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Fresh confirmation set (docs/prereg_fresh_confirm_set.md, 2026-09-28): the four caches the shipped stack reads
# (detector_round2.usable) -- BEATs, PANNs, PE-A-Frame, FlexSED (215 family queries) -- same code and arguments as the
# 415 (job_heldout_caches.sh, job_pe_frame.sh WHICH=heldout, job_flexsed_heldout.sh), new folders only.
# CACHES ONLY: nothing is scored on this set until a candidate has passed the 415.
#   sbatch slurm/job_fresh_caches.sh            (STEPS="beats panns pe flexsed" by default)
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'" ) || true
    export HF_TOKEN
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for st in ${STEPS:-beats panns pe flexsed}; do
  case "$st" in
    beats)   python -m benchmark.audioset_detector_eval --cache beats --set fresh ;;
    panns)   python -c "from benchmark import audioset_stage4_report as R; R.use_set('fresh'); R.panns_cache('cuda'); print('panns done')" ;;
    pe)      python benchmark/gold/pe_frame_run.py --clip-dir data/input/audioset_fresh \
                 --out benchmark/audioset_fresh_windows/pe_frame --shard 0 --of 1 --limit 0 ;;
    flexsed) python benchmark/gold/flexsed_run.py --clip-dir "$SLURM_SUBMIT_DIR/data/input/audioset_fresh" \
                 --out "$SLURM_SUBMIT_DIR/data/work/flexsed_fresh" --shard 0 --of 1 --batch 24 ;;
  esac
done
echo "DONE fresh caches"
