#!/bin/bash
#SBATCH --job-name=r7bcache
#SBATCH --output=logs/r7bcache_%j.out
#SBATCH --error=logs/r7bcache_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Detector round 7b (retry: descriptive queries, predict_spans=True, erase check) (docs/prereg_round7_samaudio.md): QC gate (chooses the residual route), then BEATs and FlexSED on the
# residual (and the identity route) into new folders (beats_r7bres/, flexsed_<set>_r7bres/, *_r7bident/). Env msproj.
#   sbatch --export=ALL,STEPS=qc slurm/job_round7b_cache.sh
#   sbatch --export=ALL,SET=calib,STEPS="beats-res beats-ident flex-ident flex-res" slurm/job_round7b_cache.sh
#   sbatch --export=ALL,SET=heldout,STEPS="beats-res flex-res" slurm/job_round7b_cache.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export R7_TAG=7b PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for st in ${STEPS:?}; do
  case "$st" in
    qc)          python benchmark/detector_round7.py qc ;;
    beats-res)   python benchmark/detector_round7.py cache-beats --set "${SET:?}" --route res ;;
    beats-ident) python benchmark/detector_round7.py cache-beats --set calib --route ident ;;
    flex-res)    python benchmark/detector_round7.py cache-flex --set "${SET:?}" --route res ;;
    flex-ident)  python benchmark/detector_round7.py cache-flex --set calib --route ident ;;
    *) echo "unknown step $st"; exit 2 ;;
  esac
done
echo "DONE round7b cache ${SET:-} ${STEPS}"
