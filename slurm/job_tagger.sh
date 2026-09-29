#!/bin/bash
#SBATCH --job-name=tagger
#SBATCH --output=logs/tagger_%j.out
#SBATCH --error=logs/tagger_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Confirmation set 1, REVISED (docs/prereg_round13_detector_push.md): gold-free features for the tagger set, DEV2 (6) and
# TEST2 (10, sealed), benchmark/gold/tagger_prep.py. Submit from ~/MscProj_tg (a frozen copy of the ~/MscProj code;
# data/ and ckpts/ link to ~/MscProj). One phase per job, chained with --dependency=afterok:
#   sbatch slurm/job_tagger.sh prep    # FlexSED 215 + extra, render (use_scored, placeholder pictures), wav16, BEATs,
#                                      # PANNs, DASM, B0r stage 4/5, gates
#   sbatch slurm/job_tagger.sh qwen    # listener pools + Qwen3-Omni yes/no + amendment-A variants (both splits, one load)
#   sbatch slurm/job_tagger.sh afn     # Audio Flamingo Next V4 + yes/no (both splits)
#   sbatch slurm/job_tagger.sh arms    # B1 and C1 = TO1+F7F8 stage 4/5, gates; then DEV2 ONLY is scored
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PHASE="${1:?prep|qwen|afn|arms}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=benchmark/gold/tagger_prep.py
case "$PHASE" in
  prep)
    python $P links
    for S in dev2 test2; do
      python $P --split $S flexsed
      python $P --split $S flexx
      python $P --split $S render
      python $P --split $S wav16
      python $P --split $S beats
      python $P --split $S panns
      python $P --split $S dasm
      python $P --split $S stage4 --arms B0r
      python $P --split $S stage5 --arms B0r
      python $P --split $S gates --arms B0r
    done
    python $P check ;;
  qwen)
    for S in dev2 test2; do python $P --split $S lpool; done
    python $P qwen
    python $P check ;;
  afn)
    python $P afn
    python $P check ;;
  arms)
    for S in dev2 test2; do
      python $P --split $S stage4 --arms B0r B1 "TO1+F7F8"
      python $P --split $S stage5 --arms B0r B1 "TO1+F7F8"
      python $P --split $S gates --arms B0r B1 "TO1+F7F8"
    done
    python $P check
    python $P --split dev2 score --arms B0r B1 "TO1+F7F8" ;;
esac
echo "DONE tagger $PHASE"
