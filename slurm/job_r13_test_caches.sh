#!/bin/bash
#SBATCH --job-name=r13test
#SBATCH --output=logs/r13test_%j.out
#SBATCH --error=logs/r13test_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# Round 13 TEST PREPARATION (docs/prereg_round13_detector_push.md; benchmark/gold/r13_test_prep.py). No gold is read,
# nothing is scored: builds the DEV harness's caches for the 60 TEST clips (BEATs j2 framewise, 16-kHz wav, PANNs clip
# peaks) and runs the plumbing gates D0 (stage-4 rebuild == scored trace) and D5 (B0r stage-5 rebuild == scored
# test_final_v33 augmentations / on-screen spans, structural) for B0r only. Submit from ~/MscProj_r13 (round-13 source;
# data/ and ckpts/ link to ~/MscProj). Resumable.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=benchmark/gold/r13_test_prep.py
python $P check
python $P wav16
python $P beats
python $P panns
python $P stage4
python $P stage5
python $P d5
python $P check
echo "DONE r13test"
