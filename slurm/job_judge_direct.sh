#!/bin/bash
#SBATCH --job-name=judge_direct
#SBATCH --output=logs/judgedirect_%j.out
#SBATCH --error=logs/judgedirect_%j.err
#SBATCH --partition=H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# The describer-free judge (benchmark/gold/judge_direct.py): Gemma-4-31B looks at the pictures, the
# reference is a template over the annotator's ticks. TAG, SYSTEMS, OUT; REPEAT=20 for the B3 repeat.
#   TAG=v4b4 OUT=benchmark/judge_direct_v4b4.json sbatch slurm/job_judge_direct.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
# Gemma-4 reading images needs torch >= 2.6; msproj has 2.5.1, so the judge runs in a venv on top of
# the sota env (torch 2.7.1) with transformers 5.16.1 (same version as msproj)
PY="${PY:-$HOME/venvs/judge/bin/python}"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" benchmark/gold/judge_direct.py --tag "${TAG:?}" --systems "${SYSTEMS:-proposed,blind_a2i,audio_caption}" \
    --out "${OUT:?}" ${REPEAT:+--repeat "$REPEAT"}
echo "DONE"
