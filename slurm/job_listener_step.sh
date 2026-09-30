#!/bin/bash
#SBATCH --job-name=lisstep
#SBATCH --output=logs/lisstep_%j.out
#SBATCH --error=logs/lisstep_%j.err
#SBATCH --partition=H200-4h,A100-4h,RTX6000-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#
# Rounds 25-26 screen: Step-Audio-2-mini as a further open-inventory listener (V4) on the P2/PV items already answered by
# Qwen3-Omni, Audio Flamingo Next and Kimi-Audio (benchmark/gold/listener_step.py). Generation in ~/venvs/stepaudio
# (judge-venv base, torch 2.7.1, transformers 4.49.0 per the Step-Audio2 README, official code at ~/Step-Audio2), matching in
# msproj (the stack that scored Qwen/AF). No gold.
# Usage (from ~/MscProj): sbatch slurm/job_listener_step.sh "dev dev2" [SMOKE_N] [ORDER]  (SMOKE_N > 0: *_smoke.json only;
# ORDER text_first (README multi-modal example, default) or audio_first (the Qwen / AF order))
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SMOKE="${2:-0}"; ORDER="${3:-text_first}"
for split in ${1:-dev}; do
    "$HOME/venvs/stepaudio/bin/python" benchmark/gold/listener_step.py gen "$split" --smoke "$SMOKE" --order "$ORDER"
    ( source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
      python benchmark/gold/listener_step.py match "$split" --smoke "$SMOKE" )
done
