#!/bin/bash
#SBATCH --job-name=liskimi
#SBATCH --output=logs/liskimi_%j.out
#SBATCH --error=logs/liskimi_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#
# Round 16 N3: Kimi-Audio-7B-Instruct as a third open-inventory listener (V4) on the P2/PV items already answered by Qwen3-Omni
# and Audio Flamingo Next (benchmark/gold/listener_kimi.py). Generation in ~/venvs/kimi (transformers 4.51 + flash-attn +
# kimia_infer), matching in msproj (the stack that scored Qwen/AF). No gold.
# Usage (from ~/MscProj): sbatch slurm/job_listener_kimi.sh "dev dev2" [SMOKE_N] [ORDER]  (SMOKE_N > 0: *_smoke.json only;
# ORDER text_first (README, default) or audio_first (the Qwen / AF order))
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SMOKE="${2:-0}"; ORDER="${3:-text_first}"
for split in ${1:-dev}; do
    "$HOME/venvs/kimi/bin/python" benchmark/gold/listener_kimi.py gen "$split" --smoke "$SMOKE" --order "$ORDER"
    ( source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
      python benchmark/gold/listener_kimi.py match "$split" --smoke "$SMOKE" )
done
