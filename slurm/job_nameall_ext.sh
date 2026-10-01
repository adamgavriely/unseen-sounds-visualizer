#!/bin/bash
#SBATCH --job-name=nameall_ext
#SBATCH --output=logs/nameall_ext_%j.out
#SBATCH --error=logs/nameall_ext_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=03:50:00
#
# Round 66 amendment (B, A, V) + Round 62b PRIOR on gate-gold (docs/prereg_round13_detector_push.md). Submit from ~/MscProj. Resumable per clip.
set -uo pipefail
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cd "$HOME/MscProj"
python benchmark/gold/nameall_ext.py run 2>&1 | grep -v "Loading weights"
until [ "$(ls benchmark/gold/gate_gold/nameall_Qwen38-27B/*.json 2>/dev/null | grep -vc /_)" -ge 49 ]; do sleep 120; done   # NAME-ALL cache (job_nameall.sh)
python benchmark/gold/nameall_ext.py score
echo "NAMEALL_EXT_DONE rc=$?"
