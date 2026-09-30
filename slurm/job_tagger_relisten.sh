#!/bin/bash
#SBATCH --job-name=relisten
#SBATCH --output=logs/relisten_%j.out
#SBATCH --error=logs/relisten_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Listener caches for the whole tagger DEV/TEST parts (batch 1 + 2): lpool kept the batch-1 files, so the batch-2 clips had
# no answers. Batch-1 files were renamed *_batch1.json; this rebuilds all three caches for both parts from scratch (same
# code, greedy decoding). Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=benchmark/gold/tagger_prep.py
for S in dev2 test2; do python $P --split $S lpool; done
python $P qwen
python $P afn
python $P check
echo "DONE relisten"
