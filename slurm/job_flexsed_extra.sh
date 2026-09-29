#!/bin/bash
#SBATCH --job-name=fxextra
#SBATCH --output=logs/fxextra_%j.out
#SBATCH --error=logs/fxextra_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# FlexSED extra queries (benchmark/gold/flexsed_extra.py): the 147 drawable labels with no FlexSED query of their own
# (benchmark/gold/flexsed_extra_queries.json), same code/settings/audio as data/work/flexsed_cache, on the 49 DEV and
# the 60 TEST stems into NEW folders data/work/flexsed_extra_{dev,test}. TEST: features only (run mode reads no gold).
# Then the DEV screen (gold, DEV only) -> benchmark/gold/flexsed_extra_dev.json.
#   sbatch slurm/job_flexsed_extra.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/flexsed_extra.py run --set dev --batch 24
python benchmark/gold/flexsed_extra.py run --set test --batch 24
python benchmark/gold/flexsed_extra.py screen
echo "DONE flexsed_extra"
