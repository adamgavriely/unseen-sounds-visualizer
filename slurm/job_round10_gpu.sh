#!/bin/bash
#SBATCH --job-name=r10gpu
#SBATCH --output=logs/r10gpu_%j.out
#SBATCH --error=logs/r10gpu_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Detector round 10, GPU: FlexSED paraphrase queries (R2, all clips) and FlexSED on the 3 perturbed versions (R3,
# candidate clips). SET, SHARD, OF choose the work. Skips caches that exist, so a timeout is a resubmit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/round10_flexsed.py --work "$SLURM_SUBMIT_DIR/data/work/round10_work_${SET}.json" \
    --shard ${SHARD:-0} --of ${OF:-1} --batch ${BATCH:-24}
echo "DONE round10 gpu $SET ${SHARD:-0}/${OF:-1}"
