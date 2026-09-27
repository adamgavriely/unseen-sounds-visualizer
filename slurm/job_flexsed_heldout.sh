#!/bin/bash
#SBATCH --job-name=fxheld
#SBATCH --output=logs/fxc_%j.out
#SBATCH --error=logs/fxc_%j.err
#SBATCH --partition=H200-4h,A100-4h,generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Amendment 24: FlexSED over the new held-out AudioSet-Strong set (ids disjoint from calib, slice B and gold).
# be fitted without the gold set being touched. The overlap between these ids and the gold's
# AudioSet ids was checked before this ran and is zero.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/gold/flexsed_run.py --clip-dir "$SLURM_SUBMIT_DIR/data/input/audioset_heldout" \
    --out "$SLURM_SUBMIT_DIR/data/work/flexsed_heldout" --shard ${SHARD:-0} --of ${OF:-1} --batch ${BATCH:-24}
