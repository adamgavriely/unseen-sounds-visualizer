#!/bin/bash
#SBATCH --job-name=fxcalib
#SBATCH --output=logs/fxc_%j.out
#SBATCH --error=logs/fxc_%j.err
#SBATCH --partition=generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Amendment 11: FlexSED over the 280-clip AudioSet-Strong calibration set, so a per-family bar can
# be fitted without the gold set being touched. The overlap between these ids and the gold's
# AudioSet ids was checked before this ran and is zero.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
python benchmark/gold/flexsed_run.py --clip-dir "$SLURM_SUBMIT_DIR/data/input/audioset_calib" \
    --out "$SLURM_SUBMIT_DIR/data/work/flexsed_calib" --shard ${SHARD:-0} --of ${OF:-1} --batch ${BATCH:-24}
