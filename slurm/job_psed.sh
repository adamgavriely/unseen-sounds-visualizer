#!/bin/bash
#SBATCH --job-name=psed_det
#SBATCH --output=logs/psed_%j.out
#SBATCH --error=logs/psed_%j.err
#SBATCH --partition=A100-4h,RTX6000-4h,H200-4h,L40s-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#
# PretrainedSED BEATs-strong as the detector (benchmark/psed_eval.py, docs/prereg_psed.md):
# cache DCASE test, dev and every benchmark clip in the `psed` env, then the declared bars.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export PATH="$HOME/miniconda3/envs/msproj/bin:$PATH"     # ffmpeg lives in msproj
conda activate psed   && python -m benchmark.psed_eval --cache; conda deactivate
conda activate msproj && python -m benchmark.psed_eval --eval
echo DONE
