#!/bin/bash
#SBATCH --job-name=as_det
#SBATCH --output=logs/as_det_%j.out
#SBATCH --error=logs/as_det_%j.err
#SBATCH --partition=L4-4h,L4-12h,A100-4h,RTX6000-4h,L40s-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# The three detectors on gold slice B (benchmark/audioset_detector_eval.py).
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}" HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
export PATH="$HOME/miniconda3/envs/msproj/bin:$PATH"
conda activate msproj && python -m benchmark.audioset_detector_eval --cache beats; conda deactivate
conda activate psed   && python -m benchmark.audioset_detector_eval --cache psed;  conda deactivate
conda activate sota   && python -m benchmark.audioset_detector_eval --cache flam;  conda deactivate
conda activate msproj && python -m benchmark.audioset_detector_eval --eval
echo DONE
