#!/bin/bash
#SBATCH --job-name=tagens
#SBATCH --output=logs/tagens_%j.out
#SBATCH --error=logs/tagens_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 63 TAG-ENS (docs/prereg_round13_detector_push.md): EAT + SSLAM frames on BEATs' windows for the 415, DEV and DEV2
# (manifests written first on the login node), then the half-A fit and the half-B step 1. Submit from ~/MscProj.
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
P=benchmark/gold/tagens.py
conda activate msproj
for s in heldout dev dev2; do python $P cache --set $s --model eat || exit 1; python $P decode --set $s || exit 1; done
conda activate sota          # SSLAM's remote code does not load under msproj's transformers (job 31695844); audio decoded in msproj
for s in heldout dev dev2; do python $P cache --set $s --model sslam || exit 1; done
conda activate msproj
python $P fit || { echo "STOP at fit"; exit 3; }
python $P step1
echo "step1 exit $?"
echo "DONE tagens"
