#!/bin/bash
#SBATCH --job-name=sliceB_render
#SBATCH --partition=L4-12h
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/sliceb_%j.out
#SBATCH --error=logs/sliceb_%j.err
# Gold set 2: the frozen v4b pipeline (BEATs + Qwen3.8-27B; FLUX), the adopted v4 detector row rendered once on the 111
# slice-B clips, three systems, render only (no judge): inputs for the per-sound scorer
# (docs/metric_per_sound.md). Resumable per clip.
set -uo pipefail
cd ~/MscProj
echo "[sliceB] host=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)"
env V4=5 GEN=diffusion CLIP_DIR=data/input/audioset_strong TAG=v4b_sliceB PHASE=render bash slurm/job_protocol.sh
echo "[sliceB] done"
