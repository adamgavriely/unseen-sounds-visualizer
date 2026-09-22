#!/bin/bash
#SBATCH --job-name=gold_render
#SBATCH --output=logs/gold_%j.out
#SBATCH --error=logs/gold_%j.err
#SBATCH --partition=A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#
# The re-run on the per-sound gold set (2026-09-22, docs/prereg_v4.md "Re-run outcome"): every
# clip the annotator finished (benchmark/gold/annotations/gold_AG.json, not marked bad), rendered
# under one declared configuration, one system per job, clips split into shards so that several
# GPUs work at once. Work dirs are keyed by clip stem, so shards share TAG.
#
#   SHARD=data/input/gold139/p1 SYSTEMS=proposed TAG=v4b4 V4=59 sbatch slurm/job_gold.sh
#   SHARD=... SYSTEMS=blind_a2i ...   SHARD=... SYSTEMS=audio_caption ...
#   V4=459 TAG=v4ab4 -> the PretrainedSED detector arm (needs the psed cache pre-pass, PSED_PRE=1)
#   GEN=placeholder for audio_caption: the caption row never shows its pictures (the scorer reads
#   the caption's timeline with require_image=False, the judge reads the caption text), so the
#   FLUX pass was wasted GPU time (advisor, 2026-09-22 15:00)
#
# Resumable: a clip with augmentations.json is skipped (benchmark/run_protocol.py phase_render).
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
SHARD="${SHARD:?SHARD folder required}"
SYSTEMS="${SYSTEMS:?SYSTEMS required}"
TAG="${TAG:-v4b4}"
V4="${V4:-59}"
echo "[gold] shard=$SHARD systems=$SYSTEMS tag=$TAG v4=$V4 clips=$(ls "$SHARD" | wc -l)"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
if [ "${PSED_PRE:-}" = "1" ]; then
    export PATH="$HOME/miniconda3/envs/msproj/bin:$PATH"
    conda activate psed && HF_HUB_OFFLINE=1 python -m benchmark.psed_eval --cache --clip-dir "$SHARD"
    conda deactivate
fi
env V4="$V4" BAR="${BAR:-}" GEN="${GEN:-diffusion}" CLIP_DIR="$SHARD" TAG="$TAG" SYSTEMS="$SYSTEMS" PHASE=render bash slurm/job_protocol.sh
echo "DONE shard=$SHARD systems=$SYSTEMS tag=$TAG"
