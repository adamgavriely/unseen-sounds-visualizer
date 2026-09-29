#!/bin/bash
#SBATCH --job-name=tsuggest2
#SBATCH --output=logs/tsuggest2_%j.out
#SBATCH --error=logs/tsuggest2_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# Detector suggestions for the tagging tool (clips d001-d029; the d030-d100 run was job_tagger_suggest.sh): the scored stage-4 stack (config.use_scored()) on the
# 71 videos in data/work/tagger_suggest/videos. Step 1 builds their FlexSED cache (215 family queries, same code and
# arguments as job_fresh_caches.sh), step 2 runs detect_events and writes benchmark/gold/tagger_suggest.json.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'" ) || true
    export HF_TOKEN
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python benchmark/gold/flexsed_run.py --clip-dir "$SLURM_SUBMIT_DIR/data/work/tagger_suggest/videos_d001_d029" \
    --out "$SLURM_SUBMIT_DIR/data/work/flexsed_tagger" --shard 0 --of 1 --batch 24
python benchmark/gold/tagger_suggest.py --videos data/work/tagger_suggest/videos_d001_d029 --out benchmark/gold/tagger_suggest_d001_d029.json
echo "DONE tagger_suggest"
