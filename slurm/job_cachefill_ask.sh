#!/bin/bash
#SBATCH --job-name=cfask
#SBATCH --output=logs/cfask_%j.out
#SBATCH --error=logs/cfask_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Cache fill 1 Oct (docs/prereg_round13_detector_push.md): listener answers for the runs SHIP8+MD3 asks but finds no cached
# item for (benchmark/gold/listener_fill.py; Qwen3-Omni variants + AF Next, code imported unchanged), appended to the base
# caches, replayed through the stage-4 lookup, then the affected clips' stage-4 entries / arm folders are moved aside.
# Old DEV in ~/MscProj_r13, DEV2 in ~/MscProj_tg. Submit from ~/MscProj_r13.
set -euo pipefail
mkdir -p "$HOME/MscProj_r13/logs"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
F=benchmark/gold/listener_fill.py
cd "$HOME/MscProj_r13"
python $F --split dev missing
[ -f benchmark/gold/dev_listener_fill.json ] || python $F --split dev pool
python $F --split dev score
python $F --split dev merge
python $F --split dev replay
cd "$HOME/MscProj_tg"
python $F --split dev2 missing
[ -f benchmark/gold/dev2_listener_fill.json ] || python $F --split dev2 pool
python $F --split dev2 score
python $F --split dev2 merge
python $F --split dev2 replay
cp "$HOME/MscProj_r13/benchmark/gold/dev_listener_v.json" "$HOME/MscProj_r13/benchmark/gold/dev_listener_afn.json" \
   "$HOME/MscProj_r13/benchmark/gold/dev_listener_fill.json" "$HOME/MscProj_r13/benchmark/gold/dev_listener_fill_afn.json" \
   "$HOME/MscProj_tg/benchmark/gold/"
cd "$HOME/MscProj_r13"; python $F --split dev reset
cd "$HOME/MscProj_tg"; python $F --split dev2 reset
echo "DONE cfask"
