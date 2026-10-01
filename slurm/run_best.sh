#!/bin/bash
#SBATCH --job-name=runbest
#SBATCH --output=logs/runbest_%j.out
#SBATCH --error=logs/runbest_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Run a version (default the best DEV version TO1+F7F8) on any folder of clips, gold-free, with the same harness that
# built the DEV/TEST/tagger caches (benchmark/gold/tagger_prep.py): FlexSED, scored render (B0), wav16, BEATs, PANNs,
# B0r stage 4/5, listener pools, Qwen3-Omni, Audio Flamingo Next, DASM, then the arm's stage 4/5. Output: the arm's
# augmentations.json per clip in data/work/r13<NAME>/<ARM>_proposed/<clip>/. Submit from ~/MscProj:
#   sbatch slurm/run_best.sh NAME /path/to/clip_folder [ARM]
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
NAME="${1:?name}"; SRC="${2:?clip folder}"; ARM="${3:-TO1+F7F8}"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_EXTRA_SPLITS="$NAME"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
D="data/input/tagger_$NAME"; mkdir -p "$D"
for f in "$SRC"/*.mp4; do [ -e "$D/$(basename "$f")" ] || ln -s "$(readlink -f "$f")" "$D/$(basename "$f")"; done
ls "$D" | sed 's/\.mp4$//' | sort > "benchmark/gold/${NAME}_stems.txt"
P=benchmark/gold/tagger_prep.py
python $P --split $NAME flexsed
python $P --split $NAME render
python $P --split $NAME wav16
python $P --split $NAME beats
python $P --split $NAME panns
python $P --split $NAME stage4 --arms B0r
python $P --split $NAME stage5 --arms B0r
python $P --split $NAME lpool
python $P --split $NAME qwen
python $P --split $NAME afn
python $P --split $NAME dasm
python $P --split $NAME stage4 --arms B0r "$ARM"
python $P --split $NAME stage5 --arms B0r "$ARM"
python $P --split $NAME gates --arms B0r "$ARM"
# Round 47 GROUP (shipped 1 Oct): Omni same/new answers for close repeats, read at display time via config.GROUP_CACHE
python -m src.stage6_visual_augmentation.group "data/work/r13$NAME/${ARM}_proposed" "data/work/r13$NAME/wav16"   # -> WORK_DIR/group_answers.json (use_shipped GROUP_CACHE)
echo "DONE runbest $NAME $ARM"
