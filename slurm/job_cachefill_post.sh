#!/bin/bash
#SBATCH --job-name=cfpost
#SBATCH --output=logs/cfpost_%j.out
#SBATCH --error=logs/cfpost_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Cache fill 1 Oct, after the SHIP8+MD3 re-run (job_round16_dev.sh + job_tagger_arms.sh): picture diff of the affected clips;
# clips whose drawn pictures changed are re-asked by the existing DEPICT / GROUP steps (only those clips: a folder of links),
# then merged DEV is re-scored and the missing counts are re-read. Submit from ~/MscProj_r13.
set -euo pipefail
mkdir -p "$HOME/MscProj_r13/logs"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
F=benchmark/gold/listener_fill.py
W="$HOME/MscProj/data/work"
B="$W/cachefill_bak_1oct"
ask() {   # $1 split, $2 arm root, $3 wav16 dir
  local L="$B/$1/relink"
  rm -rf "$L"; mkdir -p "$L"
  for st in $(python -c "import json; print(' '.join(json.load(open('$B/$1/post.json'))['changed']))"); do
    ln -s "$2/$st" "$L/$st"
  done
  if [ -n "$(ls -A "$L")" ]; then
    python -m src.stage6_visual_augmentation.depict "$L" "$W/depict_answers.json"
    python -m src.stage6_visual_augmentation.group "$L" "$3" "$W/group_answers.json"
  fi
}
cd "$HOME/MscProj_r13"
python $F --split dev post
ask dev "$W/r13/SHIP8+MD3_proposed" "$W/devcand/wav16"
python $F --split dev missing
cd "$HOME/MscProj_tg"
python $F --split dev2 post
ask dev2 "$W/r13dev2/SHIP8+MD3_proposed" "$W/r13dev2/wav16"
python $F --split dev2 missing
cd "$HOME/MscProj_r13"; python $F --split dev diff
cd "$HOME/MscProj_tg"; python $F --split dev2 diff
TG_ARMS="SHIP8+MD3" python benchmark/gold/merged_dev.py --arms B0r "TO1+F7F8" "SHIP8+MD3"
echo "DONE cfpost"
