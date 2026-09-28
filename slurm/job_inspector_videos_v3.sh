#!/bin/bash
#SBATCH --job-name=insp_vid3
#SBATCH --output=logs/inspvid3_%j.out
#SBATCH --error=logs/inspvid3_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#
# Inspector "ours" videos for the clips whose pictures were redrawn on 28 Sept (maker rule / VLM-written looks):
# <clip>.mp4 + <clip>_debug.mp4 from data/work/<ROOT>_<TAG> into data/output/inspector_v3/<SPLIT>/.
#   TAG=sliceB_v32 SPLIT=sliceB CLIPS="a b" ROOT=shipped_v2 sbatch slurm/job_inspector_videos_v3.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
OUT="data/output/inspector_v3/${SPLIT:?SPLIT required}"
python scripts/recomposite.py --root "data/work/${ROOT:-shipped_v2}_${TAG:?}" --shipped --require-local-images \
    --modes clean --out-name "{stem}.mp4" --out "$OUT" --drop-panels --clips ${CLIPS:?CLIPS required}
python scripts/recomposite.py --root "data/work/${ROOT:-shipped_v2}_${TAG}" --shipped --require-local-images \
    --modes debug --out-name "{stem}_debug.mp4" --out "$OUT" --drop-panels --clips ${CLIPS}
echo "DONE inspector_v3 $TAG"
