#!/bin/bash
#SBATCH --job-name=insp_vid_v
#SBATCH --output=logs/inspvidv_%j.out
#SBATCH --error=logs/inspvidv_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#
# Inspector "ours" videos from the checked repaint (data/work/shipped_v_<TAG>, PICTURE_VERIFY on) into
# data/output/inspector_v/<SPLIT>/: <clip>.mp4 (clean) and <clip>_debug.mp4. The _blind videos are unchanged.
#   TAG=dev_monocap_v31 SPLIT=DEV sbatch slurm/job_inspector_videos_v.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
OUT="data/output/inspector_v/${SPLIT:?SPLIT required}"
python scripts/recomposite.py --root "data/work/shipped_v_${TAG:?}" --shipped --require-local-images \
    --modes clean --out-name "{stem}.mp4" --out "$OUT" --drop-panels
python scripts/recomposite.py --root "data/work/shipped_v_${TAG}" --shipped --require-local-images \
    --modes debug --out-name "{stem}_debug.mp4" --out "$OUT" --drop-panels
echo "DONE inspector_v $TAG"
