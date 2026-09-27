#!/bin/bash
#SBATCH --job-name=insp_vid
#SBATCH --output=logs/inspvid_%j.out
#SBATCH --error=logs/inspvid_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#
# Inspector page videos (CPU, ffmpeg): for one scored tag, into data/output/inspector/<SPLIT>/
#   WHAT=ours   <clip>.mp4 (clean) and <clip>_debug.mp4 from data/work/shipped_<TAG> (Qwen-Image pictures, full opacity)
#   WHAT=blind  <clip>_blind.mp4 from data/work/protocol_blind_a2i_<TAG>, label chips only (no FLUX picture)
#   TAG=dev_monocap_v31 SPLIT=DEV WHAT=blind sbatch slurm/job_inspector_videos.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
OUT="data/output/inspector/${SPLIT:?SPLIT required}"
case "${WHAT:?WHAT required}" in
  ours)
    python scripts/recomposite.py --root "data/work/shipped_${TAG:?}" --shipped --require-local-images \
        --modes clean --out-name "{stem}.mp4" --out "$OUT" --drop-panels
    python scripts/recomposite.py --root "data/work/shipped_${TAG}" --shipped --require-local-images \
        --modes debug --out-name "{stem}_debug.mp4" --out "$OUT" --drop-panels ;;
  blind)
    python scripts/recomposite.py --root "data/work/protocol_blind_a2i_${TAG:?}" --shipped --render-mode minimal \
        --modes clean --out-name "{stem}_blind.mp4" --out "$OUT" --drop-panels ;;
esac
echo "DONE inspector $WHAT $TAG"
