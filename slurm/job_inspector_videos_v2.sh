#!/bin/bash
#SBATCH --job-name=insp_vid2
#SBATCH --output=logs/inspvid2_%j.out
#SBATCH --error=logs/inspvid2_%j.err
#SBATCH --partition=cpu192G-48h
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#
# Inspector videos, round 2 of the picture check (28 Sept 2026). New output dirs, because recomposite.py skips a video
# that already exists (the 06:23 blind jobs into data/output/inspector/ rendered nothing for that reason).
#   WHAT=ours   <clip>.mp4 + <clip>_debug.mp4 from data/work/shipped_v_<TAG> (PICTURE_VERIFY on) into
#               data/output/inspector_v2/<SPLIT>/; CLIPS="a b" limits it to the clips whose pictures changed
#   WHAT=blind  <clip>_blind.mp4 from data/work/protocol_blind_a2i_<TAG>, label chips, into
#               data/output/inspector_b2/<SPLIT>/ -- with --shipped, so the same display rules as ours
#               (PICTURE_MIN_CONF 0.40, MAX_AFTER_END 1.0, MERGE_GAP 1.5); the older _blind videos predate them
#   TAG=dev_monocap_v31 SPLIT=DEV WHAT=blind sbatch slurm/job_inspector_videos_v2.sh
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1
case "${WHAT:?WHAT required}" in
  ours)
    OUT="data/output/inspector_v2/${SPLIT:?SPLIT required}"
    python scripts/recomposite.py --root "data/work/shipped_v_${TAG:?}" --shipped --require-local-images \
        --modes clean --out-name "{stem}.mp4" --out "$OUT" --drop-panels ${CLIPS:+--clips $CLIPS}
    python scripts/recomposite.py --root "data/work/shipped_v_${TAG}" --shipped --require-local-images \
        --modes debug --out-name "{stem}_debug.mp4" --out "$OUT" --drop-panels ${CLIPS:+--clips $CLIPS} ;;
  blind)
    OUT="data/output/inspector_b2/${SPLIT:?SPLIT required}"
    python scripts/recomposite.py --root "data/work/protocol_blind_a2i_${TAG:?}" --shipped --render-mode minimal \
        --modes clean --out-name "{stem}_blind.mp4" --out "$OUT" --drop-panels ;;
esac
echo "DONE inspector_v2 $WHAT $TAG"
