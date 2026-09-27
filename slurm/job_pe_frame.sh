#!/bin/bash
#SBATCH --job-name=peframe
#SBATCH --output=logs/pef_%j.out
#SBATCH --error=logs/pef_%j.err
#SBATCH --partition=H200-4h,A100-4h,generic
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Amendment 24: PE-A-Frame frame scores (benchmark/gold/pe_frame_run.py). WHICH = gold | calib | heldout; LIMIT for a smoke test.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HOME/.bashrc" ]; then
    HF_TOKEN=$(sed -n 's/^[[:space:]]*export[[:space:]]*HF_TOKEN=//p' "$HOME/.bashrc" | tail -1 | tr -d "\"'" ) || true
    export HF_TOKEN
fi
case "${WHICH:-gold}" in
  gold)    ARGS=(--out data/work/pe_frame_cache) ;;
  calib)   ARGS=(--clip-dir data/input/audioset_calib --out benchmark/audioset_calib_windows/pe_frame) ;;
  heldout) ARGS=(--clip-dir data/input/audioset_heldout --out benchmark/audioset_heldout_windows/pe_frame) ;;
esac
python benchmark/gold/pe_frame_run.py "${ARGS[@]}" --shard ${SHARD:-0} --of ${OF:-1} --limit ${LIMIT:-0}
