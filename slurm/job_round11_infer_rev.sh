#!/bin/bash
#SBATCH --job-name=r11inferR
#SBATCH --output=logs/r11inferR_%j.out
#SBATCH --error=logs/r11inferR_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#
# Detector round 11 (docs/prereg_round11_masker.md): FlexSED re-run on the band-candidate clips of one set
# Second worker, reverse clip order (same outputs; each clip is skipped once done).
# (orig + Speech/Music queries, foreign speech 0 dB, foreign music 0 dB, time-reversed). Mixed wavs are temporary.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python benchmark/detector_round11.py infer --set ${SET:-calib} --shard ${SHARD:-0} --of ${OF:-1} --reverse
echo "DONE round11 infer ${SET:-calib}"
