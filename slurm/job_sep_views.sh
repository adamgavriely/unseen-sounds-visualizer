#!/bin/bash
#SBATCH --job-name=sep_views
#SBATCH --partition=L4-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:50:00
#SBATCH --output=logs/sep_views_%j.out
#SBATCH --error=logs/sep_views_%j.err
# V4 amendment 2: HTDemucs views (env msproj) then PSED on the views (env psed).
#   sbatch slurm/job_sep_views.sh smoke      # first 20 calibration clips, determinism check
#   sbatch slurm/job_sep_views.sh full       # calib + sliceB separation, then PSED on all views
#   sbatch slurm/job_sep_views.sh psed       # PSED pass only
set -uo pipefail
cd ~/MscProj
MODE=${1:-smoke}
source ~/miniconda3/etc/profile.d/conda.sh
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
echo "[job] mode=$MODE host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
if [ "$MODE" != "psed" ]; then
    conda activate msproj
    if [ "$MODE" = "smoke" ]; then
        python -m benchmark.sep_views --separate --set calib --limit 20 --twice
    else
        python -m benchmark.sep_views --separate --set calib
        python -m benchmark.sep_views --separate --set sliceB
    fi
    conda deactivate
fi
conda activate psed
export PATH=$PATH:$HOME/miniconda3/envs/msproj/bin   # ffmpeg for psed_infer.wav16
python -m benchmark.sep_views --psed --set calib
[ "$MODE" = "smoke" ] || python -m benchmark.sep_views --psed --set sliceB
echo "[job] done"
