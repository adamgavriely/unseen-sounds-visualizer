#!/bin/bash
#SBATCH --job-name=finelap
#SBATCH --output=logs/finelap_%j.out
#SBATCH --error=logs/finelap_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#
# Round 31 FLAP (docs/prereg_round13_detector_push.md): FineLAP frame scores on merged-DEV listener wavs, then the screen.
# Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source "$HOME/venv_flap/bin/activate"
python benchmark/gold/finelap_screen.py run
deactivate
python benchmark/gold/finelap_screen.py score
echo "DONE finelap"
