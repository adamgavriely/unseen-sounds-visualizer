#!/bin/bash
#SBATCH --job-name=flapfull
#SBATCH --output=logs/flapfull_%j.out
#SBATCH --error=logs/flapfull_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#
# Round 32 (docs/prereg_round13_detector_push.md): FineLAP cache 2 (every family the vetoes ask about), the DASM-layout
# copy, and the P1 clip-bar calibration. Submit from ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source "$HOME/venv_flap/bin/activate"
python benchmark/gold/finelap_full.py run
deactivate
python benchmark/gold/finelap_full.py build
python benchmark/gold/finelap_full.py calib
echo "DONE flapfull"
