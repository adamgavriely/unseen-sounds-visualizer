#!/bin/bash
#SBATCH --job-name=expecta3
#SBATCH --output=logs/expecta3_%j.out
#SBATCH --error=logs/expecta3_%j.err
#SBATCH --partition=L4-4h,A100-4h,H200-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#
# Round 40d EXPECT-A3 (docs/prereg_round13_detector_push.md): FineLAP places + confirms the 40c names, shipped gate. From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TG_ARMS=SHIP8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
source "$HOME/venv_flap/bin/activate"
python benchmark/gold/expect_a3_screen.py flap
deactivate
python benchmark/gold/expect_a3_screen.py cands
python benchmark/gold/expect_a3_screen.py gate
python benchmark/gold/expect_a3_screen.py score
echo "DONE expecta3"
