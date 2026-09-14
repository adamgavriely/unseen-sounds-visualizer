#!/bin/bash
#SBATCH --job-name=clap_k
#SBATCH --output=logs/clap_k_%j.out
#SBATCH --error=logs/clap_k_%j.err
#SBATCH --partition=L4-12h,H200-12h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#
# CLAP second-opinion: calibrate k on DCASE gold, rank dev/test detections for one split (benchmark/gate_dev_sweep.py
# --cache). python - <<'PY'
import sys, config
config.DEVICE = "cuda"
sys.argv = ["calibrate_clap_k", "--dcase", "--ranks", "dev", "--ranks", "test"]
from benchmark import calibrate_clap_k as K
K.main()
PY
echo DONE
