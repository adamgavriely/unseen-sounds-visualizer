#!/bin/bash
#SBATCH --job-name=dasmres
#SBATCH --output=logs/dasmres_%j.out
#SBATCH --error=logs/dasmres_%j.err
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=04:00:00
#
# Round 19 DR: P4 pools (DASM runs no B0r span touches) + Qwen V4 + AF V4 answers, all four parts (gold-free). From ~/MscProj_tg.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate msproj
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
W=$HOME/MscProj/data/work; G=benchmark/gold; P=benchmark/gold/dasm_rescue.py
python $P pool dev  $W/r13/stage4.json      $W/devcand/dasm_cache $W/devcand/wav16 $G/dev_listener_p4.json
python $P pool dev2 $W/r13dev2/stage4.json  $W/dasm_dev2          $W/r13dev2/wav16 $G/dev2_listener_p4.json
python $P pool test $W/r16final/stage4.json $W/dasm_test          $W/r13test/wav16 $G/test_listener_p4.json
python $P pool test2 $W/r13test2/stage4.json $W/dasm_test2        $W/r13test2/wav16 $G/test2_listener_p4.json
python $P listen $G/dev_listener_p4.json $G/dev2_listener_p4.json $G/test_listener_p4.json $G/test2_listener_p4.json
echo "DONE dasmres"
