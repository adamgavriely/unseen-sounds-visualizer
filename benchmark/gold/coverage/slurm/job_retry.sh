#!/bin/bash
#SBATCH --job-name=s6retry
#SBATCH --partition=H200-4h,A100-4h
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:55:00
#SBATCH --output=logs/s6_retry_%j.out
# Step 6 A: mix2 (CPU part, resumes per shard) -> fine-tune retry -> ear caches -> DEV-bar tier 1.
set -euo pipefail
cd ~/MscProj_tg
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
export PYTHONUNBUFFERED=1 MIX_STYLE=video
python benchmark/gold/coverage/build_mixtures.py 20000 1000
conda deactivate
export PATH=~/miniconda3/envs/sota/bin:$PATH FT_MIX=mix2 FT_LR_BACKBONE=3e-6 FT_LR_HEAD=3e-5 FT_TEACHER_W=0.7
~/venvs/psed2/bin/python benchmark/gold/coverage/train_finetune.py ATST-F atstf_ft_retry 0
E=$(python3 -c "import json;print(json.load(open('$HOME/open_data/ft/atstf_ft_retry/log.json'))['best_epoch'])")
for s in dev heldout; do
  ~/venvs/psed2/bin/python benchmark/gold/coverage/ear_cache.py atstf_ft_retry ATST-F ~/open_data/ft/atstf_ft_retry/epoch_$E.pt --set $s
done
source "$HOME/miniconda3/etc/profile.d/conda.sh"; conda activate msproj
cd ~/wt_slice
python benchmark/gold/coverage/ear_tier1_devbar.py ~/MscProj_tg/scratch_cov/ears/atstf_orig ~/MscProj_tg/scratch_cov/ears/atstf_ft_s0 ~/MscProj_tg/scratch_cov/ears/atstf_ft_retry
echo "best epoch $E"; echo S6_RETRY_DONE
