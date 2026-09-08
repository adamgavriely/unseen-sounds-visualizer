#!/bin/bash
# Submit the whole remaining experiment as one dependency chain, then walk away.
#
#   usage:  bash slurm/submit_chain.sh [jobid-to-wait-for]
#
# With an argument, the chain starts only after that job succeeds -- use it to hang
# the chain off a pilot that is still running.
#
#   A   pilot, gate, then the 100-clip SDXL main run                      (12 h)
#   A2  continuation of A, in case A hits the 12 h wall; exits in seconds if
#       A finished, since every stage is stamped and every cache is per clip
#   B   judge reliability: a second, unrelated judge re-scores A's descriptions
#   C   ablation: the same clips with retrieval instead of SDXL
#
# B and C depend on A2 succeeding, and are submitted with --kill-on-invalid-dep=yes,
# so a failed gate cancels them instead of leaving them pending for days.

set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs benchmark/.chain

DEP=""
[ $# -ge 1 ] && DEP="--dependency=afterok:$1" && echo "chain waits for job $1"

A=$(sbatch --parsable $DEP --export=LIMIT="${LIMIT:-100}" slurm/job_main.sh)
echo "A  main run            $A"

A2=$(sbatch --parsable --dependency=afterany:"$A" \
     --export=LIMIT="${LIMIT:-100}" slurm/job_main.sh)
echo "A2 continuation        $A2"

# Rendering 300 clips with SDXL, describing them and judging them runs close to the
# 12 h wall, so a second continuation is cheap insurance; it exits in seconds when
# there is nothing left to do.
A3=$(sbatch --parsable --dependency=afterany:"$A2" \
     --export=LIMIT="${LIMIT:-100}" slurm/job_main.sh)
echo "A3 continuation        $A3"

B=$(sbatch --parsable --dependency=afterok:"$A3" --kill-on-invalid-dep=yes \
    slurm/job_judge2.sh)
echo "B  judge reliability   $B"

C=$(sbatch --parsable --dependency=afterok:"$A3" --kill-on-invalid-dep=yes \
    --export=LIMIT="${LIMIT:-100}" slurm/job_ablation_gen.sh)
echo "C  generator ablation  $C"

echo
echo "watch:  squeue -u \$USER"
echo "read :  tail -f logs/main_${A}.out"
