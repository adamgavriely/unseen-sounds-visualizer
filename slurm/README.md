# slurm — cluster jobs

Job scripts for running the system on a Slurm cluster with one large GPU (H200 or A100 80 GB).

| script | purpose |
|---|---|
| `setup_env.sh` | one-time build of the conda environment `msproj` (run on a login node) |
| `sync_data.sh` | upload the benchmark clips (not in git) from a local machine to the cluster |
| `job_smoke.sh` | short check that the GPU and the environment work |
| `run_best.sh` | run the final system on a folder of clips: `sbatch slurm/run_best.sh NAME /path/to/clips` |
| `job_comfy.sh` | start the optional ComfyUI interface (see `comfyui_nodes/README.md`) |

`run_best.sh` builds the per-clip model inputs with `benchmark/gold/tagger_prep.py` (FlexSED, BEATs, PANNs, the
listeners, DASM and FineLAP), then runs detection and the on-screen check of the final system, and writes the
result of each clip under `data/work/r13<NAME>/`. It ends with the grouping and depiction answers
(`python -m src.stage6_visual_augmentation.group` and `... .depict`) that the picture step reads. Job output goes
to `logs/` (not in git).
