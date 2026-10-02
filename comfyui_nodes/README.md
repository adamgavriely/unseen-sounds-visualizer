# ComfyUI: video in, video out (frozen pipeline)

Status 2026-10-02. This top part is current. The older notes in docs/ComfyUI.md (from 23 Sept) describe the eight stage nodes,
which still use the OLD setup; they are kept only as history and sit in the menu under "MscProj/old setup".

## What you get

One workflow, `MscProj_video` (file `comfyui_nodes/MscProj_video.json`), with three boxes:

    1. Load your video  ->  2. Add sound pictures (whole pipeline)  ->  3. Your new video
                                         \-> What was drawn (a short text list)

Box 2 is the node `MscAugmentVideo`. It runs the frozen pipeline D' (tag `detector-frozen-2026-10-02`) on the new video, through
`main.py`'s code path: `config.use_shipped()`, the on-the-spot listener inputs (`src/listener_prep.py`, with FineLAP), all
stages, Qwen-Image pictures with the picture check, grouping, and the compositor. It starts
`comfyui_nodes/run_frozen.py` in a fresh process, so the ComfyUI server never holds the big models (the old nodes
were OOM-killed).

One difference from `main.py`, on purpose: stage 5 runs with the flags of the SCORED D' runs
(`benchmark/gold/round13_dev.py`: picture wording flags PICTURE_V3/SCENE/SCENE_GUARD2/FINAL/MAKER and
KINSHIP_DIRECTED off); the shipped picture flags are switched back on for the picture step, the same split as the
inspector renderer (`benchmark/gold/render_trail_media.py`). `main.py` switches them on before stage 5, which changes
the subject wording, and the duplicate-picture check reads that wording: on DEV clip tg_d088 `main.py` merged
Explosion into Thunder, while the frozen run shows both. The copy of the video is named after its content
(`comfy_<name>_<sha1 8>`), so per-clip answers from an older upload are never reused.

## How to start it

    # on the cluster (needs the BIU VPN), from ~/MscProj
    sbatch slurm/job_comfy.sh                  # H200-12h, 256G, 12 h
    grep 'ssh -N' logs/comfy_<jobid>.out       # the node name changes every job

    # on the laptop
    ssh -N -L 8188:<node>:8188 adamg@slurm-login1.lnx.biu.ac.il
    # open http://127.0.0.1:8188 -> Workflows sidebar -> MscProj_video

Then: click "choose video to upload" in box 1, press Run, wait about 10-15 minutes per short clip.
The new video is saved in ComfyUI's output folder (`~/ComfyUI/output/video/MscProj_*`) and also in
`~/MscProj/data/output/<stem>_augmented.mp4`.
