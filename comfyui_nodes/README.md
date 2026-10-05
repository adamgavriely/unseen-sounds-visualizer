# ComfyUI nodes: video in, video out

An optional graphical interface. You load a video in ComfyUI, press Run, and get the same video back with the
pictures of the off-screen sounds beside it, plus a short text list of what was drawn.

## What it does

The workflow `MscProj_video` (file `comfyui_nodes/MscProj_video.json`) has three boxes:

    1. Load your video  ->  2. Add sound pictures (whole pipeline)  ->  3. Your new video
                                         \-> What was drawn (a short text list)

Box 2 is the node `MscAugmentVideo`. It runs the final system (`config.use_shipped()`) on the uploaded video through
the same code path as `main.py`: the listener inputs are prepared on the spot, then sound detection, the on-screen
check, the pictures and the composition. The node starts `comfyui_nodes/run_frozen.py` in a separate process, so the
ComfyUI server never holds the large models. The only difference from `main.py`: the on-screen check runs with the
settings of the scored runs (`benchmark/gold/dev_harness.py`, variant `SHIP8+MD3+WW5+SL`), and the final picture
settings are used only for the drawing step.

## Install

1. Install ComfyUI in the same Python environment as this project (see the main `README.md`).
2. Link or copy this folder into `ComfyUI/custom_nodes/`.
3. If the folder is not inside the project, set `MSCPROJ_ROOT` to the project folder.
4. Optional: set `COMFYUI_ROOT` if ComfyUI is not in `~/ComfyUI` (used by `comfy_start.py`).

## Run

    python comfyui_nodes/comfy_start.py --listen 0.0.0.0 --port 8188

Open http://127.0.0.1:8188, load the workflow `MscProj_video` from the Workflows sidebar, choose a video in box 1
and press Run. `comfy_start.py` starts ComfyUI with a small fix that lets it run on this project's PyTorch 2.5.1. The
new video is saved in ComfyUI's output folder and in `data/output/<name>_augmented.mp4`.

`public_page.py` is a simple upload page (Gradio) on top of a running ComfyUI server: upload a video, press one
button, get the new video. It needs `gradio`, `requests` and `websockets`:

    python comfyui_nodes/public_page.py --comfy http://127.0.0.1:8188 --port 7860

With `--share` the page gets a temporary public gradio.live link.

## Caveat

Run time, as for `main.py`: about five minutes of GPU time (one NVIDIA H200) to prepare the listener inputs of a
15-s clip, plus the picture step (about 20 s per picture try, up to five tries per picture), because every model
input is computed for the new video. Videos without sound, without picture or shorter than 1 s are refused; other formats
(.mov, .webm) are converted to .mp4 first.
