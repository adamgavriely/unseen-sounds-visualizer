# Visual Augmentation of Audio Semantics for Accessibility

MSc final project, Bar-Ilan University (Department of Computer Science), Adam Gavriely, 2026.

Subtitles carry speech, but a deaf or hard-of-hearing viewer still misses the other sounds of a video: a siren
behind the camera, a dog barking in the next room, glass breaking off screen. This system watches a video, detects
its non-speech, non-music sounds, decides for each one whether its source is already visible, and shows a generated
picture of the sound beside the video **only while an off-screen sound is heard**. No model is trained: the system
is a chain of open models joined by rules fixed on a development set.

The full description, the benchmark, the results and the limits are in the technical report,
[`docs/report/report.pdf`](docs/report/report.pdf).

## Results

![Hits and wrong pictures per system](docs/report/figures/results_bars.png)

Scored per sound against human labels. A hit is the right kind of sound with a picture starting between 0.5 s
before and 1 s after it; cost = (4 × missed + 2 × wrong pictures) / clips, lower is better. The baselines are those
of the project proposal.

| system | development set (71 clips) | test set (88 clips) |
|---|---|---|
| show nothing | cost 3.268 | cost 2.955 |
| direct audio-to-image (and audio captioning) | 34 / 58 found, 39 wrong, cost 2.451 | 26 / 65 found, 62 wrong, cost 3.182 |
| **final system** | **29 / 58 found, 15 wrong, cost 2.056** | **24 / 65 found, 24 wrong, cost 2.409** |

On the test set the final system is significantly cheaper than direct audio-to-image generation (−0.773 per clip,
95 % interval [−1.045, −0.500], p < 0.001) and than showing nothing (−0.545, [−1.091, −0.023], p = 0.044; p = 0.087
after a Holm correction over the five main tests). It finds 10 of the 13 danger sounds of the test set (sirens,
alarms, breaking glass, a crying baby). Every decision was made on the development set; the test set was only
scored. Limits are discussed in Section 9 of the report.

## How it works

1. **Audio and context**: extract the audio track, sample frames, detect visible objects and transcribe speech.
2. **Sound detection**: BEATs and FlexSED propose sound events; the Qwen3-Omni and Audio Flamingo Next listeners,
   DASM and FineLAP confirm or reject them.
3. **On-screen check**: Qwen3.8-27B looks at the frames and decides whether each sound's source is visible.
4. **Pictures**: Qwen-Image-2512 draws each off-screen sound; a vision model checks the picture and it is redrawn
   up to five times.
5. **Composition**: the pictures are shown beside the video while their sound plays.

## Installation

```
pip install -r requirements.txt
```

- Python 3 with PyTorch 2.5.1, transformers 5.16.1, diffusers, faster-whisper, panns-inference, easyocr and
  sentence-transformers. FineLAP needs transformers 4.51.3 and runs in a second environment, set with the
  `FINELAP_PYTHON` variable. FlexSED and DASM are imported from their own repositories (paths in `config.py`).
- The models are downloaded from Hugging Face on first use.
- One GPU with about 80 GB of memory (an NVIDIA H200 was used; two A100 80 GB cards also work). Qwen-Image alone
  needs about 57 GB in bf16.
- FFmpeg on the path.

## Usage

```
python main.py --input clip.mp4 --device cuda
```

This writes `data/output/clip_augmented.mp4` (the video with the picture panel) and the intermediate files under
`data/work/clip/`. The final system is the configuration `config.use_shipped()` in [`config.py`](config.py).
On a Slurm cluster, `slurm/run_best.sh` runs the same system on a folder of clips (see
[`slurm/README.md`](slurm/README.md)).

## Reproducing the reported numbers

```
python benchmark/gold/final_vs_baselines.py
```

The scoring harness reads the human labels and the cached model outputs of each clip and replays the pipeline's own
decision functions. The exact commands, the configuration check and the source file of every number are in
Appendix C of the report; [`benchmark/gold/README.md`](benchmark/gold/README.md) lists the files involved. The video
clips are not redistributed.

## Repository layout

| path | content |
|---|---|
| `main.py`, `config.py` | entry point and configuration (`use_shipped()` = final system) |
| `src/` | the pipeline, one package per stage (`stage1_…` to `stage7_…`) |
| `benchmark/` | human labels, per-sound scorer, scoring harness and cached model answers |
| `tests/` | unit tests |
| `docs/report/` | the technical report (LaTeX sources, figures and PDF) |
| `docs/inspector2/` | decision trail: every decision of the final system on every clip, and its parity check |
| `slurm/` | cluster job scripts for running the system on a folder of clips |
| `comfyui_nodes/` | optional ComfyUI interface ([`comfyui_nodes/README.md`](comfyui_nodes/README.md)) |
| `data/input/` | a small test clip |

## Citation

See [`CITATION.cff`](CITATION.cff).
