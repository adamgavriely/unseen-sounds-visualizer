# Visual Augmentation of Audio Semantics for Accessibility

MSc final project, Bar-Ilan University (Department of Computer Science), Adam Gavriely, 2026.

Subtitles carry speech, but a deaf or hard-of-hearing viewer still misses the other sounds of a video:
a siren behind the camera, a dog barking in the next room, glass breaking off screen. This system watches
a video, detects its non-speech, non-music sounds, decides for each one whether its source is already
visible, and shows a generated picture of the sound beside the video **only while an off-screen sound is
heard**. No model is trained; the system is a chain of open models joined by rules fixed on a development set.

**The technical report is [`docs/report/report.pdf`](docs/report/report.pdf).** It describes the system, the
benchmark, the results, why the problem is hard for today's models, and how to reproduce everything. Code version:
release **v1.2.0**.

## Usage

```
python main.py --input clip.mp4 --device cuda
```

This writes `data/output/clip_augmented.mp4` (the video with the picture panel). The final system is the
configuration `config.use_shipped()` in [`config.py`](config.py). To reproduce the reported numbers:

```
python benchmark/gold/final_vs_baselines.py
```

(details and the exact-reproduction check are in Appendix C of the report). Optional: `comfyui_nodes/` wraps the
same pipeline as ComfyUI nodes.

Pipeline: audio extraction, frames and objects (context), speech (context) → sound detection (BEATs and
FlexSED, checked by the Qwen3-Omni and Audio Flamingo Next listeners, DASM and FineLAP) → on-screen check
(Qwen3.8-27B) → picture (Qwen-Image-2512, checked and redrawn up to five times) → side-by-side video.

## Results

![Hits and wrong pictures per system](docs/report/figures/results_bars.png)

Scored per sound against human labels (hit = right kind of sound, picture starting 0.5 s before to 1 s after
it; cost = (4 × missed + 2 × wrong pictures) / clips, lower is better). The baselines are those of the project
proposal.

| system | development set (71 clips) | test set (88 clips) |
|---|---|---|
| show nothing | cost 3.268 | cost 2.955 |
| direct audio-to-image (and audio captioning) | 34 / 58 found, 39 wrong, cost 2.451 | 26 / 65 found, 62 wrong, cost 3.182 |
| **final system** | **29 / 58 found, 15 wrong, cost 2.056** | **24 / 65 found, 24 wrong, cost 2.409** |

On the test set the final system is significantly cheaper than direct audio-to-image generation
(−0.773 per clip, 95 % interval [−1.045, −0.500], p < 0.001) and than showing nothing (−0.545, [−1.068, −0.023],
p = 0.041; p = 0.083 after a Holm correction over the four comparisons of the final system). Every decision was made on the development set; the test set was only scored. Limits are in Section 9 of the
report.

## Repository layout

| path | content |
|---|---|
| `main.py`, `config.py` | entry point and configuration (`use_shipped()` = final system) |
| `src/` | the pipeline, one package per stage (`stage1_…` to `stage7_…`) |
| `benchmark/` | labels, scorer and scoring harness; see [`benchmark/gold/README.md`](benchmark/gold/README.md) |
| `scripts/`, `slurm/` | rendering and analysis utilities; batch job wrappers ([`slurm/README.md`](slurm/README.md)) |
| `tagger/` | stand-alone labelling tool used for the newest clips |
| `tests/` | unit tests |
| `docs/report/` | **the technical report** (LaTeX sources, figures and PDF) |
| `docs/inspector2/` | decision trail: every decision of the final system on every clip |
| `docs/history/` | development record: pre-registrations, review panels, daily notes, analyses, earlier drafts |
| `docs/supervisor/`, `docs/proposal/`, `docs/envs/` | supervisor material, the approved proposal, environment files |
| `comfyui_nodes/`, `docs/comfyui/` | optional ComfyUI interface |
| `archive/` | retired code |
| `data/` | input clips, intermediates and outputs (not in git) |

## Environment

- Python 3 with PyTorch 2.5.1, transformers 5.16.1, diffusers, faster-whisper, panns-inference, easyocr and
  sentence-transformers; FineLAP runs in a second environment with transformers 4.51.3 (`FINELAP_PYTHON`).
- One GPU with about 80 GB of memory (an NVIDIA H200 was used). Qwen-Image needs about 57 GB in bf16.
- FFmpeg on the path.

## Citation

See [`CITATION.cff`](CITATION.cff).
