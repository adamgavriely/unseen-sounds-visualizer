# Visual Augmentation of Audio Semantics for Accessibility

MSc final project, Bar-Ilan University (Department of Computer Science), Adam Gavriely, 2026.

Subtitles carry speech, but a deaf or hard-of-hearing viewer still misses the other sounds of a video:
a siren behind the camera, a dog barking in the next room, glass breaking off screen. This system watches
a video, detects its non-speech, non-music sounds, decides for each one whether its source is already
visible, and shows a generated picture of the sound beside the video **only while an off-screen sound is
heard**. No model is trained; the system is a chain of open models joined by rules fixed on a development set.

**The technical report is [`docs/report/report.pdf`](docs/report/report.pdf).** It describes the final
system, the benchmark, the results, the error analysis and how to reproduce everything. The development
history is in its Appendix E.

## The final system

| | |
|---|---|
| Code version | release **v1.0.0** (scored code frozen under tag `detector-frozen-2026-10-02`) |
| Configuration | `config.use_shipped()` in [`config.py`](config.py) |
| Run one video | `python main.py --input clip.mp4 --device cuda` |
| Output | `data/output/<clip>_augmented.mp4` (video with a picture panel) |

Pipeline: audio extraction, frames and objects (context), speech (context) → sound detection (BEATs and
FlexSED, checked by the Qwen3-Omni and Audio Flamingo Next listeners, DASM and FineLAP) → on-screen check
(Qwen3.8-27B) → picture (Qwen-Image-2512, checked and redrawn up to five times) → side-by-side video.

## Results

![Hits and wrong pictures per system](docs/report/figures/results_bars.png)


Scored per sound against human labels (hit = right kind of sound, picture starting 0.5 s before to 1 s after
it; cost = (4 × missed + 2 × wrong pictures) / clips, lower is better).

| system | development set (71 clips) | test set (88 clips) |
|---|---|---|
| show nothing | cost 3.268 | cost 2.955 |
| September baseline | 18 / 58 found, 51 wrong, cost 3.690 | 21 / 65 found, 40 wrong, cost 2.909 |
| **final system** | **29 / 58 found, 15 wrong, cost 2.056** | **24 / 65 found, 24 wrong, cost 2.409** |

On the test set the final system is significantly cheaper than showing nothing (−0.55 per clip,
95 % interval [−1.09, −0.02], p = 0.047, `benchmark/gold/final_vs_show_nothing.py`), and the on-screen check
removes about half a wrong picture per clip (p < 0.001). The test set was read after each accepted change, so it
is a report on a seen set; see Section 9 of the report (limits and threats to validity).

## Repository layout

| path | content |
|---|---|
| `main.py`, `config.py` | entry point and configuration (`use_shipped()` = final system) |
| `src/` | the pipeline, one package per stage (`stage1_…` to `stage7_…`) |
| `benchmark/` | labels, scorer and scoring harness; see [`benchmark/gold/README.md`](benchmark/gold/README.md) |
| `slurm/` | cluster jobs; see [`slurm/README.md`](slurm/README.md) |
| `scripts/` | picture rendering, compositing and analysis utilities |
| `comfyui_nodes/` | ComfyUI wrapper of the pipeline |
| `tagger/` | stand-alone labelling tool used for the newest clips |
| `tests/` | unit tests |
| `docs/report/` | **the technical report** (LaTeX sources and PDF) |
| `docs/inspector2/` | Decision Inspector: every decision of the final system on every clip |
| `docs/history/` | development record: pre-registrations, review panels, daily notes, analyses, earlier drafts |
| `docs/supervisor/` | supervisor briefs and meeting notes |
| `docs/comfyui/`, `docs/envs/`, `docs/proposal/` | ComfyUI notes, environment files, the approved proposal |
| `archive/` | retired code (earlier speech pipeline, old helper scripts) |
| `data/` | input clips, intermediates and outputs (not in git) |

Older notes and code comments may refer to `docs/<file>.md`; those files now live under `docs/history/`
(see [`docs/README.md`](docs/README.md)).

## Environment

- Conda environment `msproj` (PyTorch 2.5.1, transformers 5.16.1, diffusers, faster-whisper,
  panns-inference, easyocr, sentence-transformers); FineLAP runs in a separate `~/venv_flap`
  (transformers 4.51.3). Details in Appendix C of the report.
- GPU work runs on the BIU SLURM cluster (H200 / A100). Qwen-Image needs about 57 GB in bf16.
- FFmpeg on the path.

## Open items at submission

See [`TODO.md`](TODO.md) and [`LIMITATIONS.md`](LIMITATIONS.md).
