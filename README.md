# Visual Augmentation of Audio Semantics for Accessibility

MSc final project. A system that makes the **non-speech** information in a video's soundtrack
accessible to deaf and hard-of-hearing (DHH) viewers. Subtitles and sign language convey speech but
largely drop ambient and environmental sound — flowing water, footsteps, a distant siren, rain,
birds, a reacting crowd. This system detects that missing audio semantics and renders it as
**complementary visual augmentations displayed alongside the original video** (supplementing, not
replacing, the video or its captions).

**Core idea — cross-modal gating:** the *video* is also analyzed, so the system augments only what
the audio conveys but the picture does **not** already show (a siren whose source is off-screen is
worth showing; one already on screen is not).

No models are trained — the contribution is the task formulation, an inference pipeline that
orchestrates open-source foundation models, a curated benchmark, and an automatic evaluation
protocol.

> Full proposal: `Final_Project.pdf`. Living research/design record: **[`docs/project_notes.tex`](docs/project_notes.tex)**.

## Pipeline (7 stages)

| # | Stage | Role | Candidate models |
|---|-------|------|------------------|
| 1 | [Audio extraction](src/stage1_audio_extraction) | standardize audio | FFmpeg |
| 2 | [Video understanding](src/stage2_video_understanding) | what's already visible | Qwen2.5-VL / InternVL / LLaVA |
| 3 | [Speech recognition](src/stage3_speech_recognition) | transcribe speech (secondary) | Whisper Large V3 / Qwen2-Audio |
| 4 | [Audio event detection](src/stage4_audio_event_detection) | non-speech sounds | PANNs / BEATs |
| 5 | [Cross-modal analysis](src/stage5_cross_modal_analysis) | **what to augment (gap-aware)** | Qwen3 / Llama 3.1 |
| 6 | [Visual augmentation](src/stage6_visual_augmentation) | generate the visuals | FLUX.1 / SDXL |
| 7 | [Evaluation](src/stage7_evaluation) | VLM-describe → LLM-judge + CLIP/CLAP | — |

Each stage is a self-contained module with its own README (purpose, I/O, models, status).

## Status

**Runnable skeleton.** The 7-stage pipeline is wired end-to-end and runs with no GPU or downloaded
models:

```bash
python main.py --input data/input/clip.mp4
```

Stages **1 (ffmpeg audio)** and **3 (whisper ASR)** are real; **2/4/5/6** are typed stubs that pass
placeholder data through so the architecture is exercised (Stage 5 runs a transparent rule-based
gate; Stage 6 writes placeholder images); **7** has the core `gating_accuracy()` metric, with the
VLM-describe → LLM-judge protocol still TODO. Artifacts land in `data/work/<stem>/`
(`media.json`, `scene.json`, `segments.json`, `events.json`, `augmentations.json`, `augmentations/`).

Research and design (literature review, model/dataset recommendations, evaluation design, 8-week
plan, tasks) are in [`docs/project_notes.tex`](docs/project_notes.tex).

The previous speech-illustration pipeline is preserved under
[`archive/legacy_speech_pipeline/`](archive/legacy_speech_pipeline) — salvageable pieces (FFmpeg
audio, Whisper ASR, ffmpeg compositing) are labelled there.

## Repository layout

```
docs/project_notes.tex        # living knowledge base: lit review, plan, decisions, tasks
src/stage1..stage7/           # 7-stage pipeline modules (design stubs for now)
benchmark/                    # curated ~300-clip evaluation set (not started)
data/input|work|output/       # clips and intermediates (gitignored except sample)
archive/legacy_speech_pipeline/  # previous project's code, preserved
Final_Project.pdf             # the approved proposal
```

## Environment

- conda env `msproj` (Python 3.11); interpreter
  `C:\Users\adamg\anaconda3\envs\msproj\python.exe`.
- FFmpeg at `C:\ffmpeg\bin`.
- Dev GPU: GTX 1060 6 GB (local); university GPU requested for SDXL/FLUX and 7B-VLM quality runs.
