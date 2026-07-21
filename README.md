# Audio-to-Illustrated-Video for Deaf Viewers

MSc project. Given a video (e.g. news), extract the audio and generate visual
content that depicts the meaning of the speech, time-aligned, so a Deaf viewer
can *see* the story rather than hear it.

## Pipeline

```
video.mp4
  -> [1] extract audio        (ffmpeg)            src/audio.py
  -> [2] ASR / segments       (faster-whisper)    src/asr.py
  -> [3] planner              (LLM: what/whether)  src/planner.py
  -> [4] visualizer           (diffusion / retrieval / placeholder)  src/visualizer.py
  -> [5] compositor           (ffmpeg, synced)    src/compositor.py
  -> output.mp4
```

Each stage has pluggable backends so implementations can be swapped without
touching the rest.

## Setup

```powershell
conda create -n msproj -c conda-forge python=3.11 -y
conda activate msproj
pip install -r requirements.txt
```

ffmpeg must be on PATH (already installed at C:\ffmpeg\bin).

## Run

Drop a video in `data/input/`, then:

```powershell
conda activate msproj

# Full real pipeline: Gemini planner + free image retrieval (needs GEMINI_API_KEY in .env)
python main.py --input data/input/clip.mp4 --planner llm --visualizer retrieve

# Offline smoke test (no key/network): rule planner + placeholder frames
python main.py --input data/input/clip.mp4
```

Output lands in `data/output/`. Intermediates (audio, `segments.json`,
`plans.json`, `images/`, `credits.json`) are under `data/work/<clip-name>/`.

`--planner`: `rule` (stub) | `llm` (Gemini, the real component).
`--visualizer`: `placeholder` | `retrieve` (Openverse) | `diffusion` (TODO, uni GPU).

## Roadmap

- [x] LLM planner (`--planner llm`) — Gemini, `GEMINI_API_KEY` in `.env`
- [x] Image retrieval backend (`--visualizer retrieve`) — Openverse, free
- [ ] Diffusion visualizer (`--visualizer diffusion`) — on the university GPU
- [ ] Better retrieval queries + generate-vs-retrieve routing in the visualizer
- [ ] Evaluation (visualizability accuracy; retrieve vs generate relevance)
- [ ] Ken Burns motion + optional image-to-video (SVD)
```
