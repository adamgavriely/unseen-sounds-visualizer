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

## Run (walking skeleton — no keys / GPU needed)

Drop a video in `data/input/`, then:

```powershell
conda activate msproj
python main.py --input data/input/clip.mp4
```

Output lands in `data/output/`. This uses the placeholder visualizer (caption on
a coloured frame) and the rule-based planner, to validate timing and sync.

## Roadmap

- [ ] LLM planner (`--planner llm`) — needs `ANTHROPIC_API_KEY` + `pip install anthropic`
- [ ] Diffusion visualizer (`--visualizer diffusion`) — on the university GPU
- [ ] Image retrieval backend (comparison / hallucination mitigation)
- [ ] Ken Burns motion + optional image-to-video (SVD)
- [ ] Evaluation
```
