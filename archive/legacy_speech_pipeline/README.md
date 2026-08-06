# Legacy speech-illustration pipeline (archived 2026-08-06)

This is the **previous** project's code: illustrate the *speech* of news clips for Deaf viewers.
The project pivoted (see `docs/project_notes.tex`) to augmenting video with visuals of **non-speech
audio semantics**, gated by video understanding. This code is kept for reference and salvage; it is
**not** the current pipeline.

**Salvageable into the new structure:**
- `audio.py` → Stage 1 (FFmpeg audio extraction) — reuse directly.
- `asr.py` → Stage 3 (faster-whisper wiring) — reuse directly.
- `compositor.py` → Stage 6 (ffmpeg compositing) — reuse ffmpeg logic; layout differs (old code
  *replaced* footage, new system shows augmentations *alongside* it).
- `planner.py` → Stage 5 (prompt-structure ideas only; decision criterion is now visual-gap-aware).

**Dropped from the core (speech-illustration specific):**
- `retrieval.py` (Openverse real-image retrieval), `visualizer.py` (PIL placeholder / retrieval),
  `models.py` (Gemini client), `planner.py`'s speech-content logic, `main.py`, `config.py`.
