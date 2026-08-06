# Stage 1 — Audio Extraction

**Role:** Extract the audio track from the input clip and standardize it (mono, fixed sample rate, WAV) for all downstream audio stages.

**Input:** video clip (`.mp4`/`.webm`/…) with an audio stream.
**Output:** standardized `audio.wav` (+ probe metadata: duration, sample rate).

**Tooling:** FFmpeg (already installed at `C:\ffmpeg\bin`).

**Status:** Not implemented (research phase).
**Salvage:** reuse `archive/legacy_speech_pipeline/audio.py` — the FFmpeg extraction logic carries over directly.
