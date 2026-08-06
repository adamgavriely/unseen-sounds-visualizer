# Stage 1 — Audio Extraction

**Role:** Extract the audio track from the input clip and standardize it (mono, fixed sample rate, WAV) for all downstream audio stages.

**Input:** video clip (`.mp4`/`.webm`/…) with an audio stream.
**Output:** standardized `audio.wav` (+ probe metadata: duration, sample rate).

**Tooling:** FFmpeg (already installed at `C:\ffmpeg\bin`).

**Status:** Implemented (skeleton) — `extract_audio()` returns a `MediaInfo`; salvaged from
`archive/legacy_speech_pipeline/audio.py`. Runs with only ffmpeg on PATH.
