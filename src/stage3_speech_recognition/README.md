# Stage 3 — Speech Recognition

**Role:** Transcribe any spoken language in the audio. Speech is now a *secondary* signal (the star
is non-speech sound), used for context in cross-modal analysis and to avoid augmenting content the
subtitles already cover.

**Input:** `audio.wav` from Stage 1.
**Output:** timestamped transcript segments.

**Candidate models:** Whisper Large V3 (quality) / faster-whisper small–medium (local); Qwen2-Audio
as an alternative that also covers Stage 4 (see LALM note in `docs/project_notes.tex`).

**Status:** Not implemented (research phase).
**Salvage:** reuse `archive/legacy_speech_pipeline/asr.py` — the faster-whisper wiring carries over.
