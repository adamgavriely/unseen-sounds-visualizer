# Stage 3 - Speech recognition

Transcribes speech into timestamped segments with faster-whisper (`transcribe()` in `__init__.py`); returns an empty
list if faster-whisper is not installed. The transcript is context only (the on-screen check asks whether people react
to a sound); it never becomes a picture. `granite.py` is an alternative transcriber (Granite Speech) not used by the
final system.
