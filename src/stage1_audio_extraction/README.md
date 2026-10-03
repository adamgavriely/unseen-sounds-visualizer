# Stage 1 - Audio extraction

Extracts the audio track of the input video with FFmpeg as a mono WAV at the configured sample rate
(`extract_audio()`) and reads the clip duration (`media_duration()`). Main module: `__init__.py`. Needs only FFmpeg.
