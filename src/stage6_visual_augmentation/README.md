# Stage 6 — Visual Augmentation Generation

**Role:** Render the selected augmentations as complementary visuals shown *alongside* the original
video (not replacing it). Static storyboard-style images first; short motion clips as a stretch.

**Input:** augmentation specs (image prompts + timing) from Stage 5.
**Output:** generated image(s) per selected event + a composited preview alongside the source video.

**Candidate models:** Stable Diffusion XL / FLUX.1 (quality, uni GPU); SD 1.5 / SDXL-Turbo (local
dev). Stretch: image-to-video (Stable Video Diffusion-style) on a generated keyframe.

**Status:** Stub wired into the pipeline — `generate_augmentations()` writes labelled placeholder
PNGs (PIL, no GPU); SDXL/FLUX generation and `composite_alongside()` are TODO.
**Salvage:** the alongside-video compositing can reuse ffmpeg logic from
`archive/legacy_speech_pipeline/compositor.py` (note: old code *replaced* the video; here we display
augmentations *beside/over* it — layout differs).
**TODO (Adam):** confirm acceptable FLUX.1 licence variant (`dev` = non-commercial vs `schnell` =
Apache-2.0) for an academic deliverable.
