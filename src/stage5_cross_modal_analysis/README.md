# Stage 5 - Cross-modal analysis

Decides, for each detected sound, whether its source is already on screen and, if not, what to draw.
`plan_augmentations()` in `__init__.py` is the rule-based first pass (drawable-label filter, confidence bars, one
entry per sound family); `reason.py` (`decide_subjects()`) asks the vision-language model (Qwen3.8-27B in the final
system) about the frames around each sound and writes the picture subject. `nameall.py` is an optional
name-and-crop check called from `reason.py`; it is off in the final system.
