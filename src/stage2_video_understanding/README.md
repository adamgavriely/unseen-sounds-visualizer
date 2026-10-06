# Stage 2 - Video understanding

Samples frames and reports which candidate sound sources are visible, as a `SceneContext` (`analyze()` in
`__init__.py`). The final system uses the OWLv2 detector (`owl.py`); CLIP, SigLIP (`siglip.py`), SAM 3 (`sam3.py`) and
Qwen2.5-VL (`vlm.py`) are alternative backends, and `scene.py` classifies the setting. The final on-screen decision per
sound is made later by the vision-language model in `src/stage5_cross_modal_analysis/reason.py`.
