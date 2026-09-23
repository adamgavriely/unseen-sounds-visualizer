"""The pipeline as a ComfyUI graph — seven nodes, one per stage.

This is a DEMO artefact, not a second implementation. Every node is a thin wrapper whose body calls
the same function `src/pipeline.py` calls; no decision, threshold or model choice lives here. The
numbers in docs/prereg_v4.md come from `benchmark/run_protocol.py` and must keep coming from there.

What the graph buys:
  * the seven stages visible as boxes, with the cross-modal gate as a switch the viewer can flip;
  * only the stages downstream of a changed knob re-run, because ComfyUI caches node outputs;
  * any video the user drops in, not only the benchmark clips.

Install: symlink or copy this folder into ComfyUI/custom_nodes/ and set MSCPROJ_ROOT to the project
root if it is not the parent of this file.

Memory: each node frees its model before returning, which is how the existing pipeline already
works (`reason.unload()`, `beats_infer.unload()`, `unload_generator()`). Peak is one model, not four.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_ROOT = Path(os.environ.get("MSCPROJ_ROOT", Path(__file__).resolve().parent.parent))
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config  # noqa: E402


def _work_dir(stem: str) -> Path:
    d = Path(config.WORK_DIR) / "comfy" / stem
    d.mkdir(parents=True, exist_ok=True)
    return d


def _free():
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# --------------------------------------------------------------------------- 1. the video
class MscLoadVideo:
    """Stage 1 — take any video, pull its audio out. The only node that touches the filesystem."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video_path": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("MSC_MEDIA",)
    RETURN_NAMES = ("media",)
    FUNCTION = "run"
    CATEGORY = "MscProj"

    def run(self, video_path):
        from src.stage1_audio_extraction import extract_audio
        p = Path(video_path).expanduser().resolve()
        if not p.exists():
            raise RuntimeError(f"video not found: {p}")
        work = _work_dir(p.stem)
        media = extract_audio(p, work / "audio.wav", config.SAMPLE_RATE)
        return ({"video_path": str(p), "stem": p.stem, "work": str(work), "media": media},)


# --------------------------------------------------------------------------- 2. what is on screen
class MscSceneUnderstanding:
    """Stage 2 — what is visible. Queried with the project's concept list, six frames over the clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media": ("MSC_MEDIA",),
            "backend": (["owlv2", "siglip", "clip"], {"default": "owlv2"}),
            "num_frames": ("INT", {"default": 6, "min": 1, "max": 32}),
        }}

    RETURN_TYPES = ("MSC_SCENE", "STRING")
    RETURN_NAMES = ("scene", "visible_entities")
    FUNCTION = "run"
    CATEGORY = "MscProj"

    def run(self, media, backend, num_frames):
        from src.stage2_video_understanding import analyze
        scene = analyze(media["video_path"], backend=backend, num_frames=num_frames,
                        model=config.VIDEO_MODEL, vlm_model=config.VLM_MODEL,
                        siglip_model=config.SIGLIP_MODEL, owl_model=config.OWL_MODEL,
                        owl_threshold=config.OWL_THRESHOLD,
                        siglip_threshold=config.SIGLIP_THRESHOLD,
                        device=config.DEVICE, threshold=config.VISIBILITY_THRESHOLD)
        _free()
        return (scene, ", ".join(scene.visible_entities) or "none")


# --------------------------------------------------------------------------- 3. speech
class MscTranscribe:
    """Stage 3 — the transcript. It never reaches a picture; it is here for the panel and the judge."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"media": ("MSC_MEDIA",),
                             "enabled": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MSC_SEGMENTS", "STRING")
    RETURN_NAMES = ("segments", "transcript")
    FUNCTION = "run"
    CATEGORY = "MscProj"

    def run(self, media, enabled):
        if not enabled:
            return ([], "")
        from src.stage3_speech_recognition import transcribe
        segs = transcribe(Path(media["media"].wav_path), model_size=config.WHISPER_MODEL,
                          device=config.DEVICE, compute_type=config.WHISPER_COMPUTE)
        _free()
        return (segs, " ".join(getattr(s, "text", "") for s in segs)[:2000])


# --------------------------------------------------------------------------- 4. what was heard
class MscDetectEvents:
    """Stage 4 — the sounds, and the two vetoes that were adopted on 2026-09-23.

    FlexSED is read from a cache keyed by clip name. The benchmark clips have one; a video the user
    just dropped in does not, so it is computed here in a subprocess (the FlexSED repo needs its own
    sys.path and working directory, which would be destructive inside a long-running server).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media": ("MSC_MEDIA",),
            "flexsed_bar": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.05}),
            "cross_detector_veto": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05}),
            "panns_veto": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01}),
        }}

    RETURN_TYPES = ("MSC_EVENTS", "STRING")
    RETURN_NAMES = ("events", "heard")
    FUNCTION = "run"
    CATEGORY = "MscProj"

    def run(self, media, flexsed_bar, cross_detector_veto, panns_veto):
        from src.stage4_audio_event_detection import detect_events
        from src.stage4_audio_event_detection import flexsed_infer as FX
        wav = Path(media["media"].wav_path)
        if flexsed_bar > 0 and not FX.cache_path(wav.parent.name).exists():
            # the demo's one piece of real work for an unseen video
            subprocess.run([sys.executable, str(_ROOT / "comfyui_nodes" / "flexsed_worker.py"),
                            "--wav", str(wav), "--stem", wav.parent.name], check=False)
        config.FLEXSED_BAR = float(flexsed_bar)
        config.FLEXSED_VETO = float(cross_detector_veto)
        config.PANNS_VETO = float(panns_veto)
        events = detect_events(wav, threshold=config.AED_THRESHOLD, min_dur=config.AED_MIN_DUR,
                               model=config.AED_MODEL, device=config.DEVICE)
        try:
            from src.stage4_audio_event_detection import beats_infer
            beats_infer.unload()
        except Exception:
            pass
        _free()
        heard = ", ".join(sorted({e.label for e in events})) or "nothing"
        return (events, heard)


# --------------------------------------------------------------------------- 5. the gate
class MscCrossModalGate:
    """Stage 5 — THE contribution: draw a sound only when its source is NOT on screen.

    `gate_enabled` is the demo. Turn it off and the system becomes the blind baseline: every
    detected sound gets a picture, including the ones the viewer can already see.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media": ("MSC_MEDIA",),
            "scene": ("MSC_SCENE",),
            "segments": ("MSC_SEGMENTS",),
            "events": ("MSC_EVENTS",),
            "gate_enabled": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("MSC_SPECS", "STRING")
    RETURN_NAMES = ("specs", "decisions")
    FUNCTION = "run"
    CATEGORY = "MscProj"

    def run(self, media, scene, segments, events, gate_enabled):
        from src.stage5_cross_modal_analysis import plan_augmentations, reason
        # Turning the switch off must produce the SAME system the thesis calls "blind", or the
        # demo would be showing a comparison nobody measured. benchmark/run_protocol.py sets four
        # flags for that arm, not one: the rule-based gate, the visibility question asked of the
        # VLM, the depiction reasoning, and the speech context. gate_enabled only covered the
        # first, which is why an earlier run produced two identical videos.
        config.GATE_ENABLED = bool(gate_enabled)
        config.VLM_VISIBILITY = bool(gate_enabled)
        config.DEPICTION_REASONING = bool(gate_enabled)
        config.SPEECH_CONTEXT = bool(gate_enabled)
        config.RENDER_MODE = "side" if gate_enabled else "full"
        specs = plan_augmentations(scene, segments, events,
                                   threshold=config.AED_THRESHOLD,
                                   gate_enabled=bool(gate_enabled),
                                   display_threshold=config.DISPLAY_THRESHOLD,
                                   augment_threshold=config.AUGMENT_THRESHOLD)
        if getattr(config, "DEPICTION_REASONING", True) and any(s.augment for s in specs):
            try:
                reason.decide_subjects(media["video_path"], specs, segments=segments,
                                       model=config.VLM_MODEL, device=config.DEVICE,
                                       display_threshold=config.DISPLAY_THRESHOLD)
            finally:
                reason.unload()
        _free()
        lines = [f"{'DRAW' if s.augment else 'skip'}  {s.event_label}  {s.start:.1f}-{s.end:.1f}s"
                 f"  {getattr(s, 'reason', '')[:70]}" for s in specs]
        return (specs, "\n".join(lines) or "no sounds")


# --------------------------------------------------------------------------- 6. the pictures
class MscGeneratePictures:
    """Stage 6a — make the picture for each sound the gate let through."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media": ("MSC_MEDIA",),
            "specs": ("MSC_SPECS",),
            "backend": (["diffusion", "placeholder", "retrieve"], {"default": "diffusion"}),
        }}

    RETURN_TYPES = ("MSC_SPECS", "IMAGE")
    RETURN_NAMES = ("specs", "preview")
    FUNCTION = "run"
    CATEGORY = "MscProj"

    def run(self, media, specs, backend):
        import numpy as np
        import torch
        from PIL import Image
        from src.stage6_visual_augmentation import generate_augmentations, unload_generator
        out = generate_augmentations(specs, Path(media["work"]), backend=backend,
                                     size=config.RESOLUTION, model=config.GEN_MODEL,
                                     device=config.DEVICE)
        try:
            unload_generator()
        except Exception:
            pass
        _free()
        imgs = [s.image_path for s in out if s.augment and s.image_path and Path(s.image_path).exists()]
        if not imgs:
            blank = torch.zeros(1, 64, 64, 3)
            return (out, blank)
        arrs = []
        for p in imgs[:8]:
            im = Image.open(p).convert("RGB").resize((512, 512))
            arrs.append(np.asarray(im, dtype=np.float32) / 255.0)
        return (out, torch.from_numpy(np.stack(arrs)))


# --------------------------------------------------------------------------- 7. the panel
class MscComposite:
    """Stage 6b — put the pictures beside the video, which is what a viewer actually sees."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "media": ("MSC_MEDIA",),
            "specs": ("MSC_SPECS",),
            "events": ("MSC_EVENTS",),
        }}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_video",)
    OUTPUT_NODE = True
    FUNCTION = "run"
    CATEGORY = "MscProj"

    def run(self, media, specs, events):
        from src.stage6_visual_augmentation import composite_alongside
        out = Path(config.OUTPUT_DIR) / f"{media['stem']}_augmented.mp4"
        out.parent.mkdir(parents=True, exist_ok=True)
        composite_alongside(Path(media["video_path"]), specs, out,
                            duration=media["media"].duration, panel=config.PANEL_SIZE,
                            fps=config.FPS, mode=config.RENDER_MODE, events=events)
        _free()
        return (str(out),)


NODE_CLASS_MAPPINGS = {
    "MscLoadVideo": MscLoadVideo,
    "MscSceneUnderstanding": MscSceneUnderstanding,
    "MscTranscribe": MscTranscribe,
    "MscDetectEvents": MscDetectEvents,
    "MscCrossModalGate": MscCrossModalGate,
    "MscGeneratePictures": MscGeneratePictures,
    "MscComposite": MscComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MscLoadVideo": "1 · Load video + extract audio",
    "MscSceneUnderstanding": "2 · What is on screen",
    "MscTranscribe": "3 · Speech recognition",
    "MscDetectEvents": "4 · Sound detection (+ the two vetoes)",
    "MscCrossModalGate": "5 · Cross-modal gate  ← the contribution",
    "MscGeneratePictures": "6 · Generate pictures",
    "MscComposite": "7 · Composite beside the video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
