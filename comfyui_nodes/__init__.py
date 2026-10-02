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


# --------------------------------------------------------------------------- the one-click node (frozen D')
def _video_file(video) -> Path:
    """The uploaded video as a file on disk (ComfyUI's VIDEO is usually a path, sometimes bytes)."""
    import tempfile
    src = video.get_stream_source() if hasattr(video, "get_stream_source") else video
    if isinstance(src, (str, Path)):
        return Path(src)
    tmp = Path(tempfile.mkdtemp()) / "upload.mp4"
    video.save_to(str(tmp))
    return tmp


def _tool(name: str) -> str:
    """ffmpeg / ffprobe next to this Python (the project env), else on PATH."""
    import shutil
    here = Path(sys.executable).parent / name
    return str(here) if here.exists() else (shutil.which(name) or name)


def _check_and_normalize(src: Path, dst: Path) -> None:
    """Refuse videos the pipeline cannot use, with a plain message, and give the pipeline an .mp4 it can read.

    The listener harness only looks for lower-case *.mp4 files (benchmark/gold/tagger_prep.py stems_of), and stage 1
    reads the container duration; a phone .MOV, a .webm with no duration, or an .MP4 would fail deep inside the run. So
    every upload becomes <stem>.mp4: copied when it already is a plain .mp4, otherwise re-encoded (H.264 + AAC).
    """
    import json
    probe = subprocess.run([_tool("ffprobe"), "-v", "error", "-show_entries", "stream=codec_type:format=duration",
                            "-of", "json", str(src)], capture_output=True, text=True)
    if probe.returncode != 0:
        raise RuntimeError(f"This file could not be read as a video ({src.name}).")
    info = json.loads(probe.stdout or "{}")
    kinds = {st.get("codec_type") for st in info.get("streams", [])}
    if "video" not in kinds:
        raise RuntimeError("This file has no picture track. Please upload a video, not an audio file.")
    if "audio" not in kinds:
        raise RuntimeError("This video has no sound track, so there are no sounds to show. Please upload a video "
                           "with sound.")
    try:
        dur = float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        dur = None
    if dur is not None and dur < 1.0:
        raise RuntimeError(f"This video is too short ({dur:.1f} s). Please upload at least 1 second.")
    if dst.exists():
        return
    tmp = dst.with_name(dst.stem + ".part.mp4")
    if src.suffix == ".mp4" and dur is not None:
        import shutil
        shutil.copy(src, tmp)
    else:
        r = subprocess.run([_tool("ffmpeg"), "-y", "-v", "error", "-i", str(src), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-movflags", "+faststart", str(tmp)], capture_output=True, text=True)
        if r.returncode != 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError("This video could not be converted to MP4: " + (r.stderr or "")[-300:])
    tmp.replace(dst)


class MscAugmentVideo:
    """Video in, video out: the frozen pipeline (D', tag detector-frozen-2026-10-02) on a new video.

    `main.py`'s code path (use_shipped + on-the-spot listener inputs), with stage 5 run under the scored D' flags
    (picture wording and KINSHIP_DIRECTED off, as benchmark/gold/round13_dev.py) and the pictures under the shipped
    flags, the same split as the inspector renderer (benchmark/gold/render_trail_media.py). See run_frozen.py.

    Every stage is computed on the spot for the new video (audio, what is on screen, the four listeners, sound
    detection, the cross-modal gate, Qwen-Image pictures with the picture check, grouping, the compositor). It runs in
    a fresh process (comfyui_nodes/run_frozen.py), so the ComfyUI server never holds the big models. A copy of the video is named after its content, so per-clip answers from an older upload with
    the same file name are never reused.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"video": ("VIDEO", {"tooltip": "Connect the Load Video node here."}),
                             "pictures": (["new drawing each run", "same as the frozen run"],
                                          {"default": "new drawing each run",
                                           "tooltip": "The sounds and times are always the same; this only changes "
                                                      "the random seed of the picture model."})},
                "hidden": {"unique_id": "UNIQUE_ID"}}

    @classmethod
    def IS_CHANGED(cls, video, pictures="new drawing each run", unique_id=None):
        # a new drawing must really run again; the frozen seed may reuse ComfyUI's cached result
        return float("nan") if pictures.startswith("new") else pictures

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video_with_pictures", "what_was_drawn")
    FUNCTION = "run"
    CATEGORY = "MscProj"
    DESCRIPTION = ("Adds pictures of the sounds you cannot see. Runs the whole frozen pipeline on the uploaded "
                   "video (about 10-15 minutes on one H200).")

    def run(self, video, pictures="new drawing each run", unique_id=None):
        import hashlib
        import time
        import json
        import re
        src = _video_file(video)
        h = hashlib.sha1()
        with open(src, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        name = re.sub(r"[^A-Za-z0-9_]+", "_", src.stem).strip("_")[:40] or "video"
        stem = f"comfy_{name}_{h.hexdigest()[:8]}"
        inp = _ROOT / "data" / "input" / "comfy" / f"{stem}.mp4"
        inp.parent.mkdir(parents=True, exist_ok=True)
        _check_and_normalize(src, inp)
        summary = Path(config.WORK_DIR) / "comfy" / f"{stem}.summary.json"
        summary.parent.mkdir(parents=True, exist_ok=True)
        summary.unlink(missing_ok=True)       # never show an older run's result

        pbar = None
        try:
            import comfy.utils
            pbar = comfy.utils.ProgressBar(7)
        except Exception:
            pass
        cmd = [sys.executable, "-u", str(_ROOT / "comfyui_nodes" / "run_frozen.py"),
               "--input", str(inp), "--summary", str(summary)] + (["--new-drawings"] if pictures.startswith("new") else [])
        env = dict(os.environ)
        for k in ("PYTHONPATH",):        # ComfyUI's own packages must not leak into the pipeline's process
            env.pop(k, None)
        print(f"[MscProj] running the frozen pipeline on {inp.name}", flush=True)
        tail = []
        t0 = time.time()
        server = None
        try:
            from server import PromptServer
            server = PromptServer.instance
        except Exception:
            pass

        def say(text):                   # the current step, in words, under the node
            if server is not None and unique_id is not None:
                try:
                    server.send_progress_text(f"{(time.time() - t0) / 60:.0f} min - {text}", unique_id)
                except Exception:
                    pass

        say("starting (a short clip takes about 20-25 minutes)")
        in_prep, prep_last = False, False
        proc = subprocess.Popen(cmd, cwd=str(_ROOT), env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1)
        for line in proc.stdout:
            print("[MscProj] " + line.rstrip(), flush=True)
            tail = (tail + [line.rstrip()])[-40:]
            s = line.strip()
            p = re.search(r"\[listener-prep\] [^:]+: (.+)$", s)
            if p:
                in_prep, prep_last = True, p.group(1).startswith("finelap")
                say("part 1 of 2: the sound listeners listen (" + p.group(1) + ")")
                continue
            m = re.match(r"\[(\d)/7\] (.*?)(\.\.\.)?$", s)
            if m:
                if in_prep and prep_last and m.group(1) == "1":
                    in_prep = False
                if in_prep:
                    continue
                say(f"part 2 of 2, step {m.group(1)} of 7: {m.group(2)}")
                if pbar is not None:
                    pbar.update_absolute(int(m.group(1)) - 1, 7)
        if proc.wait() != 0 or not summary.exists():
            raise RuntimeError("The pipeline stopped. Last lines:\n" + "\n".join(tail[-15:]))
        if pbar is not None:
            pbar.update_absolute(7, 7)
        say("done")
        res = json.loads(summary.read_text(encoding="utf-8"))
        lines = [f"Heard: {', '.join(res['heard']) or 'nothing'}", "",
                 f"Pictures shown ({len(res['shown'])}):"]
        lines += [f"  {l}  {a:.1f}-{b:.1f} s" for l, a, b in res["shown"]] or ["  none"]
        if res["skipped"]:
            lines += ["", "Not drawn:"] + [f"  {s['label']} at {s['start']:.1f} s: {s['reason']}" for s in res["skipped"]]
        try:
            from comfy_api.latest import InputImpl
            out_video = InputImpl.VideoFromFile(res["video"])
        except Exception:
            from comfy_api.input_impl import VideoFromFile
            out_video = VideoFromFile(res["video"])
        return (out_video, "\n".join(lines))


# --------------------------------------------------------------------------- 0. the system switch
class MscSystem:
    """Which system is being demonstrated. Runs BEFORE every stage and sets the flags.

    The five flags below are copied from benchmark/run_protocol.py, which is what produced every
    number in the thesis. `gate_enabled` on gives the proposed system; off gives the blind baseline
    exactly as it was measured -- not a guess at it. RENDER_MODE is "full" for both, because that is
    what the protocol used for both.

    It exists as its own node because ComfyUI caches and may execute nodes out of order: flags set
    inside a later node would arrive after earlier nodes had already read them. Everything
    downstream takes `cfg`, so the graph cannot run a stage before this has run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"gate_enabled": ("BOOLEAN", {"default": True})}}

    RETURN_TYPES = ("MSC_CFG", "STRING")
    RETURN_NAMES = ("cfg", "system")
    FUNCTION = "run"
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

    @classmethod
    def IS_CHANGED(cls, gate_enabled):
        return float(bool(gate_enabled))

    def run(self, gate_enabled):
        on = bool(gate_enabled)
        config.use_v4("590")                 # the shipping row (docs/prereg_v4.md)
        config.GATE_ENABLED = on
        config.DEPICTION_REASONING = on
        config.VLM_VISIBILITY = on
        config.SPEECH_CONTEXT = on
        config.RENDER_MODE = "full"          # the protocol uses "full" for proposed AND blind
        name = "proposed (cross-modal gate)" if on else "blind_a2i (draw every detected sound)"
        return ({"gate_enabled": on}, name)


# --------------------------------------------------------------------------- 1. the video
class MscLoadVideo:
    """Stage 1 — take any video, pull its audio out. The only node that touches the filesystem."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"cfg": ("MSC_CFG",),
                             "video_path": ("STRING", {"default": "", "multiline": False})}}

    RETURN_TYPES = ("MSC_MEDIA",)
    RETURN_NAMES = ("media",)
    FUNCTION = "run"
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

    def run(self, cfg, video_path):
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
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

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
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

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
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

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
            "cfg": ("MSC_CFG",),
        }}

    RETURN_TYPES = ("MSC_SPECS", "STRING")
    RETURN_NAMES = ("specs", "decisions")
    FUNCTION = "run"
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

    def run(self, media, scene, segments, events, cfg):
        from src.stage5_cross_modal_analysis import plan_augmentations, reason
        gate_enabled = bool(cfg.get("gate_enabled", True))
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
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

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
    CATEGORY = "MscProj/old setup (not the frozen pipeline)"

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
    "MscAugmentVideo": MscAugmentVideo,
    "MscSystem": MscSystem,
    "MscLoadVideo": MscLoadVideo,
    "MscSceneUnderstanding": MscSceneUnderstanding,
    "MscTranscribe": MscTranscribe,
    "MscDetectEvents": MscDetectEvents,
    "MscCrossModalGate": MscCrossModalGate,
    "MscGeneratePictures": MscGeneratePictures,
    "MscComposite": MscComposite,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MscAugmentVideo": "Add sound pictures to video (MscProj, frozen pipeline)",
    "MscSystem": "(old setup) 0 · System: gate ON = ours, OFF = blind baseline",
    "MscLoadVideo": "(old setup) 1 · Load video + extract audio",
    "MscSceneUnderstanding": "(old setup) 2 · What is on screen",
    "MscTranscribe": "(old setup) 3 · Speech recognition",
    "MscDetectEvents": "(old setup) 4 · Sound detection (+ the two vetoes)",
    "MscCrossModalGate": "(old setup) 5 · Cross-modal gate  ← the contribution",
    "MscGeneratePictures": "(old setup) 6 · Generate pictures",
    "MscComposite": "(old setup) 7 · Composite beside the video",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
