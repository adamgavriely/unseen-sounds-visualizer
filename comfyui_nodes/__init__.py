"""ComfyUI nodes for the project: add pictures of off-screen sounds to a video.

Main node: `MscAugmentVideo` ("Add sound pictures to video"). It takes a video, runs the final system
(`config.use_shipped()`, the same code path as `main.py`) in a separate process, and returns the video with the
picture panel and a short text list of what was drawn. The workflow file is `comfyui_nodes/MscProj_video.json`.

Install: link or copy this folder into ComfyUI/custom_nodes/, and set MSCPROJ_ROOT to the project folder if this
folder is not inside it. Start ComfyUI with `python comfyui_nodes/comfy_start.py`.

Run time: about five minutes of GPU time (one NVIDIA H200) to prepare the listener inputs of a 15-s clip, plus the
picture step (about 20 s per picture try, up to five tries per picture). See comfyui_nodes/README.md.
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


# --------------------------------------------------------------------------- the one-click node (frozen final system)
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

    The per-clip input builder only looks for lower-case *.mp4 files (benchmark/gold/clip_prep.py stems_of), and stage 1
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
    """Video in, video out: the final system on a new video.

    `main.py`'s code path (use_shipped + on-the-spot listener inputs), with stage 5 run under the flags of the scored
    runs (picture wording and KINSHIP_DIRECTED off, as benchmark/gold/dev_harness.py) and the pictures under the final
    picture flags. See run_frozen.py.

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


NODE_CLASS_MAPPINGS = {
    "MscAugmentVideo": MscAugmentVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MscAugmentVideo": "Add sound pictures to video (MscProj, frozen pipeline)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
