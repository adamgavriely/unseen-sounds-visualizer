"""Visual Augmentation of Audio Semantics for Accessibility - run the final system on one video.

    python main.py --input clip.mp4 --device cuda

Writes data/output/<stem>_augmented.mp4 (the video with the picture panel) and the intermediate files under
data/work/<stem>/. Without --listener-split, the per-clip listener inputs are computed first (about 5 minutes on an
H200). See docs/report/report.pdf, Section 5 and Appendix C.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import config
from src import pipeline

# Stage 5 (planning, on-screen check, subject wording and duplicate check) runs with the flags of the scored results;
# the picture step runs with the final picture flags. use_shipped() turns the picture-wording flags on for the whole
# run; switched on before stage 5 they change the subject wording that the duplicate check reads, so a few clips then
# show different pictures from the scored run (e.g. tg_d088 lost its Explosion picture). This is the split the scored
# picture renderer (benchmark/gold/render_trail_media.py) uses.
_SCORED_STAGE5 = {"KINSHIP_DIRECTED": False, "PICTURE_V3": False, "PICTURE_SCENE": False,
                  "PICTURE_SCENE_GUARD2": False, "PICTURE_FINAL": False, "PICTURE_MAKER": False}


def _run_stage5_as_scored() -> None:
    from src.stage5_cross_modal_analysis import reason
    saved: dict = {}

    def to_scored():
        if not saved:
            saved.update({k: getattr(config, k, None) for k in _SCORED_STAGE5})
            for k, v in _SCORED_STAGE5.items():
                setattr(config, k, v)

    def to_final():
        for k, v in saved.items():
            setattr(config, k, v)
        saved.clear()

    def wrap(fn, before, after):
        def inner(*args, **kw):
            before()
            try:
                return fn(*args, **kw)
            finally:
                after()
        inner.__wrapped__ = fn
        return inner

    pipeline.plan_augmentations = wrap(pipeline.plan_augmentations, to_scored, lambda: None)
    reason.decide_subjects = wrap(reason.decide_subjects, lambda: None, to_final)
    pipeline.generate_augmentations = wrap(pipeline.generate_augmentations, to_final, lambda: None)


def _stage_input(video: Path) -> Path:
    """Copy (or convert) any input video to data/input/live/<stem>_<content hash>.mp4.

    The per-clip caches are keyed by file name, so the content hash keeps a new video with an old name from reusing old
    answers; the .mp4 extension (lower case) is what the cache builder looks for. A video without an audio track has no
    sounds to detect, so it stops with a clear message instead of a decoder error."""
    import hashlib
    import re
    import shutil
    import subprocess
    video = Path(video).resolve()
    if not video.is_file():
        raise SystemExit(f"input not found: {video}")
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    if probe.returncode != 0:
        raise SystemExit(f"cannot read {video.name} as a video (ffprobe: {probe.stderr.strip()[:200]})")
    if not probe.stdout.strip():
        raise SystemExit(f"{video.name} has no audio track: there are no sounds to show. Nothing was written.")
    h = hashlib.sha1()
    with open(video, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", video.stem).strip("_") or "clip"
    dest = Path(__file__).resolve().parent / "data" / "input" / "live" / f"{stem}_{h.hexdigest()[:8]}.mp4"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if video.suffix == ".mp4":
        shutil.copy2(video, dest)
    else:                                   # .MP4, .mov, .mkv, .webm, ...: re-encode to a plain H.264/AAC mp4
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(dest)], check=True)
    return dest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="path to input video clip")
    ap.add_argument("--work-dir", default=str(config.WORK_DIR),
                    help="where to write intermediate artifacts")
    ap.add_argument("--raw-config", action="store_true",
                    help="use the bare config.py defaults instead of the final system (config.use_shipped())")
    ap.add_argument("--fewer-false", action="store_true",
                    help="N2b (already in the final system; kept for old command lines)")
    ap.add_argument("--listener-split", default=None,
                    help="name given to slurm/run_best.sh for this clip's folder: where the audio-LLM answers and DASM "
                         "scores of the final detector are (without them stage 4 stops)")
    ap.add_argument("--whisper-model", default=None, help="tiny|base|small|medium|large-v3")
    ap.add_argument("--device", default=None, help="cpu|cuda")
    ap.add_argument("--generator", default=None, choices=["retrieve", "placeholder", "diffusion"],
                    help="how Stage 6 produces images")
    ap.add_argument("--video-backend", default=None, choices=["siglip", "owlv2", "vlm"],
                    help="Stage 2 object finder")
    ap.add_argument("--debug-panel", action="store_true",
                    help="print the phrase and every raw detection with the gate's verdict under the panel")
    args = ap.parse_args()

    video = Path(args.input)
    if not args.listener_split:             # precomputed splits are keyed by the original file name: use it as is
        video = _stage_input(video)
        print(f"input: {video}", flush=True)

    # the final system by default (Qwen-Image pictures, scored detector stack, PANNs veto); CLI flags override it
    if not args.raw_config:
        config.use_shipped()
        _run_stage5_as_scored()
        if args.fewer_false:
            config.use_n2b()
        if args.listener_split:
            config.set_listener_split(args.listener_split)
        else:
            # no precomputed answers named: compute them on the spot (same harness as the benchmark, slow)
            from src.listener_prep import ensure_listener_inputs
            config.set_listener_split(ensure_listener_inputs(video))
            # the inputs were just built for this clip; a clip with no weak candidate sounds legitimately has empty
            # listener files, which the cache check would treat as missing (the scored runs ran with the check off)
            config.LISTENER_REQUIRE_CACHES = False
    for k, v in (("WHISPER_MODEL", args.whisper_model), ("DEVICE", args.device), ("GEN_BACKEND", args.generator),
                 ("VIDEO_BACKEND", args.video_backend)):
        if v is not None:
            setattr(config, k, v)
    if args.debug_panel:
        config.SHOW_PROMPT = True
        config.SHOW_DEBUG_SOUNDS = True

    try:
        from dotenv import load_dotenv
        load_dotenv()  # optional: makes any API keys in .env available to stages
    except ImportError:
        pass

    pipeline.run(video, work_root=Path(args.work_dir))


if __name__ == "__main__":
    main()
