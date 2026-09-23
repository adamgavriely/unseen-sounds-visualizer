"""Run the seven demo nodes in order, outside ComfyUI, on one video.

This is the test that matters: ComfyUI only calls these classes, so if the chain works here it works
in the graph. Point it at a video the pipeline has never seen -- the FlexSED cache is keyed by the
clip's name, so a fresh name exercises the subprocess worker that makes unseen videos possible.

    python comfyui_nodes/smoke.py --video /path/to/any.mp4 [--gen placeholder]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--gen", default="diffusion", choices=["diffusion", "placeholder", "retrieve"])
    ap.add_argument("--gate", default="on", choices=["on", "off"])
    ap.add_argument("--tag", default="", help="keep this run's work dir separate, e.g. the torch version")
    a = ap.parse_args()

    import config
    config.DEVICE = "cuda"
    config.VIDEO_BACKEND = "owlv2"
    config.TRANSCRIBE = True
    # gate on and gate off must not write to the same file: the A/B for the defence is two videos
    config.OUTPUT_DIR = Path(config.OUTPUT_DIR) / f"demo_gate_{a.gate}{('_' + a.tag) if a.tag else ''}"
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if a.tag:
        # a separate work dir per run, so the SAME clip under two torch builds can be diffed
        config.WORK_DIR = Path(config.WORK_DIR) / f"torch_{a.tag}"
        config.WORK_DIR.mkdir(parents=True, exist_ok=True)

    from comfyui_nodes import (MscSystem, MscLoadVideo, MscSceneUnderstanding, MscTranscribe,
                               MscDetectEvents, MscCrossModalGate, MscGeneratePictures,
                               MscComposite)

    def step(n, name, fn):
        t = time.time()
        out = fn()
        print(f"[{n}/7] {name}: {time.time() - t:.1f}s", flush=True)
        return out

    cfg, sysname = MscSystem().run(a.gate == "on")
    print(f"[0/7] system: {sysname}")
    (media,) = step(1, "load video + audio", lambda: MscLoadVideo().run(cfg, a.video))
    print(f"      {media['stem']}  {media['media'].duration:.1f}s")

    scene, vis = step(2, "what is on screen", lambda: MscSceneUnderstanding().run(media, "owlv2", 6))
    print(f"      visible: {vis}")

    segs, tr = step(3, "speech", lambda: MscTranscribe().run(media, True))
    print(f"      {len(segs)} segment(s)")

    events, heard = step(4, "sound detection + vetoes",
                         lambda: MscDetectEvents().run(media, 0.8, 0.3, 0.05))
    print(f"      heard: {heard}")

    specs, decisions = step(5, "cross-modal gate",
                            lambda: MscCrossModalGate().run(media, scene, segs, events, cfg))
    print("      " + decisions.replace("\n", "\n      "))

    specs2, _ = step(6, f"pictures ({a.gen})",
                     lambda: MscGeneratePictures().run(media, specs, a.gen))
    n = sum(1 for s in specs2 if s.augment and s.image_path)
    print(f"      {n} picture(s)")

    (out,) = step(7, "composite", lambda: MscComposite().run(media, specs2, events))
    print(f"\nOK -> {out}")


if __name__ == "__main__":
    main()
