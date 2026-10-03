"""Run the frozen pipeline (D', tag detector-frozen-2026-10-02) on one video, exactly as `main.py` does, and write a
short summary of what the viewer will see.

The ComfyUI node starts this in a fresh process: every model is loaded and freed here, so the ComfyUI server never
holds Qwen3.8-27B or Qwen-Image itself (the old in-server nodes were OOM-killed), and the code path is main.py's own.

    python comfyui_nodes/run_frozen.py --input VIDEO --summary OUT.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--summary", required=True)
    ap.add_argument("--new-drawings", action="store_true",
                    help="a new random picture seed per run (default: the frozen seed from clip name, sound and time)")
    a = ap.parse_args()

    import main as M                       # the project's entry point: use_shipped() + on-the-spot listener inputs
    from src import pipeline
    got = {}
    run = pipeline.run

    def capture(*args, **kw):              # main() does not return the result; keep it for the summary
        # main() has built this clip's listener inputs by now (ensure_listener_inputs raises if a step fails). A clip
        # with no listener items (tg: FlexSED finds no band run) has legitimately empty answer files, which
        # _require_caches reads as missing; the scored D' arm runs with LISTENER_REQUIRE_CACHES False, so does this.
        import config
        config.LISTENER_REQUIRE_CACHES = False
        got["r"] = run(*args, **kw)
        return got["r"]

    pipeline.run = capture

    if a.new_drawings:                     # same sounds and times, a different drawing each run
        import random
        from benchmark.gold import gen_screen
        offset = random.SystemRandom().randrange(1, 1 << 30)
        frozen_seed = gen_screen.seed_of
        gen_screen.seed_of = lambda item: (frozen_seed(item) + offset) & 0x7FFFFFFF
        print(f"[run_frozen] new drawings: seed offset {offset}", flush=True)

    # Stage 5 exactly as the scored D' runs did it (benchmark/gold/round13_dev.py stage5: config.use_scored(), picture
    # wording flags off, KINSHIP_DIRECTED False). use_shipped() switches these on for the picture step; turned on before
    # stage 5 they change the subject wording, and the duplicate-picture check reads that wording (tg_d088: Explosion
    # merged into Thunder). So they are off from the plan to the end of decide_subjects, and back on for the pictures,
    # the same split the inspector renderer used (benchmark/gold/render_trail_media.py).
    import config
    from src.stage5_cross_modal_analysis import reason
    scored = {"KINSHIP_DIRECTED": False, "PICTURE_V3": False, "PICTURE_SCENE": False, "PICTURE_SCENE_GUARD2": False,
              "PICTURE_FINAL": False, "PICTURE_MAKER": False}
    saved = {}

    def to_scored():
        if not saved:
            saved.update({k: getattr(config, k, None) for k in scored})
            for k, v in scored.items():
                setattr(config, k, v)

    def to_shipped():
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
        return inner

    pipeline.plan_augmentations = wrap(pipeline.plan_augmentations, to_scored, lambda: None)
    reason.decide_subjects = wrap(reason.decide_subjects, lambda: None, to_shipped)
    pipeline.generate_augmentations = wrap(pipeline.generate_augmentations, to_shipped, lambda: None)
    # the GPU flags of slurm/job_main_demo.sh (config.DEVICE defaults to "cpu")
    sys.argv = ["main.py", "--input", a.input, "--device", "cuda", "--generator", "diffusion"]
    M.main()

    r = got["r"]
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    shown = [(l, round(float(x), 2), round(float(y), 2)) for _, l, x, y, _ in
             _assign_rows(_display_spans(r.specs, r.media.duration))[0]]
    Path(a.summary).write_text(json.dumps({
        "video": str(r.output_path),
        "duration": r.media.duration,
        "heard": sorted({e.label for e in r.events}),
        "shown": shown,
        "drawn": [{"label": s.event_label, "start": s.start, "end": s.end, "subject": s.subject,
                   "image": s.image_path} for s in r.specs if s.augment],
        "skipped": [{"label": s.event_label, "start": s.start, "reason": (s.reason or "")[:120]}
                    for s in r.specs if not s.augment],
    }, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
