"""Inspector videos of a shipped arm (D' = SHIP8+MD3+WW5+SL) on merged DEV and merged TEST: the pipeline's own stage 6
(config.use_shipped(): Qwen-Image-2512 with the frozen picture setup + the picture check, src/stage6_visual_augmentation
generate_augmentations) on the arm's scored augmentations.json, then composite_alongside with the arm's display flags.

Videos are kept per version: a clip is re-rendered only when its display spans change. Key = the clip's scored picture
signature [(label, start, end), ...] (what S.load_pictures returns, rounded to 0.01 s); file
<out>/<split>/<clip>.<sha1(signature)[:10]>.mp4 with <clip>.<hash>.sig.json next to it. The same hash is what
benchmark/gold/inspector_trail_export.py writes into data.js. Each video is checked: the compositor's own display spans
must equal the scored signature (recorded in the .sig.json as "check").

    # cluster, from ~/MscProj_tg (one H200 per shard: generator + checker VLM on one card)
    python benchmark/gold/render_trail_media.py --arm SHIP8+MD3+WW5+SL --shard 0/4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import inspector_trail_export as X     # noqa: E402  (gold read before any harness stub)
from benchmark.gold import score_per_sound as S             # noqa: E402


def sig_hash(sig):
    return hashlib.sha1(json.dumps(sig).encode()).hexdigest()[:10]


def load_specs(d: Path):
    from src.types import AugmentationSpec
    out = []
    for s in json.loads((d / "augmentations.json").read_text(encoding="utf-8")):
        k = {f: v for f, v in s.items() if f in AugmentationSpec.__dataclass_fields__}
        k["spans"] = [tuple(x) for x in k.get("spans") or []]
        k["breaks"] = [tuple(x) for x in k.get("breaks") or []]
        out.append(AugmentationSpec(**k))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="SHIP8+MD3+WW5+SL")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", default=str(_ROOT / "data" / "output" / "inspector_media"))
    ap.add_argument("--pics", default=str(_ROOT / "data" / "work" / "inspector_media_pics"))
    ap.add_argument("--splits", nargs="*", default=["DEV", "TEST"])
    ap.add_argument("--clips", nargs="*", default=[])
    ap.add_argument("--no-verify", action="store_true", help="skip the picture check (debug only)")
    a = ap.parse_args()
    si, sn = (int(x) for x in a.shard.split("/"))
    import config
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import round13_dev as R
    disp = {k: R.arm_cfg(a.arm)[k] for k in R.DISPLAY_KEYS}
    todo = []
    for split, part, base, stems in X.parts(a.arm):
        if split in a.splits:
            todo += [(split, base / f"{a.arm}_proposed", st) for st in stems if not a.clips or st in a.clips]
    todo = [t for j, t in enumerate(todo) if j % sn == si]
    print(f"[media] shard {a.shard}: {len(todo)} clips", flush=True)
    changed = config.use_shipped()
    print("[cfg] use_shipped:", {k: v[1] for k, v in changed.items() if k in ("GEN_MODEL", "PICTURE_VERIFY", "PICTURE_FINAL")}, flush=True)
    config.DEVICE = "cuda"
    if a.no_verify:
        config.PICTURE_VERIFY = False
    from src.stage6_visual_augmentation import generate_augmentations, composite_alongside, _display_spans, _assign_rows
    done = skipped = bad = 0
    for split, root, st in todo:
        d = root / st
        with R.flags(disp):
            pics = S.load_pictures(root, st, "proposed") or []
        sig = DCC.pics_sig(pics)
        h = sig_hash(sig)
        out = Path(a.out) / split / f"{st}.{h}.mp4"
        sj = out.with_suffix(".sig.json")
        if out.exists() and sj.exists():
            skipped += 1
            continue
        media = json.loads((d / "media.json").read_text(encoding="utf-8"))
        video = Path(media["video_path"])
        dur = float(media["duration"])
        specs = load_specs(d)
        pw = Path(a.pics) / split / st
        rj = pw / "augmentations_rendered.json"
        want = [(s.index, s.event_label, round(s.start, 3), s.subject) for s in specs if s.augment]
        if rj.exists() and json.loads(rj.read_text(encoding="utf-8")).get("key") == [list(x) for x in want]:
            rend = json.loads(rj.read_text(encoding="utf-8"))["specs"]
            for s, r in zip(specs, rend):
                s.image_path, s.image_prompt, s.backend = r["image_path"], r["image_prompt"], r["backend"]
        else:
            pw.mkdir(parents=True, exist_ok=True)
            for f in ("media.json", "augmentations.json"):
                shutil.copy(d / f, pw / f)
            t0 = time.time()
            for s in specs:
                s.image_path = None
            specs = generate_augmentations(specs, pw, backend=config.GEN_BACKEND, size=config.RESOLUTION,
                                           model=config.GEN_MODEL, device=config.DEVICE)
            rj.write_text(json.dumps({"key": [list(x) for x in want], "specs": [
                {"image_path": s.image_path, "image_prompt": s.image_prompt, "backend": s.backend} for s in specs]}, indent=1),
                encoding="utf-8")
            print(f"[media] {split}/{st}: {len(want)} picture(s) in {time.time() - t0:.0f} s", flush=True)
        with R.flags({**disp, "GROUP_CLIP": st, "SHOW_DEBUG_SOUNDS": False, "SHOW_PROMPT": False}):
            shown = [(l, float(x), float(y)) for _, l, x, y, _ in _assign_rows(_display_spans(specs, dur))[0]]
            check = DCC.pics_sig(shown) == sig
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_name(out.stem + ".tmp.mp4")
            composite_alongside(video, specs, tmp, duration=dur, panel=config.PANEL_SIZE, fps=config.FPS, mode="full",
                                events=None)
        tmp.replace(out)
        shutil.rmtree(tmp.parent / f"_{tmp.stem}_panels", ignore_errors=True)
        sj.write_text(json.dumps({"clip": st, "split": split, "arm": a.arm, "hash": h, "sig": sig,
                                  "check": "video spans == scored pictures" if check else {"video": DCC.pics_sig(shown)},
                                  "pictures": [{"label": s.event_label, "start": s.start, "subject": s.subject,
                                                "backend": s.backend, "image": Path(s.image_path).name if s.image_path else None}
                                               for s in specs if s.augment]}, indent=1), encoding="utf-8")
        done += 1
        bad += int(not check)
        print(f"[media] {split}/{st} -> {out.name}{'' if check else '  (CHECK FAILED: spans differ)'}", flush=True)
    print(f"[media] DONE shard {a.shard}: rendered {done}, kept {skipped}, span check failed {bad}", flush=True)


if __name__ == "__main__":
    main()
