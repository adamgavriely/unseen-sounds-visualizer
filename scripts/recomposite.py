"""Re-composite the side-panel video from a saved work directory -- no models, CPU only.

The protocol run keeps every stage's output under data/work/protocol_<system>_<tag>/<clip>/
(events.json, augmentations.json with the generated pictures, media.json) but writes the
final video to data/output/<clip>_augmented.mp4, where the next system overwrites it. This
rebuilds the video from those artifacts, so the exact pictures the judge scored can be
watched, in the clean or the debug view (detections and gate verdicts under the panel).

    python scripts/recomposite.py --tag v3 --system proposed --clips a.mp4 b.mp4 --out data/output/recomp_v3
    python scripts/recomposite.py --tag v3 --system proposed --shown --tags unseen_ambient mixed --out ...

Inspector videos (shipped pictures; blind arm as label chips, so no FLUX picture is shown):
    python scripts/recomposite.py --root data/work/shipped_<tag> --shipped --require-local-images         --modes clean --out-name "{stem}.mp4" --out <dir>
    python scripts/recomposite.py --root data/work/protocol_blind_a2i_<tag> --shipped --render-mode minimal         --modes clean --out-name "{stem}_blind.mp4" --out <dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.types import AugmentationSpec, AudioEvent
from src.stage6_visual_augmentation import composite_alongside


def load_work(work: Path):
    media = json.loads((work / "media.json").read_text(encoding="utf-8"))
    events = [AudioEvent(**{k: v for k, v in e.items() if k in AudioEvent.__dataclass_fields__})
              for e in json.loads((work / "events.json").read_text(encoding="utf-8"))]
    specs = []
    for s in json.loads((work / "augmentations.json").read_text(encoding="utf-8")):
        d = {k: v for k, v in s.items() if k in AugmentationSpec.__dataclass_fields__}
        if d.get("image_path") and not Path(d["image_path"]).exists():
            cand = work / "augmentations" / Path(d["image_path"]).name      # relocated work dir
            if cand.exists():
                d["image_path"] = str(cand)
        specs.append(AugmentationSpec(**d))
    return media, events, specs


def local_images_ok(work: Path, specs) -> bool:
    """Every picture a spec will show lives in this work dir's own augmentations/ (no stale path elsewhere)."""
    own = (work / "augmentations").resolve()
    return all(Path(s.image_path).resolve().parent == own and Path(s.image_path).exists()
               for s in specs if s.augment and s.image_path)


def video_for(work: Path, stem: str) -> Path | None:
    try:
        p = Path(json.loads((work / "media.json").read_text(encoding="utf-8"))["video_path"])
        if p.exists():
            return p
    except (OSError, KeyError, ValueError):
        pass
    return find_clip(stem)


def find_clip(stem: str) -> Path | None:
    for p in (_ROOT / "data" / "input" / "benchmark").rglob("*"):
        if p.is_file() and p.stem == stem:
            return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v3"); ap.add_argument("--system", default="proposed")
    ap.add_argument("--clips", nargs="*", default=[], help="clip file names; default: every clip in the work root")
    ap.add_argument("--shown", action="store_true", help="only clips where at least one picture was shown")
    ap.add_argument("--tags", nargs="*", default=[], help="restrict to these human tags (needs benchmark/tags.json)")
    ap.add_argument("--out", default=str(_ROOT / "data" / "output" / "recomp"))
    ap.add_argument("--modes", nargs="*", default=["clean", "debug"])
    ap.add_argument("--root", default="", help="work root to read (default: data/work/protocol_<system>_<tag>)")
    ap.add_argument("--render-mode", default="full", choices=["full", "minimal"],
                    help="full = pictures; minimal = label chips, no imagery")
    ap.add_argument("--shipped", action="store_true",
                    help="config.use_shipped(): the scored display settings, pictures at full opacity")
    ap.add_argument("--require-local-images", action="store_true",
                    help="skip a clip whose pictures are not in its own work dir's augmentations/")
    ap.add_argument("--out-name", default="{mode}/{stem}_augmented.mp4", help="output path under --out")
    ap.add_argument("--keep-panels", action="store_true", default=True)
    ap.add_argument("--drop-panels", dest="keep_panels", action="store_false",
                    help="delete the per-clip panel frames after encoding")
    a = ap.parse_args()
    if a.shipped:
        print("[cfg]", config.use_shipped(), flush=True)
    root = Path(a.root) if a.root else _ROOT / "data" / "work" / f"protocol_{a.system}_{a.tag}"
    if not root.is_absolute():
        root = _ROOT / root
    tags = {}
    if a.tags:
        raw = json.loads((_ROOT / "benchmark" / "tags.json").read_text(encoding="utf-8"))
        tags = {Path(k).name: v["tag"] for k, v in raw.items()}
    stems = [Path(c).stem for c in a.clips] or sorted(p.name for p in root.iterdir() if (p / "augmentations.json").exists())
    out_root = Path(a.out)
    done = 0
    for stem in stems:
        work = root / stem
        if not (work / "augmentations.json").exists():
            print(f"  ! no work dir for {stem}"); continue
        media, events, specs = load_work(work)
        if a.shown and not any(s.augment and s.image_path for s in specs):
            continue
        if a.tags:
            t = next((tags[k] for k in tags if Path(k).stem == stem), None)
            if t not in a.tags:
                continue
        if a.require_local_images and not local_images_ok(work, specs):
            print(f"  ! {stem}: a picture is missing from {work / 'augmentations'}; skipped"); continue
        video = video_for(work, stem)
        if video is None:
            print(f"  ! no video for {stem}"); continue
        for mode in a.modes:
            config.SHOW_DEBUG_SOUNDS = config.SHOW_PROMPT = (mode == "debug")
            out = out_root / a.out_name.format(mode=mode, stem=stem)
            if out.exists():
                continue
            composite_alongside(video, specs, out, duration=media["duration"], panel=config.PANEL_SIZE,
                                fps=config.FPS, mode=a.render_mode, events=events)
            if not a.keep_panels:
                import shutil
                shutil.rmtree(out.parent / f"_{out.stem}_panels", ignore_errors=True)
            print(f"[{mode}] {stem} -> {out}", flush=True)
        done += 1
    print(f"recomposited {done} clips -> {out_root}")


if __name__ == "__main__":
    main()
