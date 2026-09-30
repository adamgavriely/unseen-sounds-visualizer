"""Stage-6 hand-off for a version run by slurm/run_best.sh: make a protocol-style work folder whose augmentations are that
version's, so the shipped picture step (scripts/repaint_shipped.py) and the compositor (scripts/recomposite.py) can draw and
render it unchanged.

  data/work/protocol_proposed_<NAME>_v33/<clip>/*       the scored render of the split (events, scene, media, ...)
  data/work/r13<NAME>/<ARM>_proposed/<clip>/            the arm's augmentations.json + gate_votes.json
  -> data/work/protocol_proposed_<NAME>best_v33/<clip>/ the render's files with the arm's augmentations / gate votes

    python scripts/best_to_protocol.py --name NAME [--arm TO1+F7F8]
    python scripts/repaint_shipped.py --tag NAMEbest_v33 --verify
    python scripts/recomposite.py --root data/work/shipped_v_NAMEbest_v33 --shipped --require-local-images --modes clean --out <dir>
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
WORK = _ROOT / "data" / "work"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--arm", default="TO1+F7F8")
    a = ap.parse_args()
    src = WORK / f"protocol_proposed_{a.name}_v33"
    arm = WORK / f"r13{a.name}" / f"{a.arm}_proposed"
    dst = WORK / f"protocol_proposed_{a.name}best_v33"
    n = 0
    for clip in sorted(p for p in arm.iterdir() if p.is_dir()):
        s = src / clip.name
        if not (s / "media.json").exists() or not (clip / "augmentations.json").exists():
            print("skip (incomplete)", clip.name)
            continue
        d = dst / clip.name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(s, d, ignore=shutil.ignore_patterns("augmentations", "*.png", "*.mp4"))
        for f in ("augmentations.json", "gate_votes.json"):
            if (clip / f).exists():
                shutil.copy2(clip / f, d / f)
        n += 1
    print(f"{n} clips -> {dst}")


if __name__ == "__main__":
    main()
