"""Rating page, pass 2: the sound's name + one frame of the scene + the picture. "Does it fit, and does
it say anything false?" (docs/picture_v3_prereg.md, amendment 2: V3.1's verdict needs pass 2, because
pass 1 scores a narrower-but-wrong kind -- a tractor drawn for a car -- as correct.)

Done AFTER pass 1 on the same sounds, never before (pass 1 shows no name). The version that drew each
picture stays hidden; the keys are sealed the same way as pass 1.

    python benchmark/gold/rating_pass2.py --bench data/work/picture_bench_sliceB \
        --today data/work/protocol_proposed_sliceBblind_v32 --arms today,N0,N1 --out data/work/rate_pass2_sliceB
"""
from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from benchmark.gold.picture_bench import clip_path  # noqa: E402


def plain(label: str) -> str:
    return label.split("(")[0].split(",")[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True)
    ap.add_argument("--today", required=True)
    ap.add_argument("--arms", default="today,N0,N1")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    bench, out = Path(a.bench), Path(a.out)
    (out / "img").mkdir(parents=True, exist_ok=True)
    (out / "frame").mkdir(parents=True, exist_ok=True)
    specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
    v3 = json.loads((bench / "subjects_V3.json").read_text(encoding="utf-8"))

    def src(arm, it):
        if arm == "today":
            return Path(a.today) / it["clip"] / "augmentations" / Path(it["shipped_image"]).name
        return bench / arm / f"{it['i']:02d}.png"

    for it in specs:                               # one frame per sound, a second into it
        f = out / "frame" / f"s{it['i']:02d}.jpg"
        if not f.exists():
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-ss", f"{it['start'] + 1.0:.2f}", "-i",
                            str(clip_path(it["clip"])), "-frames:v", "1", "-vf", "scale=512:-2", str(f)])
    rng = random.Random(25092026)
    pool = [(arm, it) for arm in a.arms.split(",") for it in specs if src(arm, it).exists()]
    rng.shuffle(pool)
    cards, sound_key, arm_key = [], {}, {}
    for n, (arm, it) in enumerate(pool, 1):
        code = f"Q{n:03d}"
        Image.open(src(arm, it)).convert("RGB").resize((384, 384)).save(out / "img" / f"{code}.jpg", quality=86)
        source = v3.get(str(it["i"]), {}).get("source", it["label"])
        cards.append({"code": code, "frame": f"s{it['i']:02d}.jpg", "sound": plain(source)})
        sound_key[code] = {"i": it["i"], "clip": it["clip"], "family": it["label"], "source": source,
                           "start": it["start"]}
        arm_key[code] = {"arm": arm, "i": it["i"]}
    (out / "cards.json").write_text(json.dumps(cards, indent=1), encoding="utf-8")
    (out.parent / f"{out.name}_SOUND_KEY_open_for_scoring.json").write_text(json.dumps(sound_key, indent=1),
                                                                             encoding="utf-8")
    (out.parent / f"{out.name}_ARM_KEY_do_not_open_until_scored.json").write_text(json.dumps(arm_key, indent=1),
                                                                                  encoding="utf-8")
    print(f"{len(cards)} pass-2 cards -> {out}")


if __name__ == "__main__":
    main()
