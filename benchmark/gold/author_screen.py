"""GP-4 amendment: the author's blind screening of the arms on picture-DEV (the 54 round-2 sounds).

Every picture of every arm, shuffled together under codes, tiled 12 per sheet at 256 px with only the code
printed; the arm key is written to a separate file the author opens only after typing his answers
(`author_answers.json`, {code: text | "__cant__"}), which are then scored with the committed round-2 answer
sheet (`score_answers.py score/unseal`, arm names prefixed so the unseal table reads per arm).

    python benchmark/gold/author_screen.py tiles --arms G_q2512,G_q21,G_q21rgba,G_q2512_R,G_q2512_RT,G_q2512_X
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

_ROOT = Path(__file__).resolve().parent.parent.parent
BENCH = _ROOT / "data" / "work" / "picture_bench_fresh"
OUT = _ROOT / "data" / "work" / "author_screen"


def cmd_tiles(a):
    pool = []
    specs = json.loads((BENCH / "specs.json").read_text(encoding="utf-8"))
    for arm in a.arms.split(","):
        for it in specs:
            p = BENCH / arm / f"{it['i']:02d}.png"
            if p.exists():
                pool.append((arm, it["i"], p))
    random.Random(a.seed).shuffle(pool)
    OUT.mkdir(parents=True, exist_ok=True)
    sound_key, arm_key = {}, {}
    per, W = 12, 256
    for s in range(0, len(pool), per):
        chunk = pool[s:s + per]
        sheet = Image.new("RGB", (4 * (W + 8), 3 * (W + 30)), "white")
        d = ImageDraw.Draw(sheet)
        for k, (arm, i, p) in enumerate(chunk):
            code = f"A{s + k + 1:03d}"
            x, y = (k % 4) * (W + 8), (k // 4) * (W + 30)
            sheet.paste(Image.open(p).convert("RGB").resize((W, W)), (x, y + 22))
            d.text((x + 4, y + 4), code, fill=(0, 0, 0))
            sound_key[code] = {"i": i}
            arm_key[code] = {"arm": arm, "i": i}
        sheet.save(OUT / f"sheet_{s // per + 1:02d}.jpg", quality=85)
    (OUT / "SOUND_KEY.json").write_text(json.dumps(sound_key, indent=1), encoding="utf-8")
    (OUT.parent / "author_screen_ARM_KEY_do_not_open_until_scored.json").write_text(json.dumps(arm_key, indent=1),
                                                                                  encoding="utf-8")
    print(f"{len(pool)} pictures on {(len(pool) + per - 1) // per} sheets -> {OUT}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tiles"])
    ap.add_argument("--arms", required=True)
    ap.add_argument("--seed", type=int, default=26092026)
    a = ap.parse_args()
    cmd_tiles(a)


if __name__ == "__main__":
    main()
