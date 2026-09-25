"""GP-4 screening sitting 3(a): the generator arms on picture-DEV (the 54 round-2 sounds + the 33 DEV bench
sounds), arm hidden, shuffled codes, 10 repeats. Same keys as round 2: a SOUND key opened for scoring and an
ARM key opened only after scoring. The page itself is built locally as an artifact (glance fidelity).

    python benchmark/gold/rating_screen.py --arms G_q2512,G_q21,G_q21rgba --out data/work/rate_screen1
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from PIL import Image

_ROOT = Path(__file__).resolve().parent.parent.parent
BENCHES = {"fresh": "data/work/picture_bench_fresh", "dev": "data/work/picture_bench"}
N_REPEATS = 10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=25092027)
    ap.add_argument("--bench", default="", help="one bench folder instead of the two screening benches")
    ap.add_argument("--subjects", default="V31")
    ap.add_argument("--repeats", type=int, default=N_REPEATS)
    ap.add_argument("--skip-cards", default="", help="arm whose manifest marks burst cards (CARD:...) to leave out")
    ap.add_argument("--prefix", default="S")
    a = ap.parse_args()
    out = _ROOT / a.out
    (out / "img").mkdir(parents=True, exist_ok=True)
    pool = []
    benches = {"confirm": a.bench} if a.bench else BENCHES
    cards = set()
    if a.skip_cards:
        man = json.loads((_ROOT / a.bench / a.skip_cards / "manifest.json").read_text(encoding="utf-8"))
        cards = {m["i"] for m in man if str(m.get("prompt", "")).startswith("CARD:")}
    for bname, bdir in benches.items():
        bench = _ROOT / bdir
        specs = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
        subj = json.loads((bench / f"subjects_{a.subjects}.json").read_text(encoding="utf-8"))
        for arm in a.arms.split(","):
            for it in specs:
                if it["i"] in cards:
                    continue                          # burst cards are counted separately, never rated
                p = bench / arm / f"{it['i']:02d}.png"
                if p.exists():
                    s = subj.get(str(it["i"]), {})
                    pool.append((arm, bname, it, s.get("source", it["label"]), p))
    rng = random.Random(a.seed)
    rng.shuffle(pool)
    order = list(pool)
    for k in sorted(rng.sample(range(len(pool) // 2), min(a.repeats, len(pool) // 2)), reverse=True):
        at = min(len(order), k + len(pool) // 3 + rng.randint(0, len(pool) // 3))
        order.insert(at, pool[k])
    sound_key, arm_key = {}, {}
    for n, (arm, bname, it, source, p) in enumerate(order, 1):
        code = f"{a.prefix}{n:03d}"
        Image.open(p).convert("RGB").resize((384, 384)).save(out / "img" / f"{code}.jpg", quality=86)
        sound_key[code] = {"bench": bname, "i": it["i"], "clip": it["clip"], "family": it["label"],
                           "source": source, "start": it["start"]}
        arm_key[code] = {"arm": arm, "bench": bname, "i": it["i"]}
    (out.parent / f"{out.name}_SOUND_KEY_open_for_scoring.json").write_text(json.dumps(sound_key, indent=1), encoding="utf-8")
    (out.parent / f"{out.name}_ARM_KEY_do_not_open_until_scored.json").write_text(json.dumps(arm_key, indent=1), encoding="utf-8")
    print(f"{len(order)} cards ({len(pool)} pictures + {len(order) - len(pool)} repeats) -> {out}")


if __name__ == "__main__":
    main()
