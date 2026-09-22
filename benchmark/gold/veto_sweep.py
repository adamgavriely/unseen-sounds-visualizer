"""The second detector as a veto, not only as a source (amendment 10 candidate, 2026-09-23).

FlexSED was added to the union because BEATs was deaf to seven needed sounds. But the diagnostic
(benchmark/gold/name_errors.py) shows it also carries the other half of the information: where BEATs
NAMES something that is not there, FlexSED's score for that same family is near the floor -- Whale
0.13, Horse 0.13, Cat 0.00, Telephone 0.00. The union throws that disagreement away.

Two knobs, deliberately asymmetric so the seven recovered sounds cannot be deleted again:

  veto tau   drop a picture whose family FlexSED never lifts above tau anywhere in the clip. This
             can only touch pictures FlexSED did not itself raise (its own bar is 0.8), i.e. the
             BEATs-only half.
  duty cap d drop a picture whose family sits above FlexSED's bar for >= d of the clip -- the
             "noise of the place", which a picture cannot usefully mark.

Both are swept on DEV only; the chosen pair is frozen in the prereg before TEST is read.

    python benchmark/gold/veto_sweep.py --tag v4b6 --subset dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

FX = _ROOT / "data" / "work" / "flexsed_cache"
BAR = 0.8
_cache = {}


def fx(stem, label):
    """(duty above the bar, peak) for one family in one clip, or None when unknown"""
    if stem not in _cache:
        p = FX / f"{stem}.npz"
        if not p.exists():
            _cache[stem] = None
        else:
            z = np.load(p, allow_pickle=False)
            _cache[stem] = ({str(x): i for i, x in enumerate(z["labels"])}, z["fw"].astype(np.float32))
    c = _cache[stem]
    if c is None or label not in c[0]:
        return None
    v = c[1][c[0][label]]
    return float((v >= BAR).mean()), float(v.max())


def keep(stem, lab, tau, duty):
    f = fx(stem, lab)
    if f is None:
        return True                      # no second opinion -> the union's own decision stands
    d, peak = f
    if peak < tau:
        return False                     # FlexSED never hears this family: BEATs named it alone
    if duty is not None and d >= duty:
        return False                     # above the bar all clip long: the noise of the place
    return True


def run(gold, stems, root, system, tau, duty):
    rows = []
    for stem in stems:
        pics = S.load_pictures(root, stem, system)
        if pics is None:
            continue
        rows.append(S.score_clip(gold[stem], [p for p in pics if keep(stem, p[0], tau, duty)]))
    return rows


def line(name, rows):
    a = S.aggregate(rows)
    return (f"{name:26s} F1 {a['F1']:.3f}  P {a['P']:.3f}  R {a['R']:.3f}  "
            f"FA/clip {S.fa_per_clip(rows):.2f}  cost {S.viewer_cost(rows):.2f}  "
            f"hits {sum(r['hit'] for r in rows):3d}  miss {sum(r['miss'] for r in rows):3d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4b6")
    ap.add_argument("--system", default="proposed")
    ap.add_argument("--subset", default="dev")
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    stems = sorted(set(subs[a.subset]) & set(subs["bench"]))
    root = _ROOT / "data" / "work" / f"protocol_{a.system}_{a.tag}"
    have = [s for s in stems if S.load_pictures(root, s, a.system) is not None]
    print(f"== {a.subset}: {len(have)} clips, tag {a.tag}\n")
    base = run(gold, have, root, a.system, 0.0, None)
    print(line("union (no veto)", base))
    print(line("silence", [S.score_clip(gold[s], []) for s in have]))
    print()
    out = {}
    for tau in (0.0, 0.1, 0.2, 0.3, 0.5):
        for duty in (None, 0.7, 0.5, 0.4):
            if tau == 0.0 and duty is None:
                continue
            rows = run(gold, have, root, a.system, tau, duty)
            nm = f"tau {tau:.1f} duty {duty if duty else '-'}"
            print(line(nm, rows))
            a2 = S.aggregate(rows)
            out[nm] = {"F1": a2["F1"], "P": a2["P"], "R": a2["R"], "cost": S.viewer_cost(rows),
                       "fa": S.fa_per_clip(rows), "hits": sum(r["hit"] for r in rows),
                       "miss": sum(r["miss"] for r in rows)}
        print()
    (_ROOT / "benchmark" / "gold" / f"veto_sweep_{a.subset}_{a.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
