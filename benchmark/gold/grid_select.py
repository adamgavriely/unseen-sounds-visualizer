"""Apply amendment 10's corrected selection rule to the DEV grid, all cells at once.

The rule, as corrected on 2026-09-23 (docs/prereg_v4.md, "Amendment 10, second clarification"): the
cell with the LOWEST viewer cost per clip AMONG those that keep DEV hits within 2 of the current
v4b6 setting -- the go/no-go's own recall cap, applied at selection instead of only after it. Ties
break on fewer misses, then on the smaller change from the current setting.

The bar 0.8 / tau 0.3 cell is a REPLICATION of the scorer-side result (3.59 on DEV), not a selection
step: it exists to show that vetoing events inside stage 4 gives what vetoing pictures in the scorer
gave. If it does not, every other cell has to be re-read, because removing an event also changes the
gate's per-stretch work and the panel's row assignment.

    python benchmark/gold/grid_select.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import GOLD

CELLS = [("v4b6 (current, no veto)", "protocol_proposed_v4b6", None),
         ("bar 0.8 + veto  [replication]", "protocol_proposed_dev_b8_v30", 3.59),
         ("bar 0.7 + veto", "protocol_proposed_dev_b7_v30", None),
         ("bar 0.6 + veto", "protocol_proposed_dev_b6_v30", None),
         ("per-family bars + veto", "protocol_proposed_dev_fam_v30", None)]
RECALL_SLACK = 2


def complete(root: Path, stems) -> tuple:
    """(clips with a finished render, augmented specs, those whose picture file exists).

    A work directory is created when a clip STARTS, and augmentations.json is written once after
    gating and again after the pictures are made, so neither proves a cell finished. The only sound
    test is that every spec marked augment has an image_path pointing at a file that is there --
    otherwise the cell scores as near-silence and, on a benchmark where most clips should show
    nothing, that looks like a win rather than a bug.
    """
    n = shown = have = 0
    for st in stems:
        f = root / st / "augmentations.json"
        if not f.exists():
            continue
        n += 1
        for x in json.loads(f.read_text(encoding="utf-8")):
            if not x.get("augment"):
                continue
            shown += 1
            ip = x.get("image_path")
            if ip and Path(ip).exists():
                have += 1
    return n, shown, have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default="dev")
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    stems = sorted(set(subs[a.subset]) & set(subs["bench"]))
    w = _ROOT / "data" / "work"
    res = {}
    print(f"== {a.subset}: {len(stems)} clips\n")
    print(f"{'cell':32s} {'clips':>5s} {'F1':>6s} {'P':>6s} {'R':>6s} {'FA/clip':>8s} {'cost':>6s} {'hits':>5s} {'miss':>5s}")
    for name, tag, expect in CELLS:
        root = w / tag
        rows = [S.score_clip(gold[s], p) for s in stems
                for p in [S.load_pictures(root, s, "proposed")] if p is not None]
        if len(rows) < len(stems):
            print(f"{name:32s} {len(rows):5d}  -- incomplete, not scored --")
            continue
        n, shown, have = complete(root, stems)
        if shown and have < shown - 2:
            print(f"{name:32s} {len(rows):5d}  -- STILL RENDERING: {have}/{shown} pictures exist, not scored --")
            continue
        agg = S.aggregate(rows)
        hits = sum(r["hit"] for r in rows)
        res[name] = {"F1": agg["F1"], "P": agg["P"], "R": agg["R"], "cost": S.viewer_cost(rows),
                     "fa": S.fa_per_clip(rows), "hits": hits, "miss": sum(r["miss"] for r in rows)}
        print(f"{name:32s} {len(rows):5d} {agg['F1']:6.3f} {agg['P']:6.3f} {agg['R']:6.3f} "
              f"{S.fa_per_clip(rows):8.2f} {S.viewer_cost(rows):6.2f} {hits:5d} {res[name]['miss']:5d}")
        if expect is not None:
            d = abs(S.viewer_cost(rows) - expect)
            print(f"{'':32s}        replication of the scorer-side {expect:.2f}: "
                  f"{'MATCHES' if d <= 0.3 else 'DOES NOT MATCH'} (difference {d:.2f})")
    sil = S.viewer_cost([S.score_clip(gold[s], []) for s in stems])
    print(f"{'silence (show nothing)':32s} {len(stems):5d} {0:6.3f} {0:6.3f} {0:6.3f} {0:8.2f} {sil:6.2f}")
    base = res.get("v4b6 (current, no veto)")
    if base and len(res) > 1:
        ok = {k: v for k, v in res.items() if k != "v4b6 (current, no veto)" and v["hits"] >= base["hits"] - RECALL_SLACK}
        print(f"\neligible under the recall cap (hits >= {base['hits'] - RECALL_SLACK}): {sorted(ok)}")
        if ok:
            pick = min(ok.items(), key=lambda kv: (round(kv[1]["cost"], 4), kv[1]["miss"]))
            print(f"SELECTED: {pick[0]}  cost {pick[1]['cost']:.2f}  hits {pick[1]['hits']}  miss {pick[1]['miss']}")
        else:
            print("SELECTED: none -- every cell loses more than the recall cap allows; v4b6 stands")
    (_ROOT / "benchmark" / "gold" / f"grid_{a.subset}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
