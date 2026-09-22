"""Dry secondary row (amendment 5): the gated system under another silence rule, re-decided on
CPU from the raw votes the render logged (data/work/protocol_proposed_<tag>/<clip>/gate_votes.json).

"unanimous" silences a sound only when all three votes say visible in every stretch, so it can
only ADD pictures to the majority render: a sound the majority rule silenced is re-decided from
its logged votes, and a sound silenced as "a kind of" a visible source follows that source. The
added pictures have no image (never generated) — the row is scored on the planned draw list
(require_image=False), approximate by construction (dedup / panel packing not re-run); it is a
secondary row, never the headline.

    python benchmark/gold/gate_redecide.py --tag v4b4 [--work <root>] [--rule unanimous]
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
from benchmark.gold.gate_gold import decide

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"


def redecided_pictures(work: Path, rule: str):
    """on-screen (label, start, end) spans of the proposed system under `rule`, or None"""
    f = work / "augmentations.json"
    if not f.exists():
        return None
    specs = json.loads(f.read_text(encoding="utf-8"))
    votes = json.loads((work / "gate_votes.json").read_text(encoding="utf-8")) if (work / "gate_votes.json").exists() else []
    dur = None
    m = work / "media.json"
    if m.exists():
        dur = float(json.loads(m.read_text(encoding="utf-8")).get("duration") or 0) or None

    def visible(spec) -> bool:
        st = [v for v in votes if v["label"] == spec["event_label"] and abs(v["start"] - spec["start"]) < 0.05]
        if not st:
            return True                          # no votes logged (silenced before the VLM ran): keep as rendered
        return decide(st, rule)

    changed = 0
    by_label = {s["event_label"]: s for s in specs}
    for s in specs:
        r = s.get("reason") or ""
        if s.get("augment") or "visible" not in r:
            continue
        if r.startswith("a kind of "):
            parent = r[len("a kind of "):].split(",")[0].strip()
            p = by_label.get(parent)
            if p is not None and not visible(p):
                s["augment"] = True; changed += 1
        elif r.startswith("source visible on screen") and not visible(s):
            s["augment"] = True; changed += 1
    from src.types import AugmentationSpec
    from src.stage6_visual_augmentation import _display_spans, _assign_rows
    objs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]), end=float(s["end"]),
                             augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)), image_path=s.get("image_path"),
                             talked_about=bool(s.get("talked_about")), spans=[tuple(x) for x in s.get("spans", [])]) for s in specs]
    d = dur or max((o.end for o in objs), default=0.0) + 5.0
    placed, _ = _assign_rows(_display_spans(objs, d, require_image=False))
    return [(lab, float(a), float(b)) for _, lab, a, b, _ in placed], changed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--work", default=None)
    ap.add_argument("--rule", default="unanimous")
    ap.add_argument("--subsets", nargs="+", default=["bench", "dev", "test_bench", "all", "cat_mixed"])
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    work = Path(a.work) if a.work else _ROOT / "data" / "work"
    root = work / f"protocol_proposed_{a.tag}"
    rows_new, rows_old, rows_blind, n_changed = {}, {}, {}, 0
    for stem, snds in gold.items():
        r = redecided_pictures(root / stem, a.rule)
        if r is None:
            continue
        pics, ch = r; n_changed += ch
        rows_new[stem] = S.score_clip(snds, pics)
        rows_old[stem] = S.score_clip(snds, S.load_pictures(root, stem, "proposed"))
        pb = S.load_pictures(work / f"protocol_blind_a2i_{a.tag}", stem, "blind_a2i")
        if pb is not None:
            rows_blind[stem] = S.score_clip(snds, pb)
    subs = S.subsets_of(gold)
    print(f"[{a.rule}] {len(rows_new)} clips, {n_changed} silenced sounds re-shown")
    out = {}
    for sub in a.subsets:
        stems = sorted(st for st in subs.get(sub, set()) if st in rows_new and st in rows_blind)
        if not stems:
            continue
        for name, rows in (("proposed(majority)", rows_old), (f"proposed({a.rule})", rows_new), ("blind_a2i", rows_blind)):
            agg = S.aggregate([rows[s] for s in stems])
            out[f"{sub}|{name}"] = agg
            print(f"--- {sub:12s} [{name:20s}] clips {len(stems):3d} needed {agg['needed']:3d} | P {agg['P']:.2f} R {agg['R']:.2f} F1 {agg['F1']:.2f} | hit {agg['hits']} miss {agg['misses']} vis {agg['visible']} cross {agg['cross']} ph {agg['phantom']}")
        d, lo, hi, p = S.paired_ci([rows_new[s] for s in stems], [rows_blind[s] for s in stems])
        print(f"    dF1 {a.rule} - blind = {d:+.3f} [{lo:+.3f}, {hi:+.3f}] P(d>0)={p:.3f}")
        out[f"{sub}|delta"] = {"dF1": d, "ci": [lo, hi], "p_gt0": p}
    o = _ROOT / "benchmark" / "gold" / f"gate_redecide_{a.tag}_{a.rule}.json"
    o.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", o)


if __name__ == "__main__":
    main()
