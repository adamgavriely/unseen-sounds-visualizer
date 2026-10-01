"""Per-miss ear table for the shipped pipeline SHIP8 on merged DEV (CPU, caches only; no model is run).

For each of the 30 misses in benchmark/gold/ledger_ship8.json: which ear heard the needed sound, raw, before thresholds.
  hit window W = [onset - 0.5, onset + 1.0] (S.EARLY / S.LATE); gold span G = [onset, end]
  1. BEATs frame max, columns with S.same_family(column, sound)          (cross_group.PARTS[part]["beats"])
  2. FlexSED frame max, same_family columns                                (DCC.FLEX_DIR)
  3. DASM max, columns with canonical(column) == canonical(sound)          (as expect_a4_screen.dasm_max; PARTS[part]["dasm"])
  4. whole-clip list ears: Qwen3-Omni (expect_a/listen/<clip>.json items -> expect_a_screen.map_item) and
     Audio Flamingo Next (agree_ears/dev/<clip>.json families); "names it" = same_family(family, sound)
  5. stage-4 SHIP8|proposed rows of the same family (stage4.json), near the sound (start in [onset - 0.5, end] or overlapping G)
     vs elsewhere, and the step that killed each near row (display threshold 0.35, stage-5 augment=false, gate seen, placed outside W)
Bucket (first that applies, D -> C -> B -> A):
  D = a placed same-family picture overlaps G but starts outside W
  C = a near same-family stage-4 row exists but was dropped / gated
  B = some ear hears it (max over W u G: BEATs >= 0.1 or FlexSED >= 0.3 or DASM >= 0.3, or a list ear names it) but no near row
  A = none of the above (no ear above the weak levels)

    TG_ARMS=SHIP8 python benchmark/gold/miss_ears.py         # from ~/MscProj_tg (has every cache of both parts)
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
import config  # noqa: F401
from benchmark.gold import cross_group as CG
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import expect_a_screen as EA
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from src.labels import canonical

G = _ROOT / "benchmark" / "gold"
LEDGER = G / "ledger_ship8.json"
OUT = G / "miss_ears.json"
ARM, SYS = "SHIP8", "proposed"
WEAK = {"beats": 0.1, "flex": 0.3, "dasm": 0.3}
DISPLAY = float(R.arm_cfg(ARM)["DISPLAY_THRESHOLD"])
AUGMENT = float(R.arm_cfg(ARM)["AUGMENT_THRESHOLD"])
TOL = 0.05
BAR_BEATS, BAR_FLEX = float(R.arm_cfg(ARM)["AED_THRESHOLD"]), float(R.arm_cfg(ARM)["FLEXSED_BAR"])


def frmax(fr, match, lo, hi):
    """max over frames lo <= t <= hi of the columns match(label) is True for; None if no column / no frame"""
    if fr is None:
        return None
    fw, t, labs = fr
    cols = [i for i, l in enumerate(labs) if match(l)]
    m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if not cols or not m.any():
        return None
    return round(float(fw[m][:, cols].max()), 3)


def fr_of(d, st):
    p = Path(d) / f"{st}.npz"
    return DCC.load_fr(p) if p.exists() else None


def jload(p):
    p = Path(p)
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def main():
    misses = json.loads(LEDGER.read_text(encoding="utf-8"))["misses"]
    assert len(misses) == 30, len(misses)
    S4 = {pt: json.loads(CG.PARTS[pt]["stage4"].read_text(encoding="utf-8"))["arms"][f"{ARM}|{SYS}"] for pt in CG.PARTS}
    roots = {pt: CG.PARTS[pt]["stage4"].parent / f"{ARM}_{SYS}" for pt in CG.PARTS}
    logs = {pt: jload(roots[pt] / "_stage5_log.json") or {} for pt in CG.PARTS}
    pics = {}
    with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
        for m in misses:
            k = (m["part"], m["clip"])
            if k not in pics:
                pics[k] = S.load_pictures(roots[m["part"]], m["clip"], SYS) or []
    out, notes = [], []
    for m in misses:
        pt, st, snd, at, end = m["part"], m["clip"], m["sound"], float(m["at"]), float(m["end"])
        w0, w1 = at - S.EARLY, at + S.LATE
        fam = canonical(snd)
        spec_ok = S.same_family(snd, snd)
        if not spec_ok:
            notes.append(f"{st} {snd}: same_family(sound, sound) is False (label not specific) -> same_family columns n/a")
        sf = lambda l: spec_ok and S.same_family(l, snd)
        cf = lambda l: canonical(l) == fam
        fb, ff, fd = fr_of(CG.PARTS[pt]["beats"], st), fr_of(DCC.FLEX_DIR, st), fr_of(CG.PARTS[pt]["dasm"], st)
        row = {"part": pt, "clip": st, "sound": snd, "family": fam, "at": at, "end": end,
               "beats_win": frmax(fb, sf, w0, w1), "beats_span": frmax(fb, sf, at, end),
               "flex_win": frmax(ff, sf, w0, w1), "flex_span": frmax(ff, sf, at, end),
               "dasm_win": frmax(fd, cf, w0, w1), "dasm_span": frmax(fd, cf, at, end),
               "cache": {"beats": fb is not None, "flex": ff is not None, "dasm": fd is not None}}
        # whole-clip list ears
        q = jload(G / "expect_a" / "listen" / f"{st}.json")
        if q is None:
            row["qwen_names"], row["qwen_fams"] = None, None
        else:
            qf = sorted({f for f in (EA.map_item(i) for i in q.get("items", [])) if f})
            row["qwen_fams"] = qf
            row["qwen_names"] = any(sf(f) for f in qf)
        a = jload(G / "agree_ears" / "dev" / f"{st}.json")
        if a is None:
            row["afn_names"], row["afn_fams"] = None, None
        else:
            row["afn_fams"] = a.get("families", [])
            row["afn_names"] = any(sf(f) for f in row["afn_fams"])
        # stage-4 rows of the same family
        rows4 = [r for r in S4[pt].get(st, []) if sf(r["label"])]
        near = [r for r in rows4 if (w0 - TOL <= r["start"] <= end + TOL) or (r["start"] <= end and r["end"] >= at)]
        far = [r for r in rows4 if r not in near]
        specs = jload(roots[pt] / st / "augmentations.json") or []
        votes = jload(roots[pt] / st / "gate_votes.json") or []
        lg = logs[pt].get(st, {}) or {}
        gl = [x for k in ("gate_reused_list", "gate_live_list") for x in lg.get(k, [])]
        placed = [(l, float(x), float(y)) for l, x, y in pics[(pt, st)] if sf(l)]
        assert not any(w0 <= x <= w1 for _l, x, _y in placed), (st, snd, placed)   # it is a miss
        led = sorted(round(x, 2) for _l, x in m.get("same_family_pictures", []))
        assert led == sorted(round(x, 2) for _l, x, _y in placed), (st, snd, led, placed)
        fates = []
        for r in near:
            s0, c = float(r["start"]), float(r["conf"])
            if c < DISPLAY:
                fates.append(f"{r['label']}@{s0:.2f} conf {c:.2f}: below display threshold {DISPLAY}"); continue
            sp = [s for s in specs if sf(s["event_label"]) or sf(s.get("detail") or "")]
            sp = [s for s in sp if any(x - TOL <= s0 <= y + TOL or (x <= r["end"] and y >= s0) for x, y in (s.get("spans") or [[s["start"], s["end"]]]))]
            if not sp:
                fates.append(f"{r['label']}@{s0:.2f} conf {c:.2f}: no stage-5 spec holds it (merged/deduped away)"); continue
            s = sp[0]
            if not s.get("augment"):
                fates.append(f"{r['label']}@{s0:.2f} conf {c:.2f}: stage 5 augment=false ({s.get('reason', '')[:80]})"); continue
            seen = [v for v in votes if sf(v["label"]) and v["stretch"][0] - TOL <= s0 <= v["stretch"][1] + TOL and v.get("seen")]
            if seen:
                fates.append(f"{r['label']}@{s0:.2f} conf {c:.2f}: gate seen (named '{seen[0].get('named')}')"); continue
            pl = [p for p in placed if p[1] <= end and p[2] >= at]
            if pl:
                fates.append(f"{r['label']}@{s0:.2f} conf {c:.2f}: placed at {pl[0][1]:.2f}, outside hit window"); continue
            gg = [x for x in gl if sf(x[0]) and x[1] <= max(r["end"], s0) + TOL and x[2] >= s0 - TOL]
            fates.append(f"{r['label']}@{s0:.2f} conf {c:.2f}: augmented, not placed in span"
                         + (f" (stage-5 gate called {gg[0]})" if gg else " (row overflow / display merge)"))
        row["stage4_near"] = [{"label": r["label"], "start": round(float(r["start"]), 2), "end": round(float(r["end"]), 2),
                               "conf": round(float(r["conf"]), 3), "origin": r.get("origin")} for r in near]
        row["stage4_far"] = [f"{r['label']}@{float(r['start']):.2f} ({float(r['conf']):.2f})" for r in far]
        row["kill"] = fates
        row["placed_same_family"] = [f"{l}@{x:.2f}" for l, x, _y in placed]
        mx = lambda *v: max([x for x in v if x is not None], default=None)
        hb, hf, hd = mx(row["beats_win"], row["beats_span"]), mx(row["flex_win"], row["flex_span"]), mx(row["dasm_win"], row["dasm_span"])
        heard = ((hb or 0) >= WEAK["beats"] or (hf or 0) >= WEAK["flex"] or (hd or 0) >= WEAK["dasm"]
                 or bool(row["qwen_names"]) or bool(row["afn_names"]))
        if any(p[1] <= end and p[2] >= at for p in placed):
            b = "D"
        elif near:
            b = "C"
        elif heard:
            b = "B"
        else:
            b = "A"
        row["heard"], row["bucket"] = heard, b
        if b == "B":                                   # no near stage-4 row: below the stage-4 bars, or above a bar and filtered/vetoed
            ab = [n for n, v, bar in (("BEATs", hb, BAR_BEATS), ("FlexSED", hf, BAR_FLEX)) if v is not None and v >= bar]
            row["b_reason"] = (f"above stage-4 bar ({', '.join(ab)}), no row: filtered/vetoed (which veto not recorded)" if ab
                               else f"below stage-4 bars (BEATs < {BAR_BEATS}, FlexSED < {BAR_FLEX})")
        out.append(row)
        print(f"{b} {pt:4s} {st} {snd} {at}-{end}: beats {row['beats_win']}/{row['beats_span']} flex {row['flex_win']}/{row['flex_span']} "
              f"dasm {row['dasm_win']}/{row['dasm_span']} qwen {row['qwen_names']} afn {row['afn_names']} near {len(near)} {fates} {row.get('b_reason', '')}", flush=True)
    counts = {k: sum(1 for r in out if r["bucket"] == k) for k in "ABCD"}
    print("buckets", counts)
    for n in notes:
        print("NOTE", n)
    OUT.write_text(json.dumps({"arm": ARM, "weak": WEAK, "display_threshold": DISPLAY, "stage4_bars": {"beats": BAR_BEATS, "flex": BAR_FLEX}, "augment_threshold": AUGMENT,
                               "window": [-S.EARLY, S.LATE], "buckets": counts, "notes": notes, "misses": out},
                              indent=1, default=float), encoding="utf-8")
    print("->", OUT)


if __name__ == "__main__":
    main()
