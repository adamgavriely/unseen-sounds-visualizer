"""Round 58b FLEX-WITNESS step A (mechanism only, merged DEV caches; CPU; no model run). Run from ~/MscProj_tg.

For the 7 pictures Round 53 WEAK-WITNESS changed (5 removed wrongs, 2 lost hits; weakwitness_diff.json): FlexSED's own-family frame
max (DCC.FLEX_DIR, columns with S.same_family(column, label)) within the picture span [start, end] and within [start-0.5, end+0.5].
label = the picture label and, separately, each same-canonical-family SHIP8+MD3|proposed stage-4 row label near the picture.
    python benchmark/gold/flexwitness_dev.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8+MD3")
from benchmark.gold import cross_group as CG  # noqa: E402
from benchmark.gold import dev_candidates_check as DCC  # noqa: E402
from benchmark.gold import score_per_sound as S  # noqa: E402
from src.labels import canonical  # noqa: E402

SPANS = [  # (part, clip, picture label, start, end, Round 53 effect)
    ("dev", "b3_laundromat", "Train", 1.0, 2.5, "removed wrong (phantom)"),
    ("dev", "mv_protest_scene_movie", "Glass", 4.75, 6.25, "removed wrong (cross)"),
    ("dev2", "tg_d022", "Dog", 7.25, 8.75, "removed wrong (cross)"),
    ("dev2", "tg_d107", "Screaming", 6.52, 9.0, "removed wrong (cross)"),
    ("dev2", "tg_d128", "Hammer", 9.0, 10.0, "removed wrong (cross)"),
    ("dev2", "tg_d032", "Thunder", 13.75, 15.0, "lost hit"),
    ("dev2", "tg_d075", "Alarm", 0.14, 9.25, "lost hit"),
]
ARM = "SHIP8+MD3|proposed"


def frmax(fr, match, lo, hi):
    fw, t, labs = fr
    cols = [i for i, l in enumerate(labs) if match(l)]
    m = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if not cols or not m.any():
        return None, []
    return round(float(fw[m][:, cols].max()), 3), [labs[i] for i in cols]


def main():
    s4 = {pt: json.loads(CG.PARTS[pt]["stage4"].read_text(encoding="utf-8"))["arms"].get(ARM, {}) for pt in CG.PARTS}
    out = []
    for pt, st, lab, a, b, eff in SPANS:
        p = Path(DCC.FLEX_DIR) / f"{st}.npz"
        fr = DCC.load_fr(p) if p.exists() else None
        rr = s4[pt].get(st, [])
        rr = rr.get("rows", rr) if isinstance(rr, dict) else rr
        rows = sorted({r["label"] for r in rr if canonical(r["label"]) == canonical(lab)
                       and float(r.get("pre_start", r["start"])) - 1.0 <= a <= float(r["end"]) + 0.5})
        rec = {"part": pt, "clip": st, "picture": [lab, a, b], "effect": eff, "cache": fr is not None,
               "flex_labels": None if fr is None else fr[2], "stage4_row_labels": rows, "by_label": {}}
        if fr is not None:
            for L in [lab] + [r for r in rows if r != lab]:
                sp, cols = frmax(fr, lambda c: S.same_family(c, L), a, b)
                pm, _ = frmax(fr, lambda c: S.same_family(c, L), a - 0.5, b + 0.5)
                cf, ccols = frmax(fr, lambda c: canonical(c) == canonical(L), a - 0.5, b + 0.5)
                rec["by_label"][L] = {"span": sp, "span_pm05": pm, "cols": cols, "canon_pm05": cf, "canon_cols": ccols}
        out.append(rec)
        print(pt, st, lab, a, b, eff, "| rows", rows, "|", rec["by_label"], flush=True)
    o = _ROOT / "benchmark" / "gold" / "flexwitness_dev.json"
    o.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(o)


if __name__ == "__main__":
    main()
