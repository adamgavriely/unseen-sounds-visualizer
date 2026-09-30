"""RT screen (re-time instead of drop), CPU only, on the saved SHIP4+BTP (= SHIP5) proposed pictures of merged DEV
(old DEV 49 + tagger dev2 22). A drawn non-rescued picture whose family's DASM frame score stays < BAR (0.575) inside
[start - 0.5, end + 0.5] (the DV-G condition) gets its start moved to the time of the family's DASM maximum within
[start - 3, start + 3] if that maximum is >= BAR; otherwise it is left unchanged (no DASM column / no frames -> unchanged).
The end is kept (pushed to keep the duration if the new start passes it; scoring is onset-only). Rescored with
score_per_sound; nothing in src/ or config.py is edited.
Pre-registered GO bar: 0 current hits lost, cross wrong down by >= 3, total wrong not up.

    python benchmark/gold/rt_screen.py        (from ~/MscProj_tg)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import round13_dev as R
from benchmark.gold import score_per_sound as S
from benchmark.gold.btp_screen import placed, summ, fmt
from src.labels import canonical

ARM, SYS = "SHIP4+BTP", "proposed"
BAR, PAD, WIN = 0.575, 0.5, 3.0
OUT = _ROOT / "benchmark" / "gold" / "rt_screen.json"


def parts():
    """btp_screen.parts() for ARM, also returning each part's DASM dir (tagger_prep.configure redirects DCC.DASM_DIR)"""
    gold, stems = DCC.dev_stems()
    ddev = Path(DCC.DASM_DIR)
    with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
        dev = [("dev", st, gold[st], placed(R.R13 / f"{ARM}_{SYS}", st), ddev) for st in stems]
    from benchmark.gold import tagger_prep as T
    DCC2, R2, stems2 = T.configure("dev2")
    ddev2 = Path(DCC2.DASM_DIR)
    keep = set(stems2)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = T.out("dev2") / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    g2 = T._REAL_LOAD_GOLD([tmp])
    o = T.out("dev2")
    with R2.flags({k: R2.arm_cfg(ARM)[k] for k in R2.DISPLAY_KEYS}):
        dev2 = [("dev2", st, g2[st], placed(o / f"{ARM}_{SYS}", st), ddev2) for st in stems2 if st in g2]
    return dev + dev2


DVG = []                                                 # every DV-G picture: (label, start, max inside, max in +-3 s)


def retime(pics, dasm):
    new, changed = [], []
    for lab, a, b, resc in pics:
        a2, b2 = a, b
        if dasm is not None and not resc:
            fw, t, labs = dasm
            cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(lab)]
            m = (t >= a - PAD) & (t <= b + PAD)
            if cols and m.any() and float(fw[m][:, cols].max()) < BAR:
                w = (t >= a - WIN) & (t <= a + WIN)
                if w.any():
                    sc = fw[:, cols].max(axis=1)
                    DVG.append((lab, round(a, 2), round(float(fw[m][:, cols].max()), 3), round(float(sc[w].max()), 3)))
                    idx = np.flatnonzero(w)
                    k = idx[int(np.argmax(sc[idx]))]
                    if float(sc[k]) >= BAR:
                        a2 = float(t[k])
                        if a2 >= b:
                            b2 = a2 + (b - a)
        new.append((lab, a2, b2))
        if a2 != a:
            changed.append((lab, a, a2))
    return new, changed


def main():
    P = parts()
    print(f"{len(P)} clips, {sum(len(p[3]) for p in P)} placed pictures, {sum(r for p in P for *_y, r in p[3])} rescued; "
          f"arm {ARM}; DASM dirs {sorted({str(p[4]) for p in P})}")
    b_rows = {"dev": [], "dev2": []}; x_rows = {"dev": [], "dev2": []}; ch, lost, nod = [], [], 0
    oc = lambda r: f"h{r['hit']} v{r['visible']} c{r['cross']} p{r['phantom']} d{r['dup']}"
    for pt, st, g, pics, dd in P:
        f = dd / f"{st}.npz"
        dasm = DCC.load_fr(f) if f.exists() else None
        nod += dasm is None
        new, changed = retime(pics, dasm)
        r0 = S.score_clip(g, [p[:3] for p in pics]); r1 = S.score_clip(g, new)
        b_rows[pt].append(r0); x_rows[pt].append(r1)
        if changed:
            ch.append({"part": pt, "clip": st, "changes": [[l, round(x, 2), round(y, 2)] for l, x, y in changed],
                       "before": oc(r0), "after": oc(r1)})
        if r1["hit"] < r0["hit"]:
            lost.append((st, r0["hit"] - r1["hit"]))
    B = {"merged": summ(b_rows["dev"] + b_rows["dev2"]), "dev": summ(b_rows["dev"]), "dev2": summ(b_rows["dev2"])}
    X = {"merged": summ(x_rows["dev"] + x_rows["dev2"]), "dev": summ(x_rows["dev"]), "dev2": summ(x_rows["dev2"])}
    go = not lost and X["merged"]["cross"] <= B["merged"]["cross"] - 3 and X["merged"]["wrong"] <= B["merged"]["wrong"]
    print(f"clips with no DASM cache: {nod}; DV-G pictures {len(DVG)}, best +-3 s DASM max {max((d[3] for d in DVG), default=0)}")
    print(f"BASE {ARM}: merged {fmt(B['merged'])} | DEV {fmt(B['dev'])} | DEV2 {fmt(B['dev2'])}")
    print(f"RT:  merged {fmt(X['merged'])} | DEV {fmt(X['dev'])} | DEV2 {fmt(X['dev2'])}  hits lost {lost} -> {'GO' if go else 'STOP'}")
    for c in ch:
        print(f"   {c['part']:4s} {c['clip']}: {c['changes']}  {c['before']} -> {c['after']}")
    OUT.write_text(json.dumps({"arm": ARM, "bar": BAR, "base": B, "rt": X, "changed": ch, "hits_lost": lost, "GO": go,
                               "no_dasm": nod, "dvg": DVG}, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
