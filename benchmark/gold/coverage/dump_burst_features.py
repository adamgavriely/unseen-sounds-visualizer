"""Step 4 (PREREG_step4_keep_score.md): detector and listener features 1-18 for every DEV burst. Cluster, CPU.

    python benchmark/gold/coverage/dump_burst_features.py   (verify_items_dev.json -> scratch_cov/burst_features_dev.json)
"""
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold.coverage.dump_evidence import CACHES
from src.labels import canonical

G = _ROOT / "benchmark" / "gold"


def fam_cols(labs, fam):
    return [i for i, c in enumerate(labs) if S.same_family(c, fam) or canonical(c) == fam]


def stat(fr, fam, a, b, pad=0.0):
    if fr is None:
        return None, None, None
    fw, t, labs = fr
    cols = fam_cols(labs, fam)
    if not cols:
        return None, None, None
    v = fw[:, cols].max(axis=1)
    m = (t >= a - pad) & (t <= b + pad)
    if not m.any():
        m[int(np.argmin(np.abs(t - 0.5 * (a + b))))] = True
    return float(v[m].max()), float(v[m].mean()), float(v.max())


def items(name):
    out = []
    for f in (G / f"dev_{name}.json", G / f"dev2_{name}.json"):
        if f.exists():
            out += json.loads(f.read_text(encoding="utf-8"))["items"]
    return out


def main():
    it = json.loads((Path(__file__).resolve().parent / "verify_items_dev.json").read_text(encoding="utf-8"))
    part = {}
    from benchmark.gold import inspector_trail_export as X
    for split, p, base, stems in X.parts("SHIP8+MD3+WW5+SL"):
        for st in stems:
            part[st] = p
    LQ, LA = items("listener"), items("listener_afn")
    res, frs, cur = {}, {}, None
    for b in it["bursts"]:
        st, fam, a, e = b["clip"], b["family"], float(b["start"]), float(b["end"])
        if st != cur:
            cur = st
            frs = {m: (DCC.load_fr(d / f"{st}.npz") if (d / f"{st}.npz").exists() else None) for m, d in CACHES[part[st]].items()}
        bm, bmean, bclip = stat(frs["beats"], fam, a, e)
        fm, fmean, fclip = stat(frs["flex"], fam, a, e)
        dm, _, dclip = stat(frs["dasm"], fam, a, e, pad=0.5)
        sm = None
        if frs["beats"] is not None:
            fw, t, labs = frs["beats"]
            cols = [i for i, c in enumerate(labs) if c in ("Speech", "Music")]
            m = (t >= a) & (t <= e)
            if not m.any():
                m[int(np.argmin(np.abs(t - 0.5 * (a + e))))] = True
            sm = float(fw[m][:, cols].max()) if cols else None
        ov = lambda x: x["clip"] == st and canonical(x["family"]) == fam and x["end"] > a and x["start"] < e
        q = [x["score"] for x in LQ if ov(x) and x.get("score") is not None]
        af = [x for x in LA if ov(x) and isinstance(x.get("accept"), dict)]
        res[b["id"]] = {"beats_max": bm, "beats_mean": bmean, "flex_max": fm, "flex_mean": fmean, "dasm_max": dm,
                        "beats_clip": bclip, "flex_clip": fclip, "dasm_clip": dclip, "speech_music": sm,
                        "listener_q": max(q) if q else None,
                        "af_v4": float(any(x["accept"].get("V4") for x in af)) if af else None,
                        "af_yn": float(any(x["accept"].get("YN0") for x in af)) if af else None}
    out = _ROOT / "scratch_cov" / "burst_features_dev.json"
    out.write_text(json.dumps(res), encoding="utf-8")
    print(len(res), "->", out)


if __name__ == "__main__":
    main()
