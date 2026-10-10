"""Stability of the screen-reasoning "off screen" question (Adam 10 Oct; screen_reason.py gave out of fold 58 / 25).
Same frames rule (1 s before the start to 1 s after min(end, start + 4) s), Qwen3.8-27B thinking off, bias-cancelled
yes/no margin (question minus twin), three wordings x two frame counts (6, 10):
  o1  the screen_reason.py wording
  o2  "Can the viewer see what is making the {l} sound in these frames?" (sign flipped: high = seen)
  o3  "Someone hears a {l} sound while watching these frames. Is its source hidden or outside the picture?"
Per variant: drop when margin < t (o1, o3) or > t (o2), bar from quantiles; clip-grouped 5-fold CV (seed 0) on all 158
clips by onset cost; then the mean of the six z-scored margins (o2 negated).
    GPU: python benchmark/gold/coverage/offscreen_var.py --score  (offscreen_var.json)
    CPU: python benchmark/gold/coverage/offscreen_var.py          -> offscreen_var.md"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics, TG, Q as Q1

HERE = Path(__file__).resolve().parent
FEAT = HERE / "offscreen_var.json"
Q = {"o1": (Q1["off"][0], Q1["off"][1], +1),
     "o2": ("Can the viewer see what is making the {l} sound in these frames? Answer yes or no.",
            "Is what makes the {l} sound impossible for the viewer to see in these frames? Answer yes or no.", -1),
     "o3": ("Someone hears a {l} sound while watching these frames. Is its source hidden or outside the picture? Answer yes or no.",
            "Someone hears a {l} sound while watching these frames. Is its source shown in the picture? Answer yes or no.", +1)}


def score(pics):
    import cv2
    from PIL import Image
    from src.stage5_cross_modal_analysis import reason as R
    config.VLM_MODEL = "Qwen/Qwen3.8-27B"; config.VLM_THINKING = False
    mdl, proc = R._load(config.VLM_MODEL, "cuda")
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text())
    out = json.loads(FEAT.read_text()) if FEAT.exists() else {}
    for st, ps in pics.items():
        cap = cv2.VideoCapture(str(TG / vids[st])); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for lab, a, b in ps:
            k = f"{st}|{lab}|{a:.3f}"
            if k in out:
                continue
            lo, hi = a - 1.0, min(b, a + 4.0) + 1.0
            l = lab.split(",")[0].split("(")[0].strip().lower(); r = {}
            for nf in (6, 10):
                ims = []
                for i in range(nf):
                    f = int(max(0, min(n - 1, (lo + (hi - lo) * i / (nf - 1)) * fps))); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
                    if ok:
                        im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); im.thumbnail((640, 640)); ims.append(im)
                for q, (p, t, _s) in Q.items():
                    r[f"{q}_{nf}"] = (R._yes_no_margin(mdl, proc, p.format(l=l), ims) - R._yes_no_margin(mdl, proc, t.format(l=l), ims)) if ims else None
            out[k] = r
            print(k, r, flush=True)
            FEAT.write_text(json.dumps(out))


def main():
    gold = S.load_gold([GOLD]); pics = base_pics()
    if "--score" in sys.argv:
        score(pics); return
    F = json.loads(FEAT.read_text()); allc = list(pics)
    keys = [f"{q}_{nf}" for q in Q for nf in (6, 10)]
    sd = {k: np.std([v[k] for v in F.values() if v.get(k) is not None]) for k in keys}
    for v in F.values():                                           # high = off screen for every variant
        xs = [Q[k[:2]][2] * v[k] / sd[k] for k in keys if v.get(k) is not None]
        v["mean"] = float(np.mean(xs)) if xs else None
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    base = {st: S.score_clip(gold[st], pics[st]) for st in allc}
    L = ["# Off-screen question: wordings x frame counts, all 158 clips (v1.4: 59 / 34, 2.076)", "",
         "| variant | out of fold hits | wrong | cost | vs v1.4 | p |", "|---|---|---|---|---|---|"]
    for k in keys + ["mean"]:
        sgn = 1 if k == "mean" else Q[k[:2]][2]
        vals = np.array([v[k] for v in F.values() if v.get(k) is not None])
        grid = [None] + [float(x) for x in np.quantile(vals, np.linspace(0.02, 0.4, 12) if sgn > 0 else np.linspace(0.6, 0.98, 12))]

        def drop(st, p, t):
            v = F.get(f"{st}|{p[0]}|{p[1]:.3f}", {}).get(k)
            return t is not None and v is not None and (v < t if sgn > 0 else v > t)
        memo = {}
        row = lambda st, t: memo.setdefault((st, t), S.score_clip(gold[st], [p for p in pics[st] if not drop(st, p, t)]))
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof = {}
        for f in range(5):
            te = set(sh[f::5]); tr = [c for c in allc if c not in te]
            best = min(grid, key=lambda t: cost([row(st, t) for st in tr]))
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        d = np.array([S.viewer_cost([oof[st]]) - S.viewer_cost([base[st]]) for st in allc])
        m = d[np.random.default_rng(0).integers(0, len(d), (100000, len(d)))].mean(1)
        L.append(f"| {k} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} | {d.mean():+.3f} | {min(1, 2 * min((m >= 0).mean(), (m <= 0).mean())):.3f} |")
    (HERE / "offscreen_var.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
