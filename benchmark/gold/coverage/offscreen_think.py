"""Off-screen question with thinking (Adam 10 Oct; yes/no margins without thinking moved with wording / frames, see
video_stability.md). Qwen3.8-27B, config.VLM_THINKING = True (reason.py _ask: reasons, then the last line is the answer),
wordings o1 and o3 of offscreen_var.py, frames 6 and 10 (1 s before the start to 1 s after min(end, start + 4) s,
640 px). Each answer is "off" (source off screen) or "on". A picture is dropped when at least k of its 4 answers say
"on"; k in {off, 1, 2, 3, 4}; clip-grouped 5-fold CV (seed 0) on all 158 clips by onset cost; agreement between the
four answers is reported as the stability measure.
    GPU: python benchmark/gold/coverage/offscreen_think.py --score  (offscreen_think.json)
    CPU: python benchmark/gold/coverage/offscreen_think.py          -> offscreen_think.md"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics, TG
from benchmark.gold.coverage.offscreen_var import Q as QV

HERE = Path(__file__).resolve().parent
FEAT = HERE / "offscreen_think.json"
Q = {"o1": QV["o1"][0], "o3": QV["o3"][0]}            # both: "yes" = off screen


def score(pics):
    import cv2
    from PIL import Image
    from src.stage5_cross_modal_analysis import reason as R
    config.VLM_MODEL = "Qwen/Qwen3.8-27B"; config.VLM_THINKING = True; config.VLM_THINKING_TOKENS = 768
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
                for q, p in Q.items():
                    ans = R._ask(mdl, proc, p.format(l=l), images=ims, max_new=8).strip().lower() if ims else ""
                    r[f"{q}_{nf}"] = "off" if ans.startswith("y") else "on" if ans.startswith("n") else None
            out[k] = r
            print(k, r, flush=True)
            FEAT.write_text(json.dumps(out))


def main():
    gold = S.load_gold([GOLD]); pics = base_pics()
    if "--score" in sys.argv:
        score(pics); return
    F = json.loads(FEAT.read_text()); allc = list(pics)
    non = lambda st, p: sum(v == "on" for v in F.get(f"{st}|{p[0]}|{p[1]:.3f}", {}).values())
    GRID = [None, 1, 2, 3, 4]
    memo = {}
    row = lambda st, k: memo.setdefault((st, k), S.score_clip(gold[st], [p for p in pics[st] if k is None or non(st, p) < k]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    vals = [list(v.values()) for v in F.values()]
    agree = np.mean([len(set(x for x in v if x)) == 1 for v in vals])
    pair = {f"{a} vs {b}": np.mean([v[a] == v[b] for v in F.values() if v.get(a) and v.get(b)])
            for a, b in (("o1_6", "o1_10"), ("o1_6", "o3_6"), ("o3_6", "o3_10"), ("o1_10", "o3_10"))}
    L = ["# Off-screen question with thinking, all 158 clips (v1.4: 59 / 34, 2.076)", "",
         f"pictures: {len(F)}; all four answers agree on {agree:.0%}; pairwise agreement " + ", ".join(f"{k} {v:.0%}" for k, v in pair.items()), "",
         "| drop when >= k answers say on screen | hits | wrong | cost |", "|---|---|---|---|"]
    for k in GRID:
        rr = [row(st, k) for st in allc]; L.append(f"| {k or 'v1.4'} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    for one in ("o1_6", "o1_10", "o3_6", "o3_10"):
        rr = [S.score_clip(gold[st], [p for p in pics[st] if F.get(f"{st}|{p[0]}|{p[1]:.3f}", {}).get(one) != "on"]) for st in allc]
        L.append(f"| only {one} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for f in range(5):
        te = set(sh[f::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda k: cost([row(st, k) for st in tr])); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    rr = [oof[st] for st in allc]
    d = np.array([S.viewer_cost([oof[st]]) - S.viewer_cost([row(st, None)]) for st in allc])
    m = d[np.random.default_rng(0).integers(0, len(d), (100000, len(d)))].mean(1)
    L += ["", f"CV choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}; vs v1.4 {d.mean():+.3f} "
          f"[{np.percentile(m, 2.5):+.3f}, {np.percentile(m, 97.5):+.3f}], p {min(1, 2 * min((m >= 0).mean(), (m <= 0).mean())):.3f}"]
    (HERE / "offscreen_think.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
