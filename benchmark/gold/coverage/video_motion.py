"""Video ideas 2 + 4 (Adam 10 Oct: "tried a lot on the audio, not enough on the video"). For every v1.4 picture, from
the frames at 10 per second (grey, 160 px wide):
  spike  motion at the sound's start: mean |frame difference| in [start - 0.3, start + 0.5] s divided by the clip's median
         frame difference (a door shown slamming, a visible gun recoiling -> on screen)
  cut    camera cut at the start: largest grey-histogram distance (1 - correlation) between consecutive frames in
         [start - 0.5, start + 0.5] s (a sound that begins on a cut usually belongs to the new shot)
A picture is dropped when spike > t or cut > t. Clip-grouped 5-fold CV (seed 0), all 158 clips, by onset cost and by
"no hit lost" (most wrongs removed among settings that lose no training hit). Cluster CPU from ~/wt_slice.
    python benchmark/gold/coverage/video_motion.py -> video_motion.md (features cached in video_motion.json)"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics, TG

HERE = Path(__file__).resolve().parent
FEAT = HERE / "video_motion.json"


def features(pics):
    import cv2
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text())
    out = {}
    for st, ps in pics.items():
        cap = cv2.VideoCapture(str(TG / vids[st])); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        step = max(1, int(round(fps / 10))); frames, i = [], 0
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            if i % step == 0:
                g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY); h = int(g.shape[0] * 160 / g.shape[1])
                frames.append(cv2.resize(g, (160, max(h, 1))).astype(np.float32))
            i += 1
        if len(frames) < 3:
            continue
        t = np.arange(len(frames)) * step / fps
        diff = np.array([0.0] + [np.abs(frames[k] - frames[k - 1]).mean() for k in range(1, len(frames))])
        hist = [cv2.calcHist([f.astype(np.uint8)], [0], None, [32], [0, 256]) for f in frames]
        cut = np.array([0.0] + [1 - cv2.compareHist(hist[k - 1], hist[k], cv2.HISTCMP_CORREL) for k in range(1, len(frames))])
        med = float(np.median(diff[1:])) or 1e-6
        for lab, a, b in ps:
            m = (t >= a - 0.3) & (t <= a + 0.5); c = (t >= a - 0.5) & (t <= a + 0.5)
            out[f"{st}|{lab}|{a:.3f}"] = {"spike": float(diff[m].mean() / med) if m.any() else None, "cut": float(cut[c].max()) if c.any() else None}
    return out


def main():
    gold = S.load_gold([GOLD]); pics = base_pics()
    F = json.loads(FEAT.read_text()) if FEAT.exists() else features(pics)
    FEAT.write_text(json.dumps(F))
    allc = list(pics); grid = [None]
    for q in ("spike", "cut"):
        vals = np.array([v[q] for v in F.values() if v[q] is not None])
        grid += [(q, float(x)) for x in np.unique(np.round(np.quantile(vals, np.linspace(0.5, 0.98, 13)), 3))]
    drop = lambda st, p, g: (lambda v: v is not None and v > g[1])(F.get(f"{st}|{p[0]}|{p[1]:.3f}", {}).get(g[0]))
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], pics[st] if g is None else [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    r0 = [row(st, None) for st in allc]
    L = ["# Video motion spike / camera cut at the picture start, all 158 clips", "", "| feature | bar | hits | wrong | cost |", "|---|---|---|---|---|",
         f"| v1.4 | - | {hits(r0)} | {wr(r0)} | {cost(r0):.3f} |"]
    for g in grid[1:]:
        rr = [row(st, g) for st in allc]; L.append(f"| {g[0]} | {g[1]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    for mode in ("cost", "no hit lost"):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            if mode == "cost":
                best = min(grid, key=lambda g: cost([row(st, g) for st in tr]))
            else:
                h0 = hits([row(st, None) for st in tr])
                best = min([g for g in grid if hits([row(st, g) for st in tr]) >= h0], key=lambda g: wr([row(st, g) for st in tr]))
            ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        L += ["", f"CV by {mode}: choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}"]
    (HERE / "video_motion.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
