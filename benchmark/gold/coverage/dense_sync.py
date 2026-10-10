"""Video idea 4 (Adam 10 Oct: "all frames if it's a short sound"; Fable): dense frames at the sound's start. For every
v1.4 picture, 12 frames at 8 per second from 0.5 s before to 1.0 s after the picture start (the start is frame 5),
Qwen3.8-27B (thinking off), bias-cancelled yes/no margins (reason.py _yes_no_margin, question minus negated twin):
  sync   does something visible in these frames make the {l} sound at the moment it starts (frame 5)?   (drop when high)
  change does anything on screen change suddenly at frame 5, in a way that would make a {l} sound?      (drop when high)
Clip-grouped 5-fold CV (seed 0) on all 158 clips, by onset cost and by "no hit lost"; also restricted to short
pictures (<= 2.5 s).
    GPU: python benchmark/gold/coverage/dense_sync.py --score   (dense_sync.json)
    CPU: python benchmark/gold/coverage/dense_sync.py           -> dense_sync.md"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics, TG

HERE = Path(__file__).resolve().parent
FEAT = HERE / "dense_sync.json"
Q = {
    "sync": ("These 12 frames are 1/8 s apart; a sound of {l} starts at frame 5. Does something visible in these frames make "
             "this {l} sound at that moment? Answer yes or no.",
             "These 12 frames are 1/8 s apart; a sound of {l} starts at frame 5. Is the {l} sound made by something NOT visible "
             "in these frames? Answer yes or no."),
    "change": ("These 12 frames are 1/8 s apart. Does something on screen change suddenly around frame 5 in a way that would "
               "make a {l} sound? Answer yes or no.",
               "These 12 frames are 1/8 s apart. Is it true that nothing on screen around frame 5 could be making a {l} sound? "
               "Answer yes or no."),
}


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
            ims = []
            for i in range(12):
                f = int(max(0, min(n - 1, (a - 0.5 + i / 8) * fps))); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
                if ok:
                    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); im.thumbnail((448, 448)); ims.append(im)
            l = lab.split(",")[0].split("(")[0].strip().lower()
            out[k] = {q: (R._yes_no_margin(mdl, proc, p.format(l=l), ims) - R._yes_no_margin(mdl, proc, t.format(l=l), ims)) if ims else None
                      for q, (p, t) in Q.items()}
            out[k]["len"] = b - a
            print(k, out[k], flush=True)
            FEAT.write_text(json.dumps(out))


def main():
    gold = S.load_gold([GOLD]); pics = base_pics()
    if "--score" in sys.argv:
        score(pics); return
    F = json.loads(FEAT.read_text()); allc = list(pics); grid = [None]
    for scope in ("all", "short"):
        for q in Q:
            vals = np.array([v[q] for v in F.values() if v[q] is not None])
            grid += [(scope, q, float(x)) for x in np.unique(np.round(np.quantile(vals, np.linspace(0.5, 0.98, 13)), 3))]

    def drop(st, p, g):
        v = F.get(f"{st}|{p[0]}|{p[1]:.3f}")
        if v is None or v[g[1]] is None or (g[0] == "short" and v["len"] > 2.5):
            return False
        return v[g[1]] > g[2]
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], pics[st] if g is None else [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    r0 = [row(st, None) for st in allc]
    L = ["# Dense frames at the sound's start (Qwen3.8-27B), all 158 clips", "", f"pictures scored: {len(F)}", "",
         "| scope | question | bar | hits | wrong | cost |", "|---|---|---|---|---|---|", f"| v1.4 | - | - | {hits(r0)} | {wr(r0)} | {cost(r0):.3f} |"]
    for g in grid[1:]:
        rr = [row(st, g) for st in allc]; L.append(f"| {g[0]} | {g[1]} | {g[2]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
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
    (HERE / "dense_sync.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
