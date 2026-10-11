"""PREREG_inventory_llm.md: clip sound-inventory LLM (J-B1) as a wrong-dropper on v1.7.
    GPU (cluster, ~/wt_v17, env msproj):  python benchmark/gold/coverage/inventory_llm.py --score [i/n]
        -> inventory_captions.json (per clip, both rates), inventory_scores_<i>of<n>.json (resumes)
    CPU:                                  python benchmark/gold/coverage/inventory_llm.py   -> inventory_llm.md"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD, IN, stems
from benchmark.gold.coverage.screen_reason import base_pics, TG

HERE = Path(__file__).resolve().parent
CAP_Q = ("Describe what is visible in this frame in one sentence: the place, the people, animals, vehicles and objects, "
         "and what they are doing.")
W = {"W1": ("Based on the video description and the sound detections, is a {f} sound really present in this clip around "
            "{t:.0f} s? Answer yes or no.", +1),
     "W2": ("Given what is seen in the video and what the detectors heard, would a {f} sound plausibly be heard in this clip "
            "at about {t:.0f} s? Answer yes or no.", +1),
     "W3": ("Is it likely that the detector's '{f}' at {t:.0f} s is a mistake, given the video description? Answer yes or no.", -1)}
RATES = (1.0, 0.5)


def short(l):
    return l.split(",")[0].split("(")[0].strip().lower()


def score(part):
    import cv2
    from PIL import Image
    from src.stage5_cross_modal_analysis import reason as R
    config.VLM_MODEL = "Qwen/Qwen3.8-27B"; config.VLM_THINKING = False
    mdl, proc = R._load(config.VLM_MODEL, "cuda")
    i, n = part
    pics = base_pics()
    st5 = json.loads((IN / "stage5_specs.json").read_text())
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text())
    clips = [st for k, st in enumerate(sorted(pics)) if k % n == i and pics[st]]
    capf, scf = HERE / f"inventory_captions_{i}of{n}.json", HERE / f"inventory_scores_{i}of{n}.json"
    caps = json.loads(capf.read_text()) if capf.exists() else {}
    out = json.loads(scf.read_text()) if scf.exists() else {}
    for st in clips:
        if st not in caps:
            cap = cv2.VideoCapture(str(TG / vids[st])); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
            nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0); dur = nfr / fps if nfr else 10.0
            c = {}
            for t in np.arange(0.5, dur, 1.0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, fr = cap.read()
                if ok:
                    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); im.thumbnail((640, 640))
                    c[f"{t:.1f}"] = R._ask(mdl, proc, CAP_Q, images=[im], max_new=48)
            caps[st] = c; capf.write_text(json.dumps(caps))
        timeline = "\n".join(f"{float(s['start']):.1f}-{float(s['end']):.1f} s: {s['event_label']} ({float(s.get('confidence', 0)):.2f})"
                             for s in sorted(st5[st]["B"], key=lambda s: float(s["start"])))
        for lab, a, b in pics[st]:
            k = f"{st}|{lab}|{a:.3f}"
            if k in out:
                continue
            r = {}
            for rate in RATES:
                ts = sorted(caps[st], key=float)
                ts = ts if rate == 1.0 else ts[::2]
                desc = "\n".join(f"t={float(t):.0f}s: {caps[st][t]}" for t in ts)
                ctx = f"Video description (one line per frame):\n{desc}\n\nSound detections:\n{timeline}\n\n"
                for w, (q, sgn) in W.items():
                    r[f"{w}_r{rate}"] = sgn * R._yes_no_margin(mdl, proc, ctx + q.format(f=short(lab), t=a))
            out[k] = r; scf.write_text(json.dumps(out))
            print(k, {x: round(v, 2) for x, v in r.items()}, flush=True)


def main():
    if "--score" in sys.argv:
        p = [x for x in sys.argv[1:] if "/" in x]
        i, n = (int(x) for x in p[0].split("/")) if p else (0, 1)
        score((i, n)); return
    gold = S.load_gold([GOLD]); pics = base_pics(); allc = stems("dev") + stems("test")
    F = {}
    for f in HERE.glob("inventory_scores_*of*.json"):
        F.update(json.loads(f.read_text()))
    keys = [f"{w}_r{r}" for w in W for r in RATES]
    sd = {k: np.std([v[k] for v in F.values()]) or 1.0 for k in keys}
    mu = {k: np.mean([v[k] for v in F.values()]) for k in keys}
    for v in F.values():
        v["mean"] = float(np.mean([(v[k] - mu[k]) / sd[k] for k in keys]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    base = {st: S.score_clip(gold[st], pics[st]) for st in allc}
    L = ["# Clip sound-inventory LLM (Qwen3.8-27B captions + detector timeline) as a wrong-dropper, all 158 clips", "",
         f"pictures scored: {len(F)} of {sum(len(v) for v in pics.values())}; v1.7: {hits(base.values())} hits, {wr(base.values())} wrong, cost {cost(list(base.values())):.3f}", "",
         "| score | out of fold hits | wrong | cost | CV choices | pass (>= 4 wrongs out, <= 1 hit lost) |", "|---|---|---|---|---|---|"]
    for k in keys + ["mean"]:
        vals = np.array([v[k] for v in F.values()])
        grid = [None] + [float(x) for x in np.unique(np.quantile(vals, np.linspace(0.02, 0.4, 12)))]
        drop = lambda st, p, t: t is not None and F.get(f"{st}|{p[0]}|{p[1]:.3f}", {}).get(k, 1e9) < t
        memo = {}
        row = lambda st, t: memo.setdefault((st, t), S.score_clip(gold[st], [p for p in pics[st] if not drop(st, p, t)]))
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for f in range(5):
            te = set(sh[f::5]); tr = [c for c in allc if c not in te]
            best = min(grid, key=lambda t: cost([row(st, t) for st in tr])); ch.append(None if best is None else round(best, 2))
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        ok = wr(base.values()) - wr(rr) >= 4 and hits(base.values()) - hits(rr) <= 1
        L.append(f"| {k} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} | {ch} | {'YES' if ok else 'no'} |")
    h = [v["mean"] for k, v in F.items() if k.split("|")[0] in pics and any(abs(p[1] - float(k.split('|')[2])) < 1e-3 and S.score_clip(gold[k.split('|')[0]], [p])["hit"] for p in pics[k.split('|')[0]])]
    L += ["", f"mean score: hits median {np.median(h):.2f}" if h else ""]
    (HERE / "inventory_llm.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
