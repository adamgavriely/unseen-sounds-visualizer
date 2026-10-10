"""Screen reasoning to drop wrongs without losing hits (Adam 10 Oct: "use screen instead of sound; reason with the screen:
does the sound even fit there, is it seen on screen"). For every v1.4 picture, the pipeline's VLM (Qwen3.8-27B, thinking
off) gets 6 frames from 1 s before the picture start to 1 s after min(end, start + 4 s) and four yes/no questions, each
as a bias-cancelled margin (logit yes - logit no for the question minus the same for its negated twin; reason.py
_yes_no_margin, Round 60L):
  fit   could the sound plausibly be heard in this scene          (keep when high)
  off   is the source of this sound most likely off screen         (keep when high)
  seen  is the thing making the sound visible in the frames        (keep when low)
  prod  is a source visibly producing the sound right now          (keep when low)
A picture is dropped when its margin is on the wrong side of a bar t (t from the margins' own quantiles). Choices by
clip-grouped 5-fold CV (seed 0) on all 158 clips, two ways: by onset cost (w=2), and "no hit lost": the most wrongs
removed among settings that lose no hit on the training clips.
    GPU:  python benchmark/gold/coverage/screen_reason.py --score   (writes screen_margins.json)
    CPU:  python benchmark/gold/coverage/screen_reason.py           -> screen_reason.md"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
TG = Path.home() / "MscProj_tg"
Q = {
    "fit": ("Could the sound of {l} plausibly be heard in this scene? Answer yes or no.",
            "Could the sound of {l} NOT plausibly be heard in this scene? Answer yes or no.", +1),
    "off": ("A sound of {l} is heard at this moment of the video. Is the thing making it most likely OFF screen, not shown in these frames? Answer yes or no.",
            "A sound of {l} is heard at this moment of the video. Is the thing making it most likely ON screen, shown in these frames? Answer yes or no.", +1),
    "seen": ("Is the thing that makes the sound of {l} visible in these frames? Answer yes or no.",
             "Is the thing that makes the sound of {l} NOT visible in any of these frames? Answer yes or no.", -1),
    "prod": ("Is a {l} source visibly PRODUCING this sound right now (for example a beak open, a bell swinging, a vehicle moving)? Answer yes or no.",
             "Is it true that NO {l} source is visibly producing this sound right now? Answer yes or no.", -1),
}
MARG = HERE / "screen_margins.json"


def base_pics():
    config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    config.ONSET_CURVES = str(IN / "detector_curves.json")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    return {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in stems("dev") + stems("test")}


def score(pics):
    from src.stage5_cross_modal_analysis import reason as R
    from src.stage2_video_understanding import _sample_frames_at
    config.VLM_MODEL = "Qwen/Qwen3.8-27B"; config.VLM_THINKING = False
    mdl, proc = R._load(config.VLM_MODEL, "cuda")
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text())
    out = json.loads(MARG.read_text()) if MARG.exists() else {}
    for st, ps in pics.items():
        for lab, a, b in ps:
            k = f"{st}|{lab}|{a:.3f}"
            if k in out:
                continue
            lo, hi = a - 1.0, min(b, a + 4.0) + 1.0
            win = _sample_frames_at(TG / vids[st], [lo + (hi - lo) * i / 5 for i in range(6)])
            l = lab.split(",")[0].split("(")[0].strip().lower()
            out[k] = {q: (R._yes_no_margin(mdl, proc, p.format(l=l), win) - R._yes_no_margin(mdl, proc, t.format(l=l), win)) if win else None
                      for q, (p, t, _s) in Q.items()}
            print(k, out[k], flush=True)
            MARG.write_text(json.dumps(out))
    return out


def main():
    gold = S.load_gold([GOLD]); pics = base_pics()
    if "--score" in sys.argv:
        score(pics); return
    M = json.loads(MARG.read_text())
    allc = list(pics)
    grid = [None]
    for q, (_p, _t, sgn) in Q.items():
        vals = np.array([v[q] for v in M.values() if v[q] is not None])
        for t in np.unique(np.round(np.quantile(vals, np.linspace(0.02, 0.5, 13) if sgn > 0 else np.linspace(0.5, 0.98, 13)), 3)):
            grid.append((q, float(t)))

    def drop(st, p, g):
        q, t = g; v = M.get(f"{st}|{p[0]}|{p[1]:.3f}", {}).get(q)
        return v is not None and (v < t if Q[q][2] > 0 else v > t)
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], pics[st] if g is None else [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    L = ["# Screen reasoning (Qwen3.8-27B yes/no margins) to drop v1.4 pictures, all 158 clips", "",
         f"pictures scored: {len(M)}", "", "Best bar per question on all clips (for reading only; the CV rows below are the result):", "",
         "| question | bar | hits | wrong | cost |", "|---|---|---|---|---|"]
    r0 = [row(st, None) for st in allc]
    L.append(f"| v1.4 | - | {hits(r0)} | {wr(r0)} | {cost(r0):.3f} |")
    for q in Q:
        for g in [g for g in grid[1:] if g[0] == q]:
            rr = [row(st, g) for st in allc]
            L.append(f"| {q} | {g[1]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    for mode in ("cost", "no hit lost"):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            if mode == "cost":
                best = min(grid, key=lambda g: cost([row(st, g) for st in tr]))
            else:
                h0 = hits([row(st, None) for st in tr])
                ok = [g for g in grid if hits([row(st, g) for st in tr]) >= h0]
                best = min(ok, key=lambda g: wr([row(st, g) for st in tr]))
            ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        L += ["", f"CV by {mode}: choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}"]
    (HERE / "screen_reason.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
