"""Open-vocabulary object veto for on-screen sources (Adam 10 Oct: keep working). On top of v1.4, all 158 clips.
For each v1.4 picture: OWLv2 ("a photo of a {name}") on frames at 2 per second over [start, start + 1] s; score = best
box score. The picture is removed when score >= s; s in {0.2, 0.3, 0.4, 0.5} or off, chosen by clip-grouped 5-fold CV
on onset cost. Cluster GPU, run from ~/wt_slice (videos read from ~/MscProj_tg).

    python benchmark/gold/coverage/owl_veto.py   -> owl_veto.md (owl scores cached in owl_scores.json)
"""
import json, random, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import decide, pictures, stems, IN, GOLD

HERE = Path(__file__).resolve().parent
TG = Path.home() / "MscProj_tg"
GRID = [None, 0.2, 0.3, 0.4, 0.5]


def owl_scores(pics):
    import cv2, torch
    from PIL import Image
    from transformers import Owlv2Processor, Owlv2ForObjectDetection
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text())
    op = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    om = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").cuda().eval()
    out = {}
    for st, ps in pics.items():
        cap = cv2.VideoCapture(str(TG / vids[st])); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        for lab, a, b in ps:
            best = 0.0
            for t in (a, a + 0.5, a + 1.0):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps)); ok, fr = cap.read()
                if not ok:
                    continue
                img = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
                q = [" ".join(f"a photo of a {lab.split(',')[0].lower()}".split()[:8])]
                inp = op(text=q, images=img, return_tensors="pt").to("cuda")
                with torch.no_grad():
                    o = om(**inp)
                r = op.post_process_object_detection(o, threshold=0.0, target_sizes=torch.tensor([[max(img.size)] * 2], device="cuda"))[0]
                if len(r["scores"]):
                    best = max(best, float(r["scores"].max()))
            out[f"{st}|{lab}|{a:.3f}"] = best
    return out


def main():
    gold = S.load_gold([GOLD]); config.use_shipped(); config.MAX_AFTER_END = None
    config.GROUP_CACHE, config.DEPICT_CACHE = str(IN / "group_answers.json"), str(IN / "depict_answers.json")
    config.FLASH_CACHE, config.HOLD_CURVES, config.HOLD_FLEXSED_DIR = str(IN / "flashes.json"), str(IN / "flexsed_curves.json"), str(IN / "x")
    st5 = json.loads((IN / "stage5_specs.json").read_text()); dur = json.loads((IN / "durations.json").read_text())
    allc = stems("dev") + stems("test")
    pics = {st: pictures(decide(st5[st]["P"], st5[st]["B"], st5[st]["gate"]), float(dur[st] or 10), st) for st in allc}
    cf = HERE / "owl_scores.json"
    sc = json.loads(cf.read_text()) if cf.exists() else owl_scores(pics)
    cf.write_text(json.dumps(sc))
    apply = lambda st, s: pics[st] if s is None else [p for p in pics[st] if sc[f"{st}|{p[0]}|{p[1]:.3f}"] < s]
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], apply(st, g)))
    summ = lambda rr: (S.aggregate(rr)["hits"], sum(x["visible"] + x["cross"] + x["phantom"] for x in rr), S.viewer_cost(rr))
    rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [c for c in allc if c not in te]
        best = min(GRID, key=lambda g: S.viewer_cost([row(st, g) for st in tr])); ch.append(best)
        for st in te:
            oof[st] = row(st, best)
    L = ["# OWLv2 object veto, all 158 clips", "", f"v1.4: {summ([row(st, None) for st in allc])}",
         f"CV choices {ch}; out of fold: {summ([oof[st] for st in allc])}", "", "| s | hits | wrong | cost |", "|---|---|---|---|"]
    for g in GRID[1:]:
        h, w, c = summ([row(st, g) for st in allc]); L.append(f"| {g} | {h} | {w} | {c:.3f} |")
    (HERE / "owl_veto.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
