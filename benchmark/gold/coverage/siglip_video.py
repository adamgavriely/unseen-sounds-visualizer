"""Video ideas 1a + 2 + 5 (Adam 10 Oct: use video representations / similarity / crops; Fable). For every v1.4 picture,
6 frames from 1 s before the start to 1 s after min(end, start + 4) s, SigLIP so400m (google/siglip-so400m-patch14-384):
  rel   label vs rivals: candidates = the picture label + the other labels heard near it (raw detector candidates of other
        families starting within +-1 s, and the sibling options of its burst); text "a video scene in which you can hear
        {c}"; mean frame embedding; softmax over candidates (logit scale of the model) -> probability of the picture label
  obj   source on screen: max over frames of sigmoid(SigLIP logit) for "a photo of a {l}" on the whole frame
  tile  the same over 3 x 3 tiles of every frame (small sources: a mirror, a face)
Drop rules (one feature, one bar from its own quantiles): rel < t (wrong kind of sound), obj > t / tile > t (source seen).
Clip-grouped 5-fold CV (seed 0) on all 158 clips, by onset cost and by "no hit lost" (most wrongs removed among bars
that lose no training hit).
    GPU: python benchmark/gold/coverage/siglip_video.py --score  (siglip_video.json)
    CPU: python benchmark/gold/coverage/siglip_video.py          -> siglip_video.md"""
import json, random, sys
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics, TG

HERE = Path(__file__).resolve().parent
FEAT = HERE / "siglip_video.json"
SNAP = "google/siglip-so400m-patch14-384"


def short(l):
    return l.split(",")[0].split("(")[0].strip().lower()


def rivals(st, lab, a, dj, bursts):
    c = {lab}
    for k in dj.get(st, []):
        if abs(k["start"] - a) <= 1.0 and not S.same_family(k["label"], lab):
            c.add(k["label"])
    for bu in bursts.get(st, []):
        if S.same_family(lab, bu["family"]) and abs(bu["start"] - a) <= 1.0:
            c.update(o for o in bu.get("options", []) if o != "none of these")
    return sorted(c)


def score(pics):
    import cv2, torch
    from PIL import Image
    from transformers import AutoModel, AutoProcessor
    m = AutoModel.from_pretrained(SNAP).cuda().eval(); p = AutoProcessor.from_pretrained(SNAP)
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text())
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = {c["clip"]: c["cands"] for c in json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))["clips"]}
    bursts = {}
    for suf in ("dev", "test"):
        for b in json.loads((HERE / f"verify_items_{suf}.json").read_text())["bursts"]:
            bursts.setdefault(b["clip"], []).append(b)
    feat = lambda o: o if torch.is_tensor(o) else o.pooler_output

    def img_emb(ims):
        with torch.no_grad():
            return torch.nn.functional.normalize(feat(m.get_image_features(**p(images=ims, return_tensors="pt").to("cuda"))), dim=-1)

    def txt_emb(ts):
        with torch.no_grad():
            return torch.nn.functional.normalize(feat(m.get_text_features(**p(text=ts, padding="max_length", return_tensors="pt").to("cuda"))), dim=-1)
    scale, bias = m.logit_scale.exp().item(), m.logit_bias.item()
    out = {}
    for st, ps in pics.items():
        cap = cv2.VideoCapture(str(TG / vids[st])); fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        for lab, a, b in ps:
            lo, hi = a - 1.0, min(b, a + 4.0) + 1.0; ims = []
            for i in range(6):
                f = int(max(0, min(n - 1, (lo + (hi - lo) * i / 5) * fps))); cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
                if ok:
                    ims.append(Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)))
            if not ims:
                continue
            E = img_emb(ims)
            cands = rivals(st, lab, a, dj, bursts)
            T = txt_emb([f"a video scene in which you can hear {short(c)}" for c in cands])
            lg = scale * (torch.nn.functional.normalize(E.mean(0, keepdim=True), dim=-1) @ T.T)[0]
            rel = float(torch.softmax(lg, 0)[cands.index(lab)])
            o = txt_emb([f"a photo of a {short(lab)}"])
            obj = float(torch.sigmoid(scale * (E @ o.T) + bias).max())
            tiles = []
            for im in ims:
                W, H = im.size
                tiles += [im.crop((W * i // 3, H * j // 3, W * (i + 1) // 3, H * (j + 1) // 3)) for i in range(3) for j in range(3)]
            tile = float(torch.sigmoid(scale * (img_emb(tiles) @ o.T) + bias).max())
            out[f"{st}|{lab}|{a:.3f}"] = {"rel": rel, "n_cands": len(cands), "obj": obj, "tile": tile, "cands": cands}
            print(st, lab, out[f"{st}|{lab}|{a:.3f}"], flush=True)
    FEAT.write_text(json.dumps(out))


def main():
    gold = S.load_gold([GOLD]); pics = base_pics()
    if "--score" in sys.argv:
        score(pics); return
    F = json.loads(FEAT.read_text()); allc = list(pics); grid = [None]
    for q, lo_side in (("rel", True), ("obj", False), ("tile", False)):
        vals = np.array([v[q] for v in F.values() if not (q == "rel" and v["n_cands"] < 2)])
        qs = np.linspace(0.02, 0.5, 13) if lo_side else np.linspace(0.5, 0.98, 13)
        grid += [(q, float(x)) for x in np.unique(np.round(np.quantile(vals, qs), 4))]

    def drop(st, p, g):
        v = F.get(f"{st}|{p[0]}|{p[1]:.3f}")
        if v is None or (g[0] == "rel" and v["n_cands"] < 2):
            return False
        return v[g[0]] < g[1] if g[0] == "rel" else v[g[0]] > g[1]
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], pics[st] if g is None else [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    r0 = [row(st, None) for st in allc]
    L = ["# SigLIP video features to drop v1.4 pictures, all 158 clips", "", f"pictures scored: {len(F)}; with rivals: {sum(v['n_cands'] >= 2 for v in F.values())}", "",
         "| feature | bar | hits | wrong | cost |", "|---|---|---|---|---|", f"| v1.4 | - | {hits(r0)} | {wr(r0)} | {cost(r0):.3f} |"]
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
    (HERE / "siglip_video.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
