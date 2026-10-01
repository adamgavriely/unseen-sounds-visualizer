"""Round 44 AVNAME (docs/prereg_round13_detector_push.md "Round 44 AVNAME"): every placed SHIP8 picture on merged DEV is re-named
by Qwen3-Omni from BOTH the audio cut and the video frames of its window (picture start - 0.5 ... + 1.5 s), one fixed question; the
answer is mapped to a depictable family with the shipped matcher (listener_variants.match_names, whole-word, no cosine). Variants:
(a) relabel to F when F != the picture's family; (b) drop when the answer names nothing / no family; (a)+(b). Picture times, the
gate and src/ are untouched. Nothing in src/ or config.py is edited.

    TG_ARMS=SHIP8 python benchmark/gold/avname_screen.py ask      # GPU msproj: Qwen3-Omni, gold never read -> avname/answers.json
    TG_ARMS=SHIP8 python benchmark/gold/avname_screen.py score    # CPU: map + rescore -> avname_screen.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import btp_screen as B
from benchmark.gold import expect_screen as E
from benchmark.gold import score_per_sound as S
from benchmark.gold.cross_group import classify
from benchmark.listener_round import MODEL
from src.labels import canonical, label_names

DIR = _ROOT / "benchmark" / "gold" / "avname"
ANS = DIR / "answers.json"
OUT = _ROOT / "benchmark" / "gold" / "avname_screen.json"
SR = 16000
PRE, POST = 0.5, 1.5                                   # window = picture start - PRE ... + POST (2.0 s, onset at 0.5 s)
N_FRAMES, FPS, LONG, MAX_NEW = 8, 4.0, 448, 16
Q = ("A sound starts at the middle of this clip. Using both what you hear and what you see, what most likely makes this sound? "
     "Answer with one short sound name.")
NONE_RE = re.compile(r"\b(nothing|silence|silent|none|no sound|quiet)\b")


def pictures():
    """[(part, clip, label, start)] of every placed SHIP8 picture (the gold loaded by btp_screen.parts is NOT returned)"""
    return [(pt, st, lab, float(a)) for pt, st, _g, pics in E.parts() for lab, a, _b, _r in pics]


def key(pt, st, lab, a):
    return f"{pt}|{st}|{lab}|{a:.3f}"


def norm(text: str):
    p = text.strip().splitlines()[0] if text.strip() else ""
    p = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", p).strip().strip(".!\"'`*").strip().lower()
    p = re.sub(r"^(a|an|the|some)\s+", "", p)
    p = re.sub(r"\s+(sounds?|noises?)$", "", p)
    return p.strip()


# ----------------------------------------------------------------------------- ask (GPU)
def cmd_ask():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from src.stage2_video_understanding import _sample_frames_at
    items = pictures()                                  # the gold btp_screen.parts loads is dropped here, never read
    DIR.mkdir(parents=True, exist_ok=True)
    ans = json.loads(ANS.read_text(encoding="utf-8")) if ANS.exists() else {}
    todo = [x for x in items if key(*x) not in ans]
    print(f"{len(items)} placed pictures, {len(todo)} to ask", flush=True)
    if not todo:
        return
    for pt, st, *_ in todo:
        assert (E.WAV[pt] / f"{st}.wav").exists(), (pt, st)
        assert Path(E.clip_file(pt, st)).exists(), (pt, st)
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    print(f"[ask] model loaded in {time.time() - t0:.0f} s", flush=True)
    conv = [{"role": "user", "content": [{"type": "video", "video": "x"}, {"type": "audio", "audio": "x"}, {"type": "text", "text": Q}]}]
    text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
    wcache = {}

    def wav(pt, st):
        k = (pt, st)
        if k not in wcache:
            w, sr = sf.read(str(E.WAV[pt] / f"{st}.wav"), dtype="float32")
            assert sr == SR and w.ndim == 1, (st, sr, w.shape)
            wcache.clear(); wcache[k] = w
        return wcache[k]

    def cut(w, a):
        """exactly 2.0 s, onset at 0.5 s, zero-padded at the clip edges"""
        n = int(round((PRE + POST) * SR)); seg = np.zeros(n, dtype=np.float32)
        i = int(round((a - PRE) * SR))
        lo, hi = max(i, 0), min(i + n, len(w))
        if hi > lo:
            seg[lo - i:hi - i] = w[lo:hi]
        return seg

    def resize(im):
        s = LONG / max(im.size)
        return im.resize((max(1, round(im.size[0] * s)), max(1, round(im.size[1] * s)))) if s < 1 else im

    t1 = time.time()
    for n, (pt, st, lab, a) in enumerate(todo, 1):
        seg = cut(wav(pt, st), a)
        times = [a - PRE + (i + 0.5) * (PRE + POST) / N_FRAMES for i in range(N_FRAMES)]
        frames = [resize(im) for im in _sample_frames_at(Path(E.clip_file(pt, st)), times)]
        assert frames, (pt, st, a)
        meta = [{"fps": FPS, "duration": len(frames) / FPS, "total_num_frames": len(frames)}]
        inp = proc(text=text, audio=[seg], videos=[frames], video_metadata=meta, return_tensors="pt", padding=True,
                   use_audio_in_video=False, cap_pixels_per_frame=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False)
        raw = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        ans[key(pt, st, lab, a)] = {"part": pt, "clip": st, "label": lab, "family": canonical(lab), "start": round(a, 3),
                                    "n_frames": len(frames), "frame_size": list(frames[0].size), "tokens": int(inp["input_ids"].shape[1]),
                                    "raw": raw, "answer": norm(raw)}
        if n <= 3 or n % 20 == 0:
            print(f"[ask] {n}/{len(todo)} {pt} {st} {lab} @{a:.2f}: {raw!r} ({time.time() - t1:.0f} s)", flush=True)
            ANS.write_text(json.dumps(ans, indent=1), encoding="utf-8")
    ANS.write_text(json.dumps(ans, indent=1), encoding="utf-8")
    print(f"[ask] done {len(todo)} in {time.time() - t1:.0f} s -> {ANS}", flush=True)


# ----------------------------------------------------------------------------- score (CPU)
def build_matcher():
    from benchmark.gold.listener_variants import Onto, match_names
    O = Onto()
    tab = []
    for fam in E.FAMILIES:
        own = {n for n in label_names(fam) if len(n) >= 3}
        for nm in match_names(O, fam):
            tab.append((fam, nm, nm in own, re.compile(r"\b" + re.escape(nm) + r"(?:s|es)?\b")))
    return tab


def map_answer(tab, answer: str):
    """-> (F | "NONE" | None, n_families_matched)"""
    if not answer:
        return None, 0
    if NONE_RE.search(answer):
        return "NONE", 0
    hits = [(fam, nm, own) for fam, nm, own, rx in tab if rx.search(answer)]
    if not hits:
        return None, 0
    fams = sorted({f for f, _n, _o in hits})
    best = {}
    for fam, nm, own in hits:
        cur = best.get(fam)
        cand = (int(own), len(nm))
        if cur is None or cand > cur:
            best[fam] = cand
    f = sorted(fams, key=lambda x: (-best[x][0], -best[x][1], x))[0]
    return f, len(fams)


def matched_needed(gold, pics):
    """set of needed, scored gold indices matched by the pictures (ledger_ship8.items, verbatim matching)"""
    gold = sorted(gold, key=lambda g: g["start"]); pics = sorted(pics, key=lambda p: p[1])
    taken, matched = [False] * len(gold), set()
    scored = lambda g: S.OLD_RULE or g["importance"] >= S.MIN_IMPORTANCE
    for lab, a, b in pics:
        cands = [i for i, g in enumerate(gold) if S.same_family(lab, g["label"]) and S.in_window(a, g["start"], S.EARLY, S.LATE)]
        if not cands:
            continue
        free = [i for i in cands if not taken[i]]
        if not free:
            continue
        i = min(free, key=lambda i: abs(a - gold[i]["start"]))
        covered = [i] + [j for j in cands if not taken[j] and S.same_family(gold[j]["label"], gold[i]["label"])]
        for j in covered:
            taken[j] = True
            if gold[j]["needed"] and scored(gold[j]):
                matched.add(j)
    return matched


def cmd_score():
    ans = json.loads(ANS.read_text(encoding="utf-8"))
    tab = build_matcher()
    P = E.parts()
    base = {"dev": [], "dev2": []}
    for pt, st, g, pics in P:
        base[pt].append(S.score_clip(g, [p[:3] for p in pics]))
    Bm = {"merged": B.summ(base["dev"] + base["dev2"]), "dev": B.summ(base["dev"]), "dev2": B.summ(base["dev2"])}
    print(f"BASE SHIP8: merged {B.fmt(Bm['merged'])} | DEV {B.fmt(Bm['dev'])} | DEV2 {B.fmt(Bm['dev2'])}")
    assert (Bm["merged"]["hits"], Bm["merged"]["wrong"]) == (E.BASE["hits"], E.BASE["wrong"]), Bm["merged"]
    assert abs(Bm["merged"]["cost"] - E.BASE["cost"]) < 0.001, Bm["merged"]
    base_w1 = S.viewer_cost(base["dev"] + base["dev2"], 1.0)
    # map every answer once
    mapped, amb = {}, 0
    for k, x in ans.items():
        f, nf = map_answer(tab, x["answer"])
        amb += nf > 1
        mapped[k] = {"F": f, "n_fam": nf}
    n_pic = sum(len(p[3]) for p in P)
    assert all(key(pt, st, lab, a) in ans for pt, st, _g, pics in P for lab, a, _b, _r in pics), "answers missing"
    tally = {"pictures": n_pic, "same_family": 0, "other_family": 0, "NONE": 0, "unmapped": 0, "ambiguous": amb}
    for pt, st, _g, pics in P:
        for lab, a, _b, _r in pics:
            f = mapped[key(pt, st, lab, a)]["F"]
            tally["NONE" if f == "NONE" else "unmapped" if f is None else "same_family" if f == canonical(lab) else "other_family"] += 1
    print(f"answers: {tally}")
    res = {"base": Bm, "base_cost_w1": round(base_w1, 4), "answers": tally, "variants": {}}
    for var in ("a", "b", "ab"):
        rows = {"dev": [], "dev2": []}; changed = []; lost = []
        for pt, st, g, pics in P:
            old = [p[:3] for p in pics]; new = []
            cl0 = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, old)}
            here = []
            for lab, a, b, _r in pics:
                f = mapped[key(pt, st, lab, a)]["F"]
                raw = ans[key(pt, st, lab, a)]["answer"]
                rec = {"var": var, "part": pt, "clip": st, "old": lab, "start": round(a, 2), "a3": round(a, 3), "answer": raw,
                       "before": cl0[(lab, round(a, 3))]}
                if "b" in var and (f is None or f == "NONE"):
                    here.append({**rec, "new": None}); continue
                if "a" in var and f not in (None, "NONE") and f != canonical(lab):
                    new.append((f, a, b)); here.append({**rec, "new": f}); continue
                new.append((lab, a, b))
            cl1 = {(l, round(a, 3)): k for l, a, b, k, _ in classify(g, new)}
            for c in here:
                c["after"] = "dropped" if c["new"] is None else cl1[(c["new"], c["a3"])]
            changed += here
            m0, m1 = matched_needed(g, old), matched_needed(g, new)
            if m0 - m1:
                gs = sorted(g, key=lambda x: x["start"])
                lost.append({"part": pt, "clip": st, "sounds": [f"{gs[j]['label']} @{gs[j]['start']}" for j in sorted(m0 - m1)]})
            rows[pt].append(S.score_clip(g, new))
        X = {"merged": B.summ(rows["dev"] + rows["dev2"]), "dev": B.summ(rows["dev"]), "dev2": B.summ(rows["dev2"])}
        w1 = S.viewer_cost(rows["dev"] + rows["dev2"], 1.0)
        main = X["merged"]["hits"] >= E.BASE["hits"] and not lost and X["merged"]["cost"] < E.BASE["cost"]
        more = (X["merged"]["hits"] > E.BASE["hits"] and all(X[p][k] <= Bm[p][k] for p in ("dev", "dev2") for k in ("cross", "phantom"))
                and w1 < base_w1)
        print(f"\n({var}) merged {B.fmt(X['merged'])} w1 {w1:.3f} (base {base_w1:.3f}) | DEV {B.fmt(X['dev'])} | DEV2 {B.fmt(X['dev2'])}"
              f"  changed {len(changed)}  hits lost {lost} -> main {'GO' if main else 'FAIL'}, more-hits {'GO' if more else 'FAIL'}")
        for c in changed:
            print(f"   {c['part']:4s} {c['clip']} {c['old']} @{c['start']} answer {c['answer']!r} -> {c['new'] or 'DROP'}: {c['before']} -> {c['after']}")
        res["variants"][var] = {"rows": X, "cost_w1": round(w1, 4), "changed": changed, "hits_lost": lost, "main_rule": main, "more_hits": more}
    res["mapped"] = {k: {**ans[k], **mapped[k]} for k in ans}
    OUT.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    print(OUT)


if __name__ == "__main__":
    {"ask": cmd_ask, "score": cmd_score}[sys.argv[1]]()
