"""Four stricter listener questions, scored in ONE
Qwen3-Omni job on the same candidates as R13-3. Same model, yes/no score and audio cut as benchmark/gold/dev_listener.py unless
stated below.

Candidates (per split):
  P2  cached FlexSED 0.4-runs with peak >= 0.5 (dev_listener.json / test_listener.json, pool P2)
  PV  FlexSED >= 0.8 spans the PANNs clip veto removed (DEV: scored trace union & flexsed_raw - veto; TEST: the cache's PV);
      keyed by the vetoed span, asked on the P2 run that contains it (the R13-3 (b) lookup)
  P1  B0r stage-4 spans, depictable, conf < 0.6 (coordinator addendum: a verifier that DROPS weak pictures) -- V1 and V2 only
Variants (X = the item's family; the null control asks the item's cached null_family on the same cut):
  V1  multiple choice, X at A and at D, 5-letter softmax (max logit over "A"/" A" ...); accept p(X) > 0.5 and > 2 x max other
  V2  s_run = yes/no score on the run cut (live), s_ctrl = same question on the nearest same-length window not overlapping the
      run cut whose FlexSED(X) max < 0.2; accept s_run > 3 and s_run - s_ctrl > 2 (no window -> reject)
  V3  localisation on [run start - 3, run end + 3], greedy, 8 tokens; accept start - 0.5 <= t <= end + 0.5
  V4  open inventory on the run cut, greedy, 64 tokens (text shared by X and the null); a line matches X by name / AudioSet
      synonym / direct child or parent name (word match), else all-mpnet-base-v2 cosine > 0.6
No gold is read by pool or score (score_per_sound.load_gold raises); DEV classes are the dev cache's 'gold' field, used by
report only.

    python benchmark/gold/listener_variants.py pool     # CPU: both candidate files (no model), sizes + examples
    python benchmark/gold/listener_variants.py score    # GPU: one model load, both files (resumable)
    python benchmark/gold/listener_variants.py report   # CPU: DEV accept counts by gold class, the 8 R13-3 hits, nulls by half

(design record: release v1.2.0)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import dev_listener as L
from benchmark.listener_round import MODEL, QUESTION, VOCAB
from src.labels import canonical, label_names, is_descendant

GOLD = _ROOT / "benchmark" / "gold"
SPLITS = {
    "dev": {"cache": GOLD / "dev_listener.json", "out": GOLD / "dev_listener_v.json", "wav": C.WORK / "devcand" / "wav16",
            "trace": C.WORK / "protocol_proposed_dev_monocap_v31"},
    "test": {"cache": GOLD / "test_listener.json", "out": GOLD / "test_listener_v.json", "wav": C.WORK / "r13test" / "wav16"},
}
FLEX_DIR = C.FLEX_DIR
LO, P1_CONF, CTRL_BAR, TOL = 0.5, 0.6, 0.2, 0.02
V1_Q = ("Listen carefully. Which ONE of these is actually present in this recording? A) {} B) {} C) {} D) {} "
        "E) none of A-D. Answer with a single letter.")
V3_Q = ("If {} occurs in this recording, reply with the second it starts (for example 4.5). If it does not occur, "
        "reply none.")
V4_Q = "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
V3_NEW, V4_NEW, COS = 8, 64, 0.6
LETTERS = "ABCDE"
SR = 16000
VSET = set(VOCAB)


def _no_gold(*a, **k):
    raise RuntimeError("listener_variants: gold must not be read in pool/score")


# ============================================================================= ontology
class Onto:
    def __init__(self):
        o = json.loads((_ROOT / "src" / "audioset_ontology.json").read_text(encoding="utf-8"))
        idn = {e["id"]: e for e in o}
        self.bad = {e["name"] for e in o if set(e.get("restrictions", [])) & {"abstract", "blacklist"}}
        self.kids, self.pars = {}, {}
        for e in o:
            for c in e.get("child_ids", []):
                cn = idn[c]["name"] if c in idn else None
                if cn:
                    self.kids.setdefault(e["name"], []).append(cn)
                    self.pars.setdefault(cn, []).append(e["name"])
        self.names = [e["name"] for e in o]

    def depth(self, n):
        d, cur, seen = 0, n, set()
        while self.pars.get(cur) and cur not in seen:
            seen.add(cur); cur = self.pars[cur][0]; d += 1
        return d

    def top(self, n):
        cur, seen = n, set()
        while self.pars.get(cur) and cur not in seen:
            seen.add(cur); cur = self.pars[cur][0]
        return cur

    def node(self, fam):
        """ontology name for a family (hand-written families: the shallowest label with that canonical family)"""
        if fam in self.pars or fam in self.kids:
            return fam
        c = [n for n in self.names if canonical(n) == fam]
        return min(c, key=lambda n: (self.depth(n), n)) if c else None


def related(a, b):
    return canonical(a) == canonical(b) or S.same_family(a, b) or is_descendant(a, b) or is_descendant(b, a)


def seed_of(*parts):
    return int(hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest(), 16) % (2 ** 32)


def distractors(O, clip, fam, start):
    """s1, s2 = ontology siblings (vocab first), cousins via the grandparent when fewer than two; u = a vocab family from
    another top-level branch. Option text = canonical family name, lower case."""
    rng = random.Random(seed_of(clip, fam, f"{start:.2f}"))
    X = O.node(fam)
    picked, src = [], "siblings"

    def take(pool):
        pool = [c for c in dict.fromkeys(pool) if c not in O.bad and not related(c, fam)]
        seenf = {canonical(p) for p in picked}
        pool = [c for c in pool if canonical(c) not in seenf]
        v = [c for c in pool if canonical(c) in VSET]; o = [c for c in pool if canonical(c) not in VSET]
        rng.shuffle(v); rng.shuffle(o)
        for c in v + o:
            if len(picked) < 2 and canonical(c) not in {canonical(p) for p in picked}:
                picked.append(c)
    if X:
        take([c for p in O.pars.get(X, []) for c in O.kids.get(p, []) if c != X])
        if len(picked) < 2:
            src = "cousins"
            take([k for p in O.pars.get(X, []) for gp in O.pars.get(p, []) for u in O.kids.get(gp, []) if u != p
                  for k in O.kids.get(u, [])])
    if len(picked) < 2:
        src = "fallback"
        tx = O.top(X) if X else None
        take([v for v in VOCAB if O.node(v) and O.top(O.node(v)) == tx])
        if len(picked) < 2:
            take(list(VOCAB))
    tx = O.top(X) if X else None
    far = [v for v in VOCAB if O.node(v) and O.top(O.node(v)) != tx and not related(v, fam)
           and canonical(v) not in {canonical(p) for p in picked}]
    u = rng.choice(sorted(far))
    return [canonical(picked[0]).lower(), canonical(picked[1]).lower(), canonical(u).lower()], src


def match_names(O, fam):
    X = O.node(fam)
    ns = set(label_names(fam))
    if X:
        ns |= set(label_names(X))
        ns |= {w for c in O.kids.get(X, []) for w in label_names(c)}
        ns |= {w for p in O.pars.get(X, []) if O.pars.get(p) for w in label_names(p)}      # no top-level roots
    ns |= {w for n in O.names if canonical(n) == fam for w in label_names(n)}
    return sorted(n for n in ns if len(n) >= 3)


# ============================================================================= candidates
def vetoed(trace_path):
    tr = json.loads(trace_path.read_text(encoding="utf-8"))
    sig = lambda n: {(x["label"], float(x["start"]), float(x["end"])) for x in tr if x["step"] == n}
    return sorted((sig("union") & sig("flexsed_raw")) - sig("veto"))


def contain(p2, clip, fam, a, b):
    c = [x for x in p2 if x["clip"] == clip and x["family"] == fam and x["start"] <= a + TOL and x["end"] >= b - TOL]
    return max(c, key=lambda x: x["run_len"]) if c else None


def run_from_flex(clip, lab, a, b, dur):
    fw, ts, labs = C.load_fr(FLEX_DIR / f"{clip}.npz")
    j = labs.index(lab)
    rr, dt = L.runs(fw[:, j], ts, L.FLEX_BAR, L.FLEX_GAP)
    for i, k in rr:
        s, e = float(ts[i]), float(ts[k - 1] + dt)
        if s <= a + TOL and e >= b - TOL:
            it = {"start": s, "end": e, "run_len": e - s, "peak": float(fw[i:k, j].max())}
            if e - s < L.MIN_CUT:
                pk_t = float(ts[i + int(np.argmax(fw[i:k, j]))] + dt / 2)
                ca = min(max(0.0, pk_t - L.MIN_CUT / 2), max(0.0, dur - L.MIN_CUT))
                it["cut_start"], it["cut_end"] = ca, ca + L.MIN_CUT
            else:
                it["cut_start"], it["cut_end"] = s, e
            return it
    return None


def gold_free_null(clip, rng):
    fw, _t, labs = C.load_fr(FLEX_DIR / f"{clip}.npz")
    hot = [l for i, l in enumerate(labs) if float(fw[:, i].max()) >= 0.1]
    return rng.choice([v for v in VOCAB if not any(related(v, h) for h in hot)])


def build(split):
    import soundfile as sf
    S.load_gold = _no_gold
    cfg = SPLITS[split]
    d = json.loads(cfg["cache"].read_text(encoding="utf-8"))
    items = d["items"]
    clips = sorted({x["clip"] for x in items})
    p2 = [x for x in items if x["pool"] == "P2"]
    keep = ("clip", "pool", "family", "label", "start", "end", "cut_start", "cut_end", "run_len", "peak", "conf", "origin",
            "depictable", "null_family", "gold", "score", "null_score")
    out = []
    for x in items:
        if (x["pool"] == "P2" and x["peak"] >= LO) or (x["pool"] == "P1" and x["depictable"] and x["conf"] < P1_CONF):
            it = {k: x[k] for k in keep if k in x}
            it["cached_score"], it["cached_null_score"] = it.pop("score", None), it.pop("null_score", None)
            it["run_start"], it["run_end"] = x["start"], x["end"]
            it["variants"] = ["V1", "V2"] if x["pool"] == "P1" else ["V1", "V2", "V3", "V4"]
            out.append(it)
    # PV: vetoed >= 0.8 spans, asked on their containing P2 run
    rng = random.Random(0)
    if split == "dev":
        pv = [{"clip": st, "label": lab, "start": a, "end": b} for st in clips
              for lab, a, b in vetoed(cfg["trace"] / st / "onset_trace.json")]
    else:
        pv = [x for x in items if x["pool"] == "PV"]
    for v in pv:
        fam = canonical(v["label"])
        run = contain(p2, v["clip"], fam, v["start"], v["end"])
        it = {"clip": v["clip"], "pool": "PV", "family": fam, "label": v["label"], "start": float(v["start"]),
              "end": float(v["end"]), "depictable": L.depictable(v["label"]), "variants": ["V1", "V2", "V3", "V4"]}
        if run:
            it.update({"run_start": run["start"], "run_end": run["end"], "cut_start": run["cut_start"], "cut_end": run["cut_end"],
                       "run_len": run["run_len"], "peak": run["peak"], "run_from_cache": True,
                       "cached_score": run.get("score")})
            if "gold" in run:
                it["gold_run"] = run["gold"]
        else:
            dur = float(sf.info(str(cfg["wav"] / f"{v['clip']}.wav")).duration)
            r = run_from_flex(v["clip"], v["label"], v["start"], v["end"], dur)
            it.update({"run_start": r["start"], "run_end": r["end"], "cut_start": r["cut_start"], "cut_end": r["cut_end"],
                       "run_len": r["run_len"], "peak": r["peak"], "run_from_cache": False, "cached_score": None})
        it["null_family"] = v.get("null_family") or (run or {}).get("null_family") or gold_free_null(v["clip"], rng)
        out.append(it)
    # distractors and control windows
    O = Onto()
    flex = {}
    for it in out:
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            ds, src = distractors(O, it["clip"], fam, it["start"])
            it[f"v1_{tag}_options"], it[f"v1_{tag}_source"] = ds, src
        if it["clip"] not in flex:
            flex[it["clip"]] = C.load_fr(FLEX_DIR / f"{it['clip']}.npz")
        dur = float(sf.info(str(cfg["wav"] / f"{it['clip']}.wav")).duration)
        a = max(0.0, it["cut_start"] - 1.0); b = min(dur, it["cut_end"] + 1.0)
        it["run_audio"] = [a, max(b, a + 1.0)]
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            it[f"v2_{tag}_ctrl"], it[f"v2_{tag}_flex_col"] = control(flex[it["clip"]], fam, it, dur)
        it["v3_cut"] = [max(0.0, it["run_start"] - 3.0), min(dur, it["run_end"] + 3.0)]
    meta = {"amendment": "A (docs/prereg_round13_detector_push.md)", "split": split, "model": MODEL, "source_cache": str(cfg["cache"]),
            "V1": V1_Q, "V2": QUESTION + f" on the run cut minus the same on the control cut (FlexSED(X) max < {CTRL_BAR}, same "
            "length, nearest, not overlapping the run cut)", "V3": V3_Q, "V4": V4_Q,
            "rules": {"V1": "p(X) > 0.5 and p(X) > 2 x max(other)", "V2": "s_run > 3 and s_run - s_ctrl > 2",
                      "V3": "run_start - 0.5 <= t <= run_end + 0.5", "V4": "some line matches X", "V12": "V1 and V2"},
            "pools": {"P2": f"cache P2 with peak >= {LO}", "PV": "PANNs-vetoed FlexSED >= 0.8 spans, asked on the containing P2 run",
                      "P1": f"cache P1 depictable with conf < {P1_CONF}, V1 and V2 only"},
            "clips": len(clips)}
    C.dump(cfg["out"], {"_meta": meta, "items": out})
    n = {p: sum(x["pool"] == p for x in out) for p in ("P2", "PV", "P1")}
    print(f"[pool] {split}: {len(out)} items {n}; PV not in cache {sum(1 for x in out if x.get('run_from_cache') is False)}; "
          f"V1 sources x {count(out, 'v1_x_source')} null {count(out, 'v1_null_source')}; control found x "
          f"{sum(x['v2_x_ctrl'] is not None for x in out)} null {sum(x['v2_null_ctrl'] is not None for x in out)}; "
          f"FlexSED column x {sum(x['v2_x_flex_col'] for x in out)}", flush=True)
    for x in out[:3] + [y for y in out if y["pool"] == "PV"][:2]:
        print(f"   {x['clip']} {x['pool']} {x['family']} {x['start']:.2f}-{x['end']:.2f} options {x['v1_x_options']} "
              f"({x['v1_x_source']}) ctrl {x['v2_x_ctrl']} null {x['null_family']} {x['v1_null_options']}", flush=True)


def count(out, k):
    c = {}
    for x in out:
        c[x[k]] = c.get(x[k], 0) + 1
    return c


def control(fr, fam, it, dur):
    """nearest window of the run audio's length, on the 0.04-s grid, inside the clip, not overlapping the run cut, whose
    FlexSED max over the family's columns is < 0.2 (no column: only the overlap condition)"""
    fw, ts, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == fam]
    Lw = it["run_audio"][1] - it["run_audio"][0]
    a0 = it["run_audio"][0]
    if Lw > dur:
        return None, bool(cols)
    s = fw[:, cols].max(axis=1) if cols else np.zeros(len(ts))
    badt = np.sort(ts[s >= CTRL_BAR])
    ws = np.arange(0.0, dur - Lw + 1e-9, 0.04)
    we = ws + Lw
    ok = (we <= it["cut_start"] + 1e-9) | (ws >= it["cut_end"] - 1e-9)
    ok &= (np.searchsorted(badt, we, "left") - np.searchsorted(badt, ws, "left")) == 0
    if not ok.any():
        return None, bool(cols)
    k = int(np.argmin(np.where(ok, np.abs(ws - a0), np.inf)))
    return [float(ws[k]), float(ws[k] + Lw)], bool(cols)


# ============================================================================= model
def score():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from sentence_transformers import SentenceTransformer
    S.load_gold = _no_gold
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    tok = proc.tokenizer
    first = lambda w: tok.encode(w, add_special_tokens=False)[0]
    yes_ids = sorted({first(w) for w in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({first(w) for w in ("no", "No", " no", " No")})
    let_ids = [sorted({first(c), first(" " + c)}) for c in LETTERS]
    all_let = {i for g in let_ids for i in g}
    print(f"[score] model + mpnet loaded in {time.time() - t0:.0f} s; yes {yes_ids} no {no_ids} letters {let_ids}", flush=True)
    O = Onto()

    def prep(w, q):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        return inp.to(model.thinker.device).to(torch.bfloat16) if hasattr(inp, "to") else inp

    def last(w, q):
        with torch.inference_mode():
            return model.thinker(**prep(w, q)).logits[0, -1].float()

    def yesno(w, fam):                                            # dev_listener / listener_round ask(), unchanged
        lg = last(w, QUESTION.format(fam.lower()))
        return float(lg[yes_ids].max() - lg[no_ids].max())

    def mc(w, opts):
        lg = last(w, V1_Q.format(*opts))
        v = torch.stack([lg[g].max() for g in let_ids])
        p = torch.softmax(v, 0).tolist()
        top = int(lg.argmax())
        return p, top, top in all_let

    def gen(w, q, n):
        inp = prep(w, q)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=n, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    for split, cfg in SPLITS.items():
        d = json.loads(cfg["out"].read_text(encoding="utf-8"))
        todo = sorted([x for x in d["items"] if "accept" not in x], key=lambda x: (x["clip"], x["cut_start"]))
        cache, v4cache = {}, {}
        t1 = time.time()
        print(f"[score] {split}: {len(todo)} items to do", flush=True)

        def wav(st):
            if st not in cache:
                w, sr = sf.read(str(cfg["wav"] / f"{st}.wav"), dtype="float32")
                assert sr == SR and w.ndim == 1, (st, sr, w.shape)
                cache.clear(); cache[st] = w
            return cache[st]

        def cut(w, a, b):
            i = int(a * SR); j = int(b * SR)
            return w[i:max(j, i + SR)]
        for n, it in enumerate(todo, 1):
            w = wav(it["clip"])
            seg = cut(w, *it["run_audio"])
            it["accept"], it["null_accept"] = {}, {}
            for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
                acc = it["accept"] if tag == "x" else it["null_accept"]
                # V1: X at A (X, s1, s2, u), X at D (s1, s2, u, X)
                s1, s2, u = it[f"v1_{tag}_options"]
                o1 = [fam.lower(), s1, s2, u]; o2 = [s1, s2, u, fam.lower()]
                p1, t1_, l1 = mc(seg, o1); p2, t2_, l2 = mc(seg, o2)
                opt = {"X": (p1[0] + p2[3]) / 2, "s1": (p1[1] + p2[0]) / 2, "s2": (p1[2] + p2[1]) / 2, "u": (p1[3] + p2[2]) / 2,
                       "none": (p1[4] + p2[4]) / 2}
                it[f"v1_{tag}"] = {"probs_XatA": p1, "probs_XatD": p2, "top1_letter": [l1, l2], "top1_id": [t1_, t2_], "p": opt}
                pX = opt["X"]; po = max(v for k, v in opt.items() if k != "X")
                acc["V1"] = bool(pX > 0.5 and pX > 2 * po)
                # V2
                s_run = yesno(seg, fam)
                ctrl = it[f"v2_{tag}_ctrl"]
                s_ctrl = yesno(cut(w, *ctrl), fam) if ctrl else None
                it[f"v2_{tag}"] = {"s_run": s_run, "s_ctrl": s_ctrl, "ctrl": ctrl}
                acc["V2"] = bool(ctrl is not None and s_run > 3 and s_run - s_ctrl > 2)
                acc["V12"] = acc["V1"] and acc["V2"]
                if "V3" not in it["variants"]:
                    continue
                # V3
                ca, cb = it["v3_cut"]
                txt = gen(cut(w, ca, cb), V3_Q.format(fam.lower()), V3_NEW)
                m = re.search(r"-?\d+(?:\.\d+)?", txt)
                t_rel = float(m.group(0)) if m else None
                t_clip = ca + t_rel if t_rel is not None else None
                it[f"v3_{tag}"] = {"text": txt, "t_rel": t_rel, "t_clip": t_clip}
                acc["V3"] = bool(t_clip is not None and it["run_start"] - 0.5 <= t_clip <= it["run_end"] + 0.5)
                # V4 (text independent of X; one generation per run cut)
                k = (it["clip"], round(it["run_audio"][0], 3), round(it["run_audio"][1], 3))
                if k not in v4cache:
                    v4cache[k] = gen(seg, V4_Q, V4_NEW)
                txt = v4cache[k]
                it["v4_text"] = txt
                lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in txt.splitlines()]
                lines = [ln for ln in lines if ln]
                names = match_names(O, fam)
                hit_word = [ln for ln in lines if any(re.search(r"\b" + re.escape(nm) + r"(?:s|es)?\b", ln.lower()) for nm in names)]
                cos = []
                if lines:
                    e = emb.encode(lines + [fam.lower()], normalize_embeddings=True, convert_to_numpy=True)
                    cos = [float(x) for x in e[:-1] @ e[-1]]
                hit_cos = [ln for ln, c in zip(lines, cos) if c > COS and ln not in hit_word]
                it[f"v4_{tag}"] = {"names": names, "matched_word": hit_word, "matched_cos": hit_cos, "cos": cos}
                acc["V4"] = bool(hit_word or hit_cos)
            if n == 10:
                print(f"[score] {split} 10 items in {time.time() - t1:.0f} s", flush=True)
            if n % 100 == 0:
                C.dump(cfg["out"], d)
                print(f"[score] {split} {n}/{len(todo)} ({time.time() - t1:.0f} s)", flush=True)
        d["_meta"].update({"yes_ids": yes_ids, "no_ids": no_ids, "letter_ids": let_ids, "seconds": time.time() - t1})
        C.dump(cfg["out"], d)
        print(f"[score] {split} done {len(todo)} in {time.time() - t1:.0f} s -> {cfg['out']}", flush=True)
        summary(d["items"], split)


def summary(items, split):
    its = [x for x in items if "accept" in x]
    for p in ("P2", "PV", "P1"):
        pp = [x for x in its if x["pool"] == p]
        if not pp:
            continue
        rules = [r for r in ("V1", "V2", "V3", "V4", "V12") if r in pp[0]["accept"]]
        print(f"[summary] {split} {p} n {len(pp)}: " + ", ".join(
            f"{r} {sum(x['accept'][r] for x in pp)} (null {sum(x['null_accept'][r] for x in pp)})" for r in rules), flush=True)
    top = [f for x in its for f in x["v1_x"]["top1_letter"]]
    print(f"[summary] {split} V1 first token is a letter: {np.mean(top):.1%}", flush=True)


# ============================================================================= DEV report (gold field of the dev cache)
MISSES = [("ambient_citywalk_nyc_1689", "Hammer", 13.7), ("ambient_nature_rainforest_7629", "Bird", 0.1),
          ("as_explosion_XJ8lc3I6", "Gunshot, gunfire", 0.0), ("as_explosion_XJ8lc3I6", "Walk, footsteps", 2.1),
          ("as_explosion_XJ8lc3I6", "Explosion", 2.8), ("as_explosion_XJ8lc3I6", "Explosion", 5.6),
          ("as_explosion_XJ8lc3I6", "Gasp", 6.7), ("b3_carnival_parade", "Whistle", 6.1)]


def report():
    d = json.loads(SPLITS["dev"]["out"].read_text(encoding="utf-8"))
    its = [x for x in d["items"] if "accept" in x]
    rules = ("V1", "V2", "V3", "V4", "V12")
    gold = S.load_gold([GOLD / "annotations" / "gold_AG.json"])       # DEV clips only (report)
    miss = []
    for st, lab, on in MISSES:
        g = [x for x in gold[st] if S.same_family(x["label"], lab) and abs(x["start"] - on) <= 0.06 and x["needed"]]
        assert len(g) >= 1, (st, lab, on)
        miss.append((st, g[0]["label"], g[0]["start"]))
    for x in its:
        if x["pool"] == "PV":                                         # the vetoed span's own class (as the cache's rule)
            x["gold"] = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
    res = {}
    for p in ("P2", "PV", "P1"):
        pp = [x for x in its if x["pool"] == p]
        for r in rules:
            if r not in pp[0]["accept"]:
                continue
            by = {k: [sum(1 for x in pp if x.get("gold") == k and x["accept"][r]), sum(1 for x in pp if x.get("gold") == k)]
                  for k in ("hit_needed", "none", "other_gold")}
            res[f"{p}|{r}"] = by
            print(f"[dev] {p} {r}: accepted/all hit_needed {by['hit_needed']}, none {by['none']}, other_gold {by['other_gold']}")
    cand = [x for x in its if x["pool"] in ("P2", "PV")]
    for r in rules:
        kept = []
        for st, lab, on in miss:
            ok = False
            for x in cand:
                if x["clip"] != st or not x["accept"][r] or not S.same_family(x["label"], lab):
                    continue
                eff = x["start"] if (x["pool"] == "PV" or x["run_len"] >= 0.5) else x["cut_start"]
                ok = ok or S.in_window(eff, on, S.EARLY, S.LATE)
            kept.append(ok)
        res[f"miss|{r}"] = kept
        print(f"[dev] {r} keeps {sum(kept)}/8 R13-3 hits: {[f'{m[1]} {m[2]:.1f}' for m, k in zip(miss, kept) if k]}")
    clips = sorted({x["clip"] for x in its})
    half = {c: "AB"[i % 2] for i, c in enumerate(clips)}
    for r in rules:
        s = []
        for h in "AB":
            hh = [x for x in cand if half[x["clip"]] == h]
            s.append(f"{h} {np.mean([x['null_accept'][r] for x in hh]):.1%} (n {len(hh)})")
        p1 = [x for x in its if x["pool"] == "P1" and r in x["null_accept"]]
        extra = f"; P1 null {np.mean([x['null_accept'][r] for x in p1]):.1%}" if p1 else ""
        print(f"[dev] {r} null accept-rate P2+PV by half: {', '.join(s)}{extra}")
    top = [f for x in its for f in x["v1_x"]["top1_letter"]]
    print(f"[dev] V1 first token is a letter: {np.mean(top):.1%}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pool", "score", "report"))
    a = ap.parse_args()
    if a.step == "pool":
        build("dev"); build("test")
    elif a.step == "score":
        score()
    else:
        report()


if __name__ == "__main__":
    main()
