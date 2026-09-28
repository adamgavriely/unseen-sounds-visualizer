"""Round 13 (docs/prereg_round13_detector_push.md): a cache of audio-LLM "listener" scores for candidate sound spans on the
49 DEV clips. Same model, question, scoring and audio cut as amendment 25 (benchmark/listener_round.py): Qwen3-Omni-30B-A3B-
Instruct, "Is the sound of {family} present in this recording? Answer yes or no.", score = logit(yes) - logit(no), audio =
the clip's 16-kHz mono from start - 1 s to end + 1 s (at least 1 s). DEV only.

Pools per clip:
  P1  every B0r|proposed stage-4 span (data/work/devcand/stage4.json, all origins, conf as stored)
  P2  every FlexSED run (one query, frames >= 0.4, gaps <= 0.24 s merged, any length) with no same-family P1 span overlapping;
      a run shorter than 1 s is cut for the listener as 1 s centred on its peak frame (cut_start/cut_end), start/end stay the run
  P3  every BEATs run (one label, windows >= 0.175) whose peak is < 0.35, with no same-family P1 span overlapping
Null control: the same audio, asked about a vocab family that is not the family of any gold sound of the clip (seed 0).
Gold class (analysis only): score_per_sound.score_clip on the span as a lone picture: hit -> hit_needed, phantom -> none,
anything else (visible / cross / dup / don't-care) -> other_gold.

    python benchmark/gold/dev_listener.py pool     # CPU: build the pool into benchmark/gold/dev_listener.json, print sizes
    python benchmark/gold/dev_listener.py score    # GPU: one model load, fills score / null_score / p_yes (resumable)
    python benchmark/gold/dev_listener.py report   # CPU: yes-rates per pool, AUROC hit_needed vs none
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import score_per_sound as S
from benchmark.listener_round import MODEL, QUESTION, VOCAB, auroc
from src.labels import canonical, is_salient_nonspeech

OUT = _ROOT / "benchmark" / "gold" / "dev_listener.json"
STAGE4_ARM = "B0r|proposed"
FLEX_BAR, FLEX_GAP = 0.4, 0.24
BEATS_LO, BEATS_HI = C.F["AED"], C.F["DISP"]          # 0.175, 0.35
MIN_CUT = 1.0
SR = 16000


def depictable(label):
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        return bool(is_salient_nonspeech(label))
    finally:
        config.LABEL_FILTER = old


def runs(col, ts, bar, gap_s):
    """contiguous frames >= bar, runs separated by <= gap_s merged; (i, j) frame ranges, end = times[j-1] + dt"""
    dt = float(ts[1] - ts[0])
    on = col >= bar
    out, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1; continue
        j = i
        while j < n and on[j]:
            j += 1
        if out and (i - out[-1][1]) * dt <= gap_s + 1e-6:
            out[-1] = (out[-1][0], j)
        else:
            out.append((i, j))
        i = j
    return out, dt


def covered(p1, fam, a, b):
    return any(canonical(r["label"]) == fam and min(b, r["end"]) - max(a, r["start"]) > 0 for r in p1)


def gold_class(g, label, a, b):
    r = S.score_clip(g, [(label, a, b)])
    return "hit_needed" if r["hit"] > 0 else ("none" if r["phantom"] > 0 else "other_gold")


def build_pool():
    gold, stems = C.dev_stems()
    assert len(stems) == 49, len(stems)
    s4 = json.loads(C.STAGE4.read_text(encoding="utf-8"))["arms"][STAGE4_ARM]
    rng = random.Random(0)
    items = []
    for st in stems:
        g = gold[st]
        dur = _dur(st)
        absent = [v for v in VOCAB if not any(S.same_family(v, x["label"]) or canonical(v) == canonical(x["label"]) for x in g)]
        new = []
        p1 = s4[st]
        for r in p1:
            new.append({"pool": "P1", "label": r["label"], "start": float(r["start"]), "end": float(r["end"]),
                        "conf": float(r["conf"]), "origin": r["origin"], "run_len": float(r["end"] - r["start"])})
        # P2: FlexSED runs
        fw, ts, labs = C.load_fr(C.FLEX_DIR / f"{st}.npz")
        for c, lab in enumerate(labs):
            fam = canonical(lab)
            rr, dt = runs(fw[:, c], ts, FLEX_BAR, FLEX_GAP)
            for i, j in rr:
                a, b = float(ts[i]), float(ts[j - 1] + dt)
                if covered(p1, fam, a, b):
                    continue
                k = i + int(np.argmax(fw[i:j, c]))
                pk_t = float(ts[k] + dt / 2)
                it = {"pool": "P2", "label": lab, "start": a, "end": b, "peak": float(fw[k, c]), "peak_t": pk_t, "run_len": b - a}
                if b - a < MIN_CUT:
                    ca = min(max(0.0, pk_t - MIN_CUT / 2), max(0.0, dur - MIN_CUT))
                    it["cut_start"], it["cut_end"] = ca, ca + MIN_CUT
                new.append(it)
        # P3: BEATs runs >= 0.175 with peak < 0.35
        fw, ts, labs = C.load_fr(C.BEATS_DIR / f"{st}.npz")
        seen = set()
        for c, lab in enumerate(labs):
            fam = canonical(lab)
            rr, dt = runs(fw[:, c], ts, BEATS_LO, 0.0)
            for i, j in rr:
                pk = float(fw[i:j, c].max())
                if pk >= BEATS_HI:
                    continue
                a, b = float(ts[i]), float(ts[j - 1] + dt)
                if covered(p1, fam, a, b) or (fam, round(a, 3), round(b, 3)) in seen:
                    continue
                seen.add((fam, round(a, 3), round(b, 3)))
                k = i + int(np.argmax(fw[i:j, c]))
                new.append({"pool": "P3", "label": lab, "start": a, "end": b, "peak": pk, "peak_t": float(ts[k] + dt / 2),
                            "run_len": b - a})
        for it in new:
            it["clip"] = st
            it["family"] = canonical(it["label"])
            it["depictable"] = depictable(it["label"])
            it["gold"] = gold_class(g, it["label"], it["start"], it["end"])
            it["null_family"] = rng.choice(absent)
            it.setdefault("cut_start", it["start"]); it.setdefault("cut_end", it["end"])
        items += new
    meta = {"round": "13", "model": MODEL, "question": QUESTION, "score": "max logit over yes ids - max logit over no ids",
            "p_yes": "sigmoid(score) (two-way yes/no)", "audio": "data/work/devcand/wav16/<clip>.wav (ffmpeg of the mp4, 16 kHz mono), "
            "[cut_start - 1, cut_end + 1] s clipped to the clip, at least 1 s", "stage4": f"{C.STAGE4} arm {STAGE4_ARM}",
            "P2": f"FlexSED query runs >= {FLEX_BAR}, gaps <= {FLEX_GAP} s merged, any length, not covered by same-family P1 "
            "(canonical family, overlap > 0); runs < 1 s cut as 1 s centred on the peak frame",
            "P3": f"BEATs label runs >= {BEATS_LO} with peak < {BEATS_HI}, not covered by same-family P1; identical family spans deduped",
            "gold": "score_per_sound.score_clip on the span as a lone picture: hit -> hit_needed, phantom -> none, else other_gold",
            "null": "vocab family not the family of any gold sound of the clip (random.Random(0))", "clips": len(stems)}
    C.dump(OUT, {"_meta": meta, "items": items})
    sizes(items)


def _dur(st):
    import soundfile as sf
    return float(sf.info(str(C.WAV16 / f"{st}.wav")).duration)


def sizes(items):
    for p in ("P1", "P2", "P3"):
        its = [x for x in items if x["pool"] == p]
        dep = [x for x in its if x["depictable"]]
        cnt = {k: sum(x["gold"] == k for x in dep) for k in ("hit_needed", "other_gold", "none")}
        print(f"[pool] {p}: {len(its)} spans ({len(dep)} depictable; depictable gold {cnt})", flush=True)
    print(f"[pool] total {len(items)}; asks {2 * len(items)}", flush=True)


def score():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    d = json.loads(OUT.read_text(encoding="utf-8"))
    items = d["items"]
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
    print(f"[score] model loaded in {time.time() - t0:.0f} s; yes ids {yes_ids} no ids {no_ids}", flush=True)
    d["_meta"]["yes_ids"], d["_meta"]["no_ids"] = yes_ids, no_ids
    cache = {}

    def wav(st):
        if st not in cache:
            w, sr = sf.read(str(C.WAV16 / f"{st}.wav"), dtype="float32")
            assert sr == SR and w.ndim == 1, (st, sr, w.shape)
            cache.clear(); cache[st] = w
        return cache[st]

    def ask(w, fam):                                              # listener_round.score's ask(), unchanged
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": QUESTION.format(fam.lower())}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16) if hasattr(inp, "to") else inp
        with torch.inference_mode():
            logits = model.thinker(**inp).logits[0, -1].float()
        return float(logits[yes_ids].max() - logits[no_ids].max())

    todo = sorted([it for it in items if "score" not in it], key=lambda x: (x["clip"], x["cut_start"]))
    t1 = time.time()
    for n, it in enumerate(todo, 1):
        w = wav(it["clip"])
        a, b = int(max(0.0, it["cut_start"] - 1.0) * SR), int(min(len(w) / SR, it["cut_end"] + 1.0) * SR)
        seg = w[a:max(b, a + SR)]
        it["score"] = ask(seg, it["family"])
        it["null_score"] = ask(seg, it["null_family"])
        it["p_yes"] = 1.0 / (1.0 + math.exp(-it["score"]))
        it["null_p_yes"] = 1.0 / (1.0 + math.exp(-it["null_score"]))
        if n == 10:
            print(f"[score] 10 spans in {time.time() - t1:.0f} s", flush=True)
        if n % 200 == 0:
            C.dump(OUT, d)
            print(f"[score] {n}/{len(todo)} ({time.time() - t1:.0f} s)", flush=True)
    d["_meta"]["score_seconds"] = time.time() - t0
    C.dump(OUT, d)
    print(f"[score] done {len(todo)} spans in {time.time() - t0:.0f} s -> {OUT}", flush=True)
    report()


def report():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    items = [x for x in d["items"] if "score" in x]
    sizes(d["items"])
    print(f"[report] scored {len(items)} / {len(d['items'])}")
    null = np.array([x["null_score"] for x in items])
    print(f"[report] null control (absent family): yes-rate {np.mean(null > 0):.1%} (n {len(null)})")
    for p in ("P1", "P2", "P3"):
        for dep in (True, False):
            its = [x for x in items if x["pool"] == p and (x["depictable"] or not dep)]
            if not its:
                continue
            s = np.array([x["score"] for x in its])
            ns = np.array([x["null_score"] for x in its])
            yr = {k: float(np.mean([x["score"] > 0 for x in its if x["gold"] == k] or [np.nan])) for k in ("hit_needed", "other_gold", "none")}
            hn = [x for x in its if x["gold"] in ("hit_needed", "none")]
            y = np.array([x["gold"] == "hit_needed" for x in hn])
            au = auroc(np.array([x["score"] for x in hn]), y) if y.any() and (~y).any() else float("nan")
            print(f"[report] {p} {'depictable' if dep else 'all      '} n {len(its)}: yes {np.mean(s > 0):.1%} (null {np.mean(ns > 0):.1%}); "
                  f"yes by gold {', '.join(f'{k} {v:.0%}' for k, v in yr.items())}; AUROC hit_needed {int(y.sum())} vs none "
                  f"{int((~y).sum())}: {au:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pool", "score", "report"))
    a = ap.parse_args()
    {"pool": build_pool, "score": score, "report": report}[a.step]()


if __name__ == "__main__":
    main()
