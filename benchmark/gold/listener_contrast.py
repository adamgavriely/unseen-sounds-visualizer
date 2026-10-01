"""Round 39 CONTRAST (docs/prereg_round13_detector_push.md, "Round 39 CONTRAST"): on every merged-DEV P2/PV candidate that the
shipped TIER rule rejects (peak >= TIER_SPLIT: Qwen V4; below: Qwen V4 AND AF V4), ask Qwen3-Omni a forced choice on the
SAME listener cut (run_audio, as listener_v4d, no ducking):
    "Which sound is in this recording? (a) {A} (b) {B} (c) neither"
in both A/B orders ((c) stays last). A = the candidate's family; B = the family with the highest raw frame score inside
run_audio over the FlexSED / BEATs / DASM caches (max over the family's columns and the three models), excluding families
related to A (listener_variants.related) and speech / music (SPEECH_LABELS + descendants, is_music, Singing descendants).
Tie: equal score -> FlexSED before BEATs before DASM -> alphabetical family. No B (no cache, nothing left) -> not asked.
Greedy, max_new_tokens 8; the answer is the first "(a)" / "(b)" / "(c)" (or lone letter) in the reply; unparsed = not A.
accept {"CONTRAST": A chosen in BOTH orders}. TIER-accepted items are copied with asked = False. No gold read here.

    python benchmark/gold/listener_contrast.py VCACHE:AFCACHE:WAVDIR:BEATSDIR:DASMDIR:OUT [...]
"""
from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import listener_variants as LV
from benchmark.gold import dev_candidates_check as C
from src.labels import SPEECH_LABELS, canonical, is_descendant, is_music

Q = "Which sound is in this recording? (a) {} (b) {} (c) neither"
NEW = 8
SPLIT = float(getattr(config, "TIER_SPLIT", 0.6))
MODELS = ("flex", "beats", "dasm")                                     # tie-break order


def key(x):
    return (x["clip"], x["pool"], x["label"], round(x["start"], 2), round(x["end"], 2))


def tier(q, a, peak):
    return bool(q) if peak >= SPLIT else bool(q and a)


def speech_or_music(fam):
    return (fam in SPEECH_LABELS or any(is_descendant(fam, s) for s in SPEECH_LABELS) or is_music(fam)
            or fam == "Singing" or is_descendant(fam, "Singing"))


def load(p):
    if not p.exists():
        return None
    fw, ft, labs = C.load_fr(p)
    assert fw.shape[0] == len(ft) and fw.shape[1] == len(labs), (p, fw.shape)
    return fw, ft, labs


def competitor(caches, fam_a, a, b):
    """(B, score, model) = the family with the highest raw frame score inside [a, b] over the three caches, or None"""
    best = None                                                        # (-score, model rank, family)
    for mi, m in enumerate(MODELS):
        fr = caches.get(m)
        if fr is None:
            continue
        fw, ft, labs = fr
        sel = (ft >= a - 1e-6) & (ft <= b + 1e-6)
        if not sel.any():
            continue
        fams = {}
        for c, l in enumerate(labs):
            f = canonical(l)
            if LV.related(fam_a, f) or speech_or_music(f):
                continue
            fams[f] = max(fams.get(f, -1.0), float(fw[sel, c].max()))
        for f, s in fams.items():
            cand = (-s, mi, f)
            if best is None or cand < best:
                best = cand
    return None if best is None else (best[2], -best[0], MODELS[best[1]])


def parse(txt):
    m = re.search(r"\(([abc])\)|\b([abc])\b", txt.strip().lower())
    return (m.group(1) or m.group(2)) if m else None


def main():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    LV.S.load_gold = LV._no_gold
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass

    def gen(w, q):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=NEW, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    for spec in sys.argv[1:]:
        vc, afc, wavdir, bdir, ddir, outp = spec.split(":")
        items = [x for x in json.loads(Path(vc).read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")]
        af = {key(x): x for x in json.loads(Path(afc).read_text(encoding="utf-8"))["items"]}
        out, wcache, ccache, t0 = [], {}, {}, time.time()
        n = {"items": 0, "tier": 0, "asked": 0, "no_B": 0, "accept": 0, "unparsed": 0}
        for it in sorted(items, key=lambda x: (x["clip"], x["run_audio"][0])):
            st = it["clip"]
            q4 = bool(it["accept"].get("V4"))
            a4 = bool((af.get(key(it), {}).get("accept") or {}).get("V4"))
            peak = float(it.get("peak") or 0.0)
            t = tier(q4, a4, peak)
            rec = {kk: it[kk] for kk in ("clip", "pool", "family", "label", "start", "end", "run_audio") if kk in it}
            rec |= {"peak": peak, "qwen_v4": q4, "af_v4": a4, "tier": t, "asked": False, "B": None,
                    "accept": {"CONTRAST": False}}
            n["items"] += 1
            if t:
                n["tier"] += 1
                out.append(rec)
                continue
            if st not in ccache:
                ccache.clear()
                ccache[st] = {"flex": load(C.FLEX_DIR / f"{st}.npz"), "beats": load(Path(bdir) / f"{st}.npz"),
                              "dasm": load(Path(ddir) / f"{st}.npz")}
            a, b = float(it["run_audio"][0]), float(it["run_audio"][1])
            comp = competitor(ccache[st], it["family"], a, b)
            if comp is None:
                n["no_B"] += 1
                out.append(rec | {"B": None, "no_B": True})
                continue
            B, bs, bm = comp
            if st not in wcache:
                w, sr = sf.read(str(Path(wavdir) / f"{st}.wav"), dtype="float32")
                assert sr == LV.SR, (st, sr)
                wcache.clear(); wcache[st] = w
            w = wcache[st]
            i, j = int(a * LV.SR), int(b * LV.SR)
            seg = w[i:max(j, i + LV.SR)]
            A = it["family"]
            t_ab, t_ba = gen(seg, Q.format(A, B)), gen(seg, Q.format(B, A))
            p_ab, p_ba = parse(t_ab), parse(t_ba)
            n["unparsed"] += (p_ab is None) + (p_ba is None)
            ok = p_ab == "a" and p_ba == "b"
            n["asked"] += 1; n["accept"] += ok
            out.append(rec | {"asked": True, "B": B, "B_score": bs, "B_model": bm, "text_AB": t_ab, "text_BA": t_ba,
                              "ans_AB": p_ab, "ans_BA": p_ba, "accept": {"CONTRAST": bool(ok)}})
        C.dump(Path(outp), {"_meta": {"prompt": Q, "max_new_tokens": NEW, "tier_split": SPLIT, "vcache": vc, "afcache": afc,
                                      "beats": bdir, "dasm": ddir, "flex": str(C.FLEX_DIR), "counts": n}, "items": out})
        print(f"[contrast] {outp}: {json.dumps(n)}, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
