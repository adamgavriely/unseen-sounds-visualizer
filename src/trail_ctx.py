"""Pure lookups for the decision trail (src/trail.py): the exact listener questions and the cached answers behind a
stage-4 decision. Logging only -- every function here reads caches / files the decision itself read and returns text; none
calls a model or changes a cache, so no decision can depend on it.
"""
from __future__ import annotations

import json
from pathlib import Path

import config

# the exact prompts (benchmark/gold/listener_variants.py V1_Q / V4_Q, benchmark/gold/dev_listener.py yes/no,
# src/stage5_cross_modal_analysis/reason.py SCENE_FIT_PROMPT / SCENE_FIT_TWIN)
V1_Q = ("Listen carefully. Which ONE of these is actually present in this recording? A) {} B) {} C) {} D) {} "
        "E) none of A-D. Answer with a single letter.")
V2_Q = "Is the sound of {family} present in this recording? Answer yes or no."
V4_Q = "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
SCENE_Q = "Could the sound of {label} plausibly be heard in this scene? Answer yes or no."
SCENE_TWIN = "Could the sound of {label} NOT plausibly be heard in this scene? Answer yes or no."
QWEN = "Qwen3-Omni-30B-A3B"
AFN = "Audio Flamingo Next"

_ITEMS: dict = {}


def _items(path, clip):
    if not path or clip is None:
        return []
    k = (str(path), str(clip))
    if k not in _ITEMS:
        out = []
        for p in str(path).split(";"):
            if Path(p).exists():
                d = json.loads(Path(p).read_text(encoding="utf-8"))
                out += [x for x in (d["items"] if isinstance(d, dict) else d) if x.get("clip") == clip]
        _ITEMS[k] = out
    return _ITEMS[k]


def _clip():
    return getattr(config, "_CURRENT_CLIP", None)


def _fam(label):
    from src.labels import canonical
    return canonical(label)


def _p1_match(its, e, pool="P1", tol=0.02):
    """the P1 item of a span: same family, end within tol, start inside [start - tol, end]; nearest start (as stage 4)"""
    fam = _fam(e.label)
    c = [x for x in its if (pool is None or x.get("pool") == pool) and x["family"] == fam
         and abs(x["end"] - e.end) <= tol and e.start - tol <= x["start"] <= e.end]
    if not c:
        return None
    same = [x for x in c if x.get("label") == e.label] or c
    return min(same, key=lambda x: abs(x["start"] - e.start))


def _p1v4_match(e):
    fam = _fam(e.label)
    its = _items(getattr(config, "RELABEL_P1V4", None), _clip())
    c = [x for x in its if x["family"] == fam and abs(x["end"] - e.end) <= 0.02 and e.start - 0.02 <= x["start"] <= e.end]
    return min(c, key=lambda x: abs(x["start"] - e.start)) if c else None


def _cut(it):
    ra = it.get("run_audio") or [it.get("cut_start"), it.get("cut_end")]
    try:
        return f"{float(ra[0]):.2f}-{float(ra[1]):.2f} s"
    except Exception:
        return "?"


def _short(t, n=400):
    t = "" if t is None else str(t)
    return t if len(t) <= n else t[:n] + " ..."


def qwen_v4_p1(e):
    """K-V4 / WEAK-WITNESS / K4A: Qwen3-Omni's open list on the span's P1 cut (RELABEL_P1V4 cache)"""
    it = _p1v4_match(e)
    if it is None:
        return {"who": f"{QWEN} V4 (open list)", "q": V4_Q, "a": "(not asked: no P1 cut for this span)", "vote": "missing"}
    fam = _fam(e.label)
    names = it.get("qwen_fams") or []
    return {"who": f"{QWEN} V4 (open list) on {_cut(it)}", "q": V4_Q, "a": _short(it.get("qwen_v4_text")),
            "vote": ("names " + fam) if fam in names else ("does not name " + fam + "; parsed as " + (", ".join(names) or "nothing"))}


def af_v4_p1(e):
    """N2b / WEAK-WITNESS / K4A: Audio Flamingo Next's open list on the span's P1 cut (accept from LISTENER_AFCACHE P1, text
    from the AF cache, parsed families from RELABEL_P1V4)"""
    a = _p1_match(_items(getattr(config, "LISTENER_AFCACHE", None), _clip()), e)
    pv = _p1v4_match(e)
    fam = _fam(e.label)
    if a is None and pv is None:
        return {"who": f"{AFN} V4 (open list)", "q": V4_Q, "a": "(not asked: no P1 cut for this span)", "vote": "missing"}
    text = (a or {}).get("afn_v4_text") or (pv or {}).get("af_v4_text")
    acc = bool(((a or {}).get("accept") or {}).get("V4", False))
    names = (pv or {}).get("af_fams")
    vote = ("accepts " + fam) if acc else ("does not accept " + fam)
    if names is not None:
        vote += "; parsed as " + (", ".join(names) or "nothing")
    return {"who": f"{AFN} V4 (open list) on {_cut(a or pv)}", "q": V4_Q, "a": _short(text), "vote": vote}


def p1_rule(e, how):
    """F7 / N2 / DASM-clip listener rule (listener_p1_lookup): V4 / V12 on the variants cache, else yes/no score > 3"""
    fam = _fam(e.label)
    if how in ("V4", "V12"):
        it = _p1_match(_items(getattr(config, "LISTENER_VCACHE", None), _clip()), e)
        if it is None:
            return [{"who": QWEN, "q": "(P1 item not found)", "a": "", "vote": how}]
        acc = it.get("accept") or {}
        out = []
        opts = it.get("v1_x_options") or []
        p = ((it.get("v1_x") or {}).get("p") or {})
        out.append({"who": f"{QWEN} V1 (multiple choice) on {_cut(it)}",
                    "q": V1_Q.format(*(([fam] + list(opts) + ["?"] * 4)[:4])) + " (the true family is placed at A and at D)",
                    "a": "p(X) " + str(p.get("X")) + "; p(none) " + str(p.get("none")),
                    "vote": "accept" if acc.get("V1") else "reject"})
        v2 = it.get("v2_x") or {}
        out.append({"who": f"{QWEN} V2 (yes/no logit, run cut vs control cut {v2.get('ctrl')})", "q": V2_Q.format(family=fam),
                    "a": f"s_run {v2.get('s_run')}, s_ctrl {v2.get('s_ctrl')} (accept: s_run > 3 and s_run - s_ctrl > 2)",
                    "vote": "accept" if acc.get("V2") else "reject"})
        out.append({"who": "rule " + how, "q": "V1 and V2 both accept" if how == "V12" else "V4", "a": str(acc.get(how)),
                    "vote": "accept" if acc.get(how) else "reject"})
        return out
    if how == "yesno":
        it = _p1_match(_items(getattr(config, "LISTENER_CACHE", None), _clip()), e, pool="P1")
        sc = (it or {}).get("score")
        return [{"who": f"{QWEN} yes/no logit on {(it or {}).get('cut_start')}-{(it or {}).get('cut_end')} s",
                 "q": V2_Q.format(family=fam), "a": f"logit(yes) - logit(no) = {sc}",
                 "vote": "accept (> 3)" if sc is not None and float(sc) > 3.0 else "reject (<= 3)"}]
    return [{"who": QWEN, "q": V2_Q.format(family=fam), "a": "(not asked: no listener item for this span)", "vote": "missing"}]


def tier_item(it, af_item=None):
    """rescue (a) / keep (b): the TIER rule on a P2 / PV item of the variants cache (Qwen V4) + the AF cache (AF V4)"""
    if it is None:
        return [{"who": "listeners", "q": V4_Q, "a": "(not asked: no listener item for this run)", "vote": "missing"}]
    acc = it.get("accept") or {}
    fam = it.get("family", "")
    if af_item is None:
        af_item = af_for(it)
    pk = it.get("peak")
    out = [{"who": f"{QWEN} V4 (open list) on {_cut(it)}", "q": V4_Q, "a": _short(it.get("v4_text")),
            "vote": ("names " + fam) if acc.get("V4") else ("does not name " + fam)},
           {"who": f"{AFN} V4 (open list) on {_cut(af_item or it)}", "q": V4_Q,
            "a": _short((af_item or {}).get("afn_v4_text")) if af_item else "(not asked)",
            "vote": ("names " + fam) if acc.get("AF_V4") else ("does not name " + fam)}]
    out.append({"who": "rule TIER", "q": f"FlexSED run peak {pk}: >= {getattr(config, 'TIER_SPLIT', 0.6)} -> Qwen V4; below -> Qwen V4 AND AF V4",
                "a": str(acc.get("TIER")), "vote": "accept" if acc.get("TIER") else "reject"})
    return out


def af_for(it):
    """the Audio Flamingo item with the same (pool, family, label, start, end) as a variants-cache item"""
    k = (it.get("pool"), it.get("family"), it.get("label"), round(float(it["start"]), 2), round(float(it["end"]), 2))
    for x in _items(getattr(config, "LISTENER_AFCACHE", None), it.get("clip", _clip())):
        if (x.get("pool"), x.get("family"), x.get("label"), round(float(x["start"]), 2), round(float(x["end"]), 2)) == k:
            return x
    return None


def p4_asks(x):
    """DR2: the P4 item's two open lists"""
    fam = x.get("family", "")
    return [{"who": f"{QWEN} V4 (open list) on {_cut(x)}", "q": V4_Q, "a": _short(x.get("qwen_v4_text")),
             "vote": ("names " + fam) if x.get("qwen_v4") else ("does not name " + fam)},
            {"who": f"{AFN} V4 (open list) on {_cut(x)}", "q": V4_Q, "a": _short(x.get("afn_v4_text")),
             "vote": ("names " + fam) if (x.get("accept") or {}).get("V4") else ("does not name " + fam)}]


def dasm_max(e, pad=0.5):
    """DASM family max in [start - pad, end + pad] (LISTENER_DASM_DIR/<clip>.npz), or None"""
    import numpy as np
    from src.labels import canonical
    d, clip = getattr(config, "LISTENER_DASM_DIR", None), _clip()
    if not d or clip is None or not (Path(d) / f"{clip}.npz").exists():
        return None
    z = np.load(Path(d) / f"{clip}.npz", allow_pickle=True)
    L = [str(x) for x in z["labels"]]
    cols = [i for i, l in enumerate(L) if canonical(l) == canonical(e.label)]
    m = (z["times"] >= e.start - pad) & (z["times"] <= e.end + pad)
    return float(z["fw"][m][:, cols].max()) if cols and m.any() else None


def scene_asks(e):
    """SCENE-MARGIN: the logged answers of the scene question for this span (<map>.logit_answers.jsonl, last entry)"""
    from src.labels import canonical
    mp = Path(str(getattr(config, "DASM_LOCAL_SCENE", "") or ""))
    logp = mp.with_name(mp.name + (".logit_answers.jsonl" if getattr(config, "SCENE_FIT_LOGIT", False) else ".answers.jsonl"))
    key = [str(_clip()), canonical(e.label), round(float(e.start), 3), round(float(e.end), 3)]
    rec = None
    if logp.exists():
        for ln in logp.read_text(encoding="utf-8").splitlines():
            if ln.strip() and ln.startswith('{"key": ' + json.dumps(key)[:-1]):
                try:
                    r = json.loads(ln)
                except Exception:
                    continue
                if r.get("key") == key:
                    rec = r
    lab = canonical(e.label).split(",")[0].split("(")[0].strip().lower()
    if rec is None:
        return [{"who": "Qwen3.8-27B scene question", "q": SCENE_Q.format(label=lab), "a": "(no logged answer)", "vote": "?"}]
    out = []
    for a in rec.get("answers") or []:
        st = a.get("stretch")
        out.append({"who": f"Qwen3.8-27B on 6 frames, stretch {st} (logit margin, twin-cancelled)",
                    "q": SCENE_Q.format(label=lab) + "  |  twin: " + SCENE_TWIN.format(label=lab),
                    "a": f"d = margin(Q) - margin(twin) = {a.get('d')}" if "d" in a else str(a.get("answer")),
                    "vote": a.get("answer") or "none"})
    out.append({"who": "majority of stretches", "q": "yes > no", "a": str(rec.get("verdict")),
                "vote": "plausible" if rec.get("verdict") is True else "not plausible"})
    return out


def label_rule(label: str) -> str:
    """which LABEL_FILTER 'depictable' rule refuses a label (the name only; the decision is labels.is_salient_nonspeech)"""
    from src import labels as L
    if label in L.TEXTURE_LABELS or label == "Wind" or L.is_descendant(label, "Wind"):
        return "steady texture / wind (raw name)"
    lab = L.canonical(label)
    if lab in L.SPEECH_LABELS or any(L.is_descendant(lab, sp) for sp in L.SPEECH_LABELS):
        return f"speech ({lab})"
    if L.is_music(lab) or L.is_descendant(lab, "Singing"):
        return f"music / singing ({lab})"
    if lab in (L.ENV_BRANCH, "Silence", "Sound effect", "Video game sound", "Human voice", "Respiratory sounds") \
            or L.is_descendant(lab, L.ENV_BRANCH):
        return f"environment / no-source name ({lab})"
    if lab in L.GENERIC_LABELS:
        return f"generic name ({lab})"
    if lab in L.TEXTURE_LABELS:
        return f"texture ({lab})"
    if lab == "Source-ambiguous sounds" or L.is_descendant(lab, "Source-ambiguous sounds"):
        return f"source-ambiguous ({lab})"
    return f"not drawable ({lab})"
