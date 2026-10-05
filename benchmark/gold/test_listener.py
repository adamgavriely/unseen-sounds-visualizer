"""The audio-LLM listener cache for the 59 TEST clips, gold-free, a SUPERSET of dev_listener.py's pools so it
serves any R13-3 arm. PREPARATION only: no TEST gold is read, nothing is scored against gold.

Same model, question, yes/no token ids, score and audio cut as benchmark/gold/dev_listener.py (its score() is reused
unchanged); audio = data/work/r13test/wav16/<clip>.wav. Pools per clip:
  P1  every B0r stage-4 span on TEST (data/work/r13test/stage4.json, B0r|proposed and B0r|blind_a2i, deduped)
  P2  every FlexSED query run >= 0.4 (gaps <= 0.24 s merged, any length; < 1 s cut as 1 s around the peak) -- dev_listener's
      P2 construction WITHOUT the "not covered by P1" filter
  P3  every BEATs label run >= 0.175 with peak < 0.35 (identical family spans deduped) -- P3 WITHOUT the coverage filter
  PV  every FlexSED-only span the PANNs clip veto removed in the scored TEST run (onset_trace: union & flexsed_raw - veto);
      the harness's (b) lookup reads the P2 run that contains it, which the pool build checks exists
Null control, gold-free: a vocab family with no BEATs or FlexSED score >= 0.1 anywhere in the clip (canonical family or
score_per_sound.same_family), random.Random(0), one choice per item in pool order (as dev_listener).
No "gold" field. Item keys as dev_listener.json, so src.stage4_audio_event_detection.listener_from_cache reads it.

    python benchmark/gold/test_listener.py devcheck   # CPU: the superset construction on DEV vs dev_listener.json keys
    python benchmark/gold/test_listener.py pool       # CPU: TEST pool -> benchmark/gold/test_listener.json
    python benchmark/gold/test_listener.py score      # GPU: dev_listener.score() on the TEST file (resumable)
    python benchmark/gold/test_listener.py report     # CPU: sizes, yes-rates per pool, null yes-rate (no gold)

(design record: release v1.2.0)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

OUT = _ROOT / "benchmark" / "gold" / "test_listener.json"
DEV_JSON = _ROOT / "benchmark" / "gold" / "dev_listener.json"
NULL_BAR = 0.1


def _mods():
    from benchmark.gold import dev_candidates_check as C
    from benchmark.gold import dev_listener as L
    from benchmark.gold import score_per_sound as S
    from src.labels import canonical
    return C, L, S, canonical


def superset(stems, p1_by_clip, beats_dir, flex_dir, wav16_dir, vetoed_by_clip=None, null=True):
    """the superset pool; pure (every path passed in), never reads gold"""
    import soundfile as sf
    C, L, S, canonical = _mods()
    from benchmark.listener_round import VOCAB
    rng = random.Random(0)
    items, contain_miss = [], []
    for st in stems:
        dur = float(sf.info(str(wav16_dir / f"{st}.wav")).duration)
        new = []
        for r in p1_by_clip.get(st, []):
            new.append({"pool": "P1", "label": r["label"], "start": float(r["start"]), "end": float(r["end"]),
                        "conf": float(r["conf"]), "origin": r["origin"], "run_len": float(r["end"] - r["start"])})
        present = {}
        Ff = C.load_fr(flex_dir / f"{st}.npz")
        fw, ts, labs = Ff
        p2 = []
        for c, lab in enumerate(labs):
            present[lab] = max(present.get(lab, 0.0), float(fw[:, c].max()) if len(fw) else 0.0)
            rr, dt = L.runs(fw[:, c], ts, L.FLEX_BAR, L.FLEX_GAP)
            for i, j in rr:
                a, b = float(ts[i]), float(ts[j - 1] + dt)
                k = i + int(np.argmax(fw[i:j, c]))
                pk_t = float(ts[k] + dt / 2)
                it = {"pool": "P2", "label": lab, "start": a, "end": b, "peak": float(fw[k, c]), "peak_t": pk_t, "run_len": b - a}
                if b - a < L.MIN_CUT:
                    ca = min(max(0.0, pk_t - L.MIN_CUT / 2), max(0.0, dur - L.MIN_CUT))
                    it["cut_start"], it["cut_end"] = ca, ca + L.MIN_CUT
                p2.append(it)
        new += p2
        fw, ts, labs = C.load_fr(beats_dir / f"{st}.npz")
        seen = set()
        for c, lab in enumerate(labs):
            present[lab] = max(present.get(lab, 0.0), float(fw[:, c].max()) if len(fw) else 0.0)
            fam = canonical(lab)
            rr, dt = L.runs(fw[:, c], ts, L.BEATS_LO, 0.0)
            for i, j in rr:
                pk = float(fw[i:j, c].max())
                if pk >= L.BEATS_HI:
                    continue
                a, b = float(ts[i]), float(ts[j - 1] + dt)
                if (fam, round(a, 3), round(b, 3)) in seen:
                    continue
                seen.add((fam, round(a, 3), round(b, 3)))
                k = i + int(np.argmax(fw[i:j, c]))
                new.append({"pool": "P3", "label": lab, "start": a, "end": b, "peak": pk, "peak_t": float(ts[k] + dt / 2),
                            "run_len": b - a})
        for (lab, a, b) in (vetoed_by_clip or {}).get(st, []):
            new.append({"pool": "PV", "label": lab, "start": float(a), "end": float(b), "run_len": float(b - a)})
            fam = canonical(lab)
            if not any(canonical(x["label"]) == fam and x["start"] <= a + 0.02 and x["end"] >= b - 0.02 for x in p2):
                contain_miss.append([st, lab, a, b])
        if null:
            hot = [l for l, v in present.items() if v >= NULL_BAR]
            hotf = {canonical(l) for l in hot}
            absent = [v for v in VOCAB if canonical(v) not in hotf and not any(S.same_family(v, l) for l in hot)]
            assert absent, st
        for it in new:
            it["clip"] = st
            it["family"] = canonical(it["label"])
            it["depictable"] = L.depictable(it["label"])
            if null:
                it["null_family"] = rng.choice(absent)
            it.setdefault("cut_start", it["start"]); it.setdefault("cut_end", it["end"])
        items += new
    return items, contain_miss


def _vetoed(trace_path):
    tr = json.loads(trace_path.read_text(encoding="utf-8"))
    sig = lambda n: {(x["label"], float(x["start"]), float(x["end"])) for x in tr if x["step"] == n}
    return sorted((sig("union") & sig("flexsed_raw")) - sig("veto"))


def pool():
    from benchmark.gold import test_harness_prep as T      # gold guard + TEST redirects (TAG, BEATS_DIR, WAV16)
    C, L, S, canonical = _mods()
    s4 = json.loads(T.R.STAGE4.read_text(encoding="utf-8"))["arms"]
    p1 = {}
    for st in T.STEMS:
        rows, keys = [], set()
        for sysn in ("proposed", "blind_a2i"):
            for r in s4[f"B0r|{sysn}"][st]:
                k = (r["label"], round(r["start"], 3), round(r["end"], 3))
                if k not in keys:
                    keys.add(k); rows.append(r)
        p1[st] = rows
    vet = {st: _vetoed(C.scored_dir("proposed") / st / "onset_trace.json") for st in T.STEMS}
    items, miss = superset(T.STEMS, p1, C.BEATS_DIR, C.FLEX_DIR, C.WAV16, vet)
    from benchmark.listener_round import MODEL, QUESTION
    meta = {"round": "13", "split": "TEST (59 clips, benchmark/gold/test_stems.txt); gold-free", "model": MODEL,
            "question": QUESTION, "score": "max logit over yes ids - max logit over no ids (dev_listener.score, unchanged)",
            "audio": f"{C.WAV16}/<clip>.wav, [cut_start - 1, cut_end + 1] s clipped to the clip, at least 1 s",
            "P1": f"{T.R.STAGE4} B0r|proposed + B0r|blind_a2i, deduped",
            "P2": f"FlexSED query runs >= {L.FLEX_BAR}, gaps <= {L.FLEX_GAP} s merged, any length, NO coverage filter; runs < 1 s "
                  "cut as 1 s centred on the peak frame",
            "P3": f"BEATs label runs >= {L.BEATS_LO} with peak < {L.BEATS_HI}, NO coverage filter; identical family spans deduped",
            "PV": "FlexSED-only spans removed by the PANNs clip veto in the scored run (onset_trace union & flexsed_raw - veto)",
            "null": f"vocab family with no BEATs/FlexSED score >= {NULL_BAR} in the clip (canonical or same_family), "
                    "random.Random(0)", "pv_not_contained_in_P2": miss, "clips": len(T.STEMS)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    old = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else None
    if old and any("score" in x for x in old["items"]):
        raise SystemExit(f"{OUT} already has scores; not overwriting")
    C.dump(OUT, {"_meta": meta, "items": items})
    sizes(items)
    print(f"[pool] PV spans not inside a same-family P2 run: {len(miss)} {miss}", flush=True)


def sizes(items):
    for p in ("P1", "P2", "P3", "PV"):
        its = [x for x in items if x["pool"] == p]
        print(f"[pool] {p}: {len(its)} spans ({sum(x['depictable'] for x in its)} depictable)", flush=True)
    print(f"[pool] total {len(items)}; asks {2 * len(items)}", flush=True)


def report():
    d = json.loads(OUT.read_text(encoding="utf-8"))
    items = [x for x in d["items"] if "score" in x]
    sizes(d["items"])
    print(f"[report] scored {len(items)} / {len(d['items'])}", flush=True)
    if not items:
        return
    print(f"[report] null control yes-rate {np.mean([x['null_score'] > 0 for x in items]):.1%} (n {len(items)})", flush=True)
    for p in ("P1", "P2", "P3", "PV"):
        its = [x for x in items if x["pool"] == p]
        if its:
            print(f"[report] {p} n {len(its)}: yes {np.mean([x['score'] > 0 for x in its]):.1%} "
                  f"(null {np.mean([x['null_score'] > 0 for x in its]):.1%})", flush=True)


def score():
    from benchmark.gold import test_harness_prep as T      # noqa: F401  gold guard; C.WAV16 -> data/work/r13test/wav16
    C, L, S, canonical = _mods()
    assert C.WAV16 == T.OUT / "wav16", C.WAV16
    L.OUT = OUT
    L.report = report                                   # dev_listener.report reads "gold"; the TEST one does not
    L.score()


def devcheck():
    """the superset construction on DEV (DEV paths, no model) vs dev_listener.json: every DEV P1/P2/P3 key must be in the
    superset's same pool with the same cut; the extra superset items are the ones the DEV coverage filter left out"""
    C, L, S, canonical = _mods()
    d = json.loads(DEV_JSON.read_text(encoding="utf-8"))["items"]
    stems = sorted({x["clip"] for x in d})
    s4 = json.loads(C.STAGE4.read_text(encoding="utf-8"))["arms"][L.STAGE4_ARM]
    items, _m = superset(stems, {st: s4[st] for st in stems}, C.BEATS_DIR, C.FLEX_DIR, C.WAV16, None, null=False)
    k = lambda x: (x["pool"], x["clip"], x["family"], round(x["start"], 3), round(x["end"], 3))
    cut = lambda x: (round(x["cut_start"], 3), round(x["cut_end"], 3))
    sup = {}
    for x in items:
        sup.setdefault(k(x), []).append(x)
    res = {}
    for p in ("P1", "P2", "P3"):
        dv = [x for x in d if x["pool"] == p]
        miss = [k(x) for x in dv if k(x) not in sup]
        cutd = [k(x) for x in dv if k(x) in sup and cut(x) not in {cut(y) for y in sup[k(x)]}]
        labd = [k(x) for x in dv if k(x) in sup and x["label"] not in {y["label"] for y in sup[k(x)]}]
        n_sup = sum(1 for x in items if x["pool"] == p)
        res[p] = {"dev": len(dv), "superset": n_sup, "missing": len(miss), "cut_differs": len(cutd), "label_differs": len(labd),
                  "examples": (miss + cutd + labd)[:5]}
        print(f"[devcheck] {p}: DEV {len(dv)} keys, superset {n_sup}; missing {len(miss)}, cut differs {len(cutd)}, "
              f"label differs {len(labd)} {res[p]['examples']}", flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("devcheck", "pool", "score", "report"))
    a = ap.parse_args()
    {"devcheck": devcheck, "pool": pool, "score": score, "report": report}[a.step]()


if __name__ == "__main__":
    main()
