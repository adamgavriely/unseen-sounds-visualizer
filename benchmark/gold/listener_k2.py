"""Round 14 amendment K2 supplement (docs/prereg_round13_detector_push.md): the listener answers for the FlexSED band runs that
K2 (RESCUE_COVERED) newly offers to the rescue -- runs a same-family BEATs span below the display bar covers, so they were
never in the P2 pool and have no cached answer. Same two verifiers, prompts, decoding, matching and audio cut as the
amendment-A / C caches, by importing their code unchanged:
  Qwen3-Omni   benchmark/gold/listener_variants.py (V1, V2, V3, V4; V12 = V1 and V2)   -> dev_listener_k2.json
  AF Next      benchmark/gold/listener_afnext.py   (V4 + yes/no)                     -> dev_listener_k2_afn.json
Candidates: data/work/r13/k2_runs.json (the 27 runs arm TO1F7F8+K2 asked for and found missing, from its stage-4 log). Each is
built like a P2 item of listener_variants.build (cut = the run, or 1 s centred on its peak frame when shorter than 1 s; V1
distractors; V2 control window; V3 cut). Null family: listener_variants.gold_free_null (the gold-free rule of the TEST pool;
the DEV P2 nulls came from the gold-based dev cache, which these runs are not in). No gold is read. DEV only.

    python benchmark/gold/listener_k2.py pool     # CPU
    python benchmark/gold/listener_k2.py score    # GPU: Qwen3-Omni, then AF Next (one model at a time)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import dev_listener as L
from benchmark.gold import listener_variants as LV
from benchmark.gold import score_per_sound as S
from src.labels import canonical

GOLD = _ROOT / "benchmark" / "gold"
RUNS = C.WORK / "r13" / "k2_runs.json"
OUT_Q = GOLD / "dev_listener_k2.json"
OUT_A = GOLD / "dev_listener_k2_afn.json"
WAV = LV.SPLITS["dev"]["wav"]


def pool():
    import soundfile as sf
    S.load_gold = LV._no_gold
    runs = json.loads(RUNS.read_text(encoding="utf-8"))
    rng = random.Random(0)
    O = LV.Onto()
    out, flex = [], {}
    for r in runs:
        st, lab = r["clip"], r["label"]
        dur = float(sf.info(str(WAV / f"{st}.wav")).duration)
        run = LV.run_from_flex(st, lab, float(r["start"]), float(r["end"]), dur)
        assert run is not None, r
        it = {"clip": st, "pool": "P2", "family": canonical(lab), "label": lab, "start": run["start"], "end": run["end"],
              "cut_start": run["cut_start"], "cut_end": run["cut_end"], "run_len": run["run_len"], "peak": run["peak"],
              "depictable": L.depictable(lab), "run_start": run["start"], "run_end": run["end"],
              "variants": ["V1", "V2", "V3", "V4"], "source": "K2 (covered by a sub-display BEATs span)"}
        it["null_family"] = LV.gold_free_null(st, rng)
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            ds, src = LV.distractors(O, st, fam, it["start"])
            it[f"v1_{tag}_options"], it[f"v1_{tag}_source"] = ds, src
        if st not in flex:
            flex[st] = C.load_fr(LV.FLEX_DIR / f"{st}.npz")
        a = max(0.0, it["cut_start"] - 1.0); b = min(dur, it["cut_end"] + 1.0)
        it["run_audio"] = [a, max(b, a + 1.0)]
        for tag, fam in (("x", it["family"]), ("null", it["null_family"])):
            it[f"v2_{tag}_ctrl"], it[f"v2_{tag}_flex_col"] = LV.control(flex[st], fam, it, dur)
        it["v3_cut"] = [max(0.0, it["run_start"] - 3.0), min(dur, it["run_end"] + 3.0)]
        out.append(it)
    meta = {"amendment": "round 14 K2 supplement", "source": str(RUNS), "model": LV.MODEL, "V1": LV.V1_Q, "V3": LV.V3_Q,
            "V4": LV.V4_Q, "null": "listener_variants.gold_free_null, random.Random(0)", "n": len(out)}
    C.dump(OUT_Q, {"_meta": meta, "items": out})
    print(f"[k2 pool] {len(out)} items -> {OUT_Q}; peaks {sorted(round(x['peak'], 2) for x in out)}", flush=True)


def score():
    import gc
    import torch
    from benchmark.gold import listener_afnext as LA
    S.load_gold = LV._no_gold
    LV.SPLITS = {"dev": {"cache": None, "out": OUT_Q, "wav": WAV}}          # listener_variants.score on these items only
    LV.score()
    gc.collect(); torch.cuda.empty_cache()
    M = LA.load_model()
    if OUT_A.exists():
        d = json.loads(OUT_A.read_text(encoding="utf-8"))
        items, meta = d["items"], d["_meta"]
    else:
        items, meta = LA.base_items("dev"), LA.make_meta(M, "dev")
    LA.run(M, items, "dev", OUT_A, meta)
    LA.summary(items, "dev")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pool", "score"))
    a = ap.parse_args()
    {"pool": pool, "score": score}[a.step]()


if __name__ == "__main__":
    main()
