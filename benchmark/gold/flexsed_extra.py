"""FlexSED extra queries (2026-09-29): the drawable sounds that FlexSED is never asked about.

The FlexSED cache (data/work/flexsed_cache, benchmark/gold/flexsed_run.py) asks one query per depictable FAMILY (215).
Two kinds of sound have no query of their own:
  (a) "folded": the 120 labels of benchmark/gold/depictable_vocab.json whose name is not one of the 215 families
      (Bang is asked only as "Explosion", Slam only as "Door", Shatter only as "Glass" ...);
  (b) "outside": AudioSet labels that can be a gold sound (the gold loader's default filter) but are not in the
      depictable vocab at all -- the Source-ambiguous event sounds (Clang, Whack/thwack, Clatter, Slap ...). Category
      nodes, textures (src.labels.TEXTURE_LABELS), sourceless tones (fixed list SKIP_B below), music, voice and the
      recording/environment branch are left out.
The list is built from the vocab and the ontology only; no gold label is read to build it.

    python benchmark/gold/flexsed_extra.py queries                        # CPU: write benchmark/gold/flexsed_extra_queries.json
    python benchmark/gold/flexsed_extra.py run --set dev|test             # GPU: FlexSED, same code/settings/audio as the cache
    python benchmark/gold/flexsed_extra.py screen                         # CPU, DEV only: -> benchmark/gold/flexsed_extra_dev.json

`run` never opens a gold file (TEST: features only). Output: data/work/flexsed_extra_<set>/<stem>.npz with
fw [n_new, T] float16, labels (the new queries only), fps 25 -- the cache's own format.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
VOCAB = _ROOT / "benchmark" / "gold" / "depictable_vocab.json"
QFILE = _ROOT / "benchmark" / "gold" / "flexsed_extra_queries.json"
OUTJ = _ROOT / "benchmark" / "gold" / "flexsed_extra_dev.json"
STEMS = {"dev": _ROOT / "benchmark" / "gold" / "dev_stems.txt", "test": _ROOT / "benchmark" / "gold" / "test_stems.txt"}
CACHE = _ROOT / "data" / "work" / "flexsed_cache"
WAVS = _ROOT / "data" / "work" / "gold_wav_flat"          # the 16 kHz mono wavs the cache was computed from
FLEXSED = Path(os.environ.get("FLEXSED_ROOT", Path.home() / "FlexSED"))
FPS = 25.0

# (b) candidates that are not one sound event: grouping nodes, sourceless tones, textures (never drawn)
SKIP_B = {"Brief tone", "Clicking", "Deformable shell", "Other sourceless", "Sine wave", "Source-ambiguous sounds",
          "Human voice", "Respiratory sounds", "Channel, environment and background", "Bass (frequency range)",
          "Infrasound", "Pulse", "Chirp tone", "Harmonic", "Hum", "Rumble", "Whir", "Rustle", "Breathing"}


def out_dir(which: str) -> Path:
    return _ROOT / "data" / "work" / f"flexsed_extra_{which}"


# ============================================================================= query list (CPU)
def build_queries():
    sys.path.insert(0, str(_ROOT))
    import config
    from src.labels import canonical, is_descendant, is_salient_nonspeech
    from benchmark.gold import score_per_sound as S
    d = json.loads(VOCAB.read_text(encoding="utf-8"))
    fams, labs = d["families"], d["labels"]
    V = set(fams) | set(labs)
    folded = [{"query": l, "bucket": "a_folded", "asked_as": canonical(l)} for l in labs if l not in set(fams)]
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "lists"                    # the gold loader's filter (score_per_sound at import time)
    try:
        outside = sorted(n for n in S._ontology_names()
                         if n not in V and n not in SKIP_B and is_salient_nonspeech(n)
                         and n != "Music" and not is_descendant(n, "Music")
                         and not is_descendant(n, "Channel, environment and background") and not is_descendant(n, "Human voice"))
        config.LABEL_FILTER = "depictable"
        drawn = {n: bool(is_salient_nonspeech(n)) for n in outside}
    finally:
        config.LABEL_FILTER = old
    outside = [{"query": n, "bucket": "b_outside", "asked_as": None,
                "nearest_family": [f for f in fams if S.same_family(n, f)],
                "drawn_under_depictable_filter": drawn[n]} for n in outside]
    q = folded + outside
    QFILE.write_text(json.dumps({"note": __doc__.split("\n\n")[1], "n": len(q), "queries": q}, indent=1), encoding="utf-8")
    print(f"{len(folded)} folded + {len(outside)} outside = {len(q)} -> {QFILE}")
    print("outside:", [x["query"] for x in outside])


# ============================================================================= FlexSED (GPU)
def wav_of(stem: str) -> Path | None:
    w = WAVS / f"{stem}.wav"
    if w.exists():
        return w
    # fallback = flexsed_run.clip_path + wav_for (same ffmpeg call), without reading the gold file
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        for p in (_ROOT / "data" / "input" / "benchmark" / sub).glob(stem + ".*"):
            WAVS.mkdir(parents=True, exist_ok=True)
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ac", "1", "-ar", "16000", str(w)], check=True)
            print("made wav", w, flush=True)
            return w
    for p in (_ROOT / "data" / "input" / "audioset_strong").glob(stem + ".*"):
        WAVS.mkdir(parents=True, exist_ok=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(p), "-ac", "1", "-ar", "16000", str(w)], check=True)
        print("made wav", w, flush=True)
        return w
    return None


def run(which: str, batch: int = 24):
    queries = [x["query"] for x in json.loads(QFILE.read_text(encoding="utf-8"))["queries"]]
    vocab = json.loads(VOCAB.read_text(encoding="utf-8"))["families"]
    stems = STEMS[which].read_text().split()
    assert len(stems) == {"dev": 49, "test": 60}[which], len(stems)
    out = out_dir(which); out.mkdir(parents=True, exist_ok=True)
    wavs = {s: wav_of(s) for s in stems}
    # --- from here: benchmark/gold/flexsed_run.py's setup, unchanged (no --prompts, batch 24) ---
    sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != _ROOT]
    sys.modules.pop("src", None)
    local_clap = Path.home() / "clap-htsat-unfused-st"
    if local_clap.exists():
        import transformers
        _clap, _tok = transformers.ClapTextModelWithProjection, transformers.AutoTokenizer
        _from_clap, _from_tok = _clap.from_pretrained, _tok.from_pretrained
        _clap.from_pretrained = staticmethod(lambda name, *x, **k: _from_clap(str(local_clap) if "clap" in str(name) else name, *x, **k))
        _tok.from_pretrained = staticmethod(lambda name, *x, **k: _from_tok(str(local_clap) if "clap" in str(name) else name, *x, **k))
    sys.path.insert(0, str(FLEXSED))
    os.chdir(FLEXSED)
    from api import FlexSED
    import torch
    m = FlexSED(device="cuda")
    _split = m.split_audio_fixed

    def split(audio, sr, chunk_duration=10.0):
        o = []
        for c in _split(audio, sr, chunk_duration):
            if len(c) < sr:
                c = np.pad(c, (0, sr - len(c)))
            o.append(c)
        return o
    m.split_audio_fixed = split

    def infer(wav, qs):
        parts = []
        for k in range(0, len(qs), batch):
            with torch.inference_mode():
                preds = m.run_inference(str(wav), qs[k:k + batch])
            parts.append(preds.detach().squeeze(1).numpy().astype(np.float16))
            del preds
            torch.cuda.empty_cache()
        return np.concatenate(parts, axis=0)

    # reproduction check: the cache's first batch (24 existing families) on the first stem, recomputed here and
    # compared to the stored cache -- shows this setup gives the cache's numbers; nothing of it is saved
    s0 = stems[0]
    if (CACHE / f"{s0}.npz").exists() and wavs.get(s0):
        ref = np.load(CACHE / f"{s0}.npz")["fw"][:batch].astype(np.float32)
        new = infer(wavs[s0], vocab[:batch]).astype(np.float32)
        print(f"REPRO {s0}: shape {new.shape} vs {ref.shape}, max |diff| {float(np.abs(new - ref).max()):.4f}", flush=True)
    for i, s in enumerate(stems, 1):
        f = out / f"{s}.npz"
        if f.exists():
            continue
        if wavs.get(s) is None:
            print("missing", s, flush=True); continue
        fw = infer(wavs[s], queries)
        np.savez_compressed(f, fw=fw, labels=np.array(queries), fps=FPS)
        print(f"[{i}] {s} {fw.shape}", flush=True)
    print("done ->", out)


# ============================================================================= DEV screen (CPU)
RUN_BAR, RUN_GAP = 0.4, 0.24           # the pipeline's FlexSED run rule (stage 4 _runs: LISTEN_RUN_BAR / LISTEN_RUN_GAP)
BARS = (0.4, 0.5, 0.8)
EARLY, LATE = 0.5, 1.0                 # score_per_sound's onset window
# the four DEV misses no detector hears (docs/history/analyses/dev_miss_table_2026-09-28.md bucket (a), release v1.2.0)
UNHEARD = [("ambient_citywalk_nyc_1689", "Hammer", 8.1), ("ambient_citywalk_nyc_2627", "Clang", 3.8),
           ("b3_golf_course", "Whack, thwack", 6.5), ("b3_golf_course", "Whack, thwack", 24.4)]
# existing queries shown next to the new ones at those onsets
NEAREST = ["Hammer", "Knock", "Explosion", "Door", "Bell", "Dishes", "Glass", "Gunshot"]


def _runs(col, dt, bar, gap_s):
    """copy of src/stage4_audio_event_detection._runs: frames >= bar, runs separated by <= gap_s merged"""
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
    return out


def _load(p):
    z = np.load(p, allow_pickle=False)
    return z["fw"].astype(np.float32), [str(x) for x in z["labels"]], float(z["fps"])


def screen():
    sys.path.insert(0, str(_ROOT))
    import config
    from benchmark.gold import score_per_sound as S
    assert getattr(config, "LABEL_FILTER", "lists") == "lists", "gold must load under the default filter (as DCC.dev_stems)"
    gold = S.load_gold([_ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"])     # = DCC.dev_stems()
    stems = sorted(S.subsets_of(gold)["dev"])
    assert len(stems) == 49 and set(stems) == set(STEMS["dev"].read_text().split())
    qmeta = {x["query"]: x for x in json.loads(QFILE.read_text(encoding="utf-8"))["queries"]}
    res = {"what": "FlexSED extra queries, DEV screen (gold read: DEV only)", "n_queries": len(qmeta),
           "run_rule": f"frames >= bar, runs with gaps <= {RUN_GAP} s merged (stage-4 _runs); a run is counted at bar b if its peak >= b "
                       f"(the runs are cut at {RUN_BAR}); it 'falls on' a gold sound if it overlaps [start - {EARLY}, end] "
                       f"and same_family(query, gold label); 'needed' = that gold sound is needed",
           "unheard": [], "runs": {}, "per_query": {}}
    # --- the four unheard misses
    for st, lab, on in UNHEARD:
        g = [x for x in gold[st] if x["label"] == lab and abs(x["start"] - on) < 0.15]
        assert g, (st, lab, on)
        on = g[0]["start"]
        fw, ql, fps = _load(out_dir("dev") / f"{st}.npz")
        a, b = max(0, int(np.floor((on - EARLY) * fps))), int(np.ceil((on + LATE) * fps)) + 1
        pk = fw[:, a:b].max(axis=1)
        order = np.argsort(-pk)
        new_same = {q: round(float(pk[k]), 3) for k, q in enumerate(ql) if S.same_family(q, lab)}
        efw, el, efps = _load(CACHE / f"{st}.npz")
        epk = efw[:, a:b].max(axis=1)
        eo = np.argsort(-epk)
        res["unheard"].append({
            "clip": st, "label": lab, "onset": round(on, 2), "window": [round(on - EARLY, 2), round(on + LATE, 2)],
            "new_same_family": new_same,
            "new_top5": [(ql[k], round(float(pk[k]), 3)) for k in order[:5]],
            "existing_nearest": {q: round(float(epk[el.index(q)]), 3) for q in NEAREST if q in el},
            "existing_top3": [(el[k], round(float(epk[k]), 3)) for k in eo[:3]]})
    # --- runs of the new queries over all DEV clips
    tot = {bk: {str(b): {"on_same_family_gold": 0, "on_needed": 0, "not_on_gold": 0} for b in BARS}
           for bk in ("a_folded", "b_outside", "all")}
    perq = {}
    for st in stems:
        fw, ql, fps = _load(out_dir("dev") / f"{st}.npz")
        dt = 1.0 / fps
        for k, q in enumerate(ql):
            col = fw[k]
            for i, j in _runs(col, dt, RUN_BAR, RUN_GAP):
                t0, t1, peak = i * dt, j * dt, float(col[i:j].max())
                hit = [g for g in gold[st] if S.same_family(q, g["label"]) and t1 > g["start"] - EARLY and t0 < g["end"]]
                need = any(g["needed"] for g in hit)
                for b in BARS:
                    if peak < b:
                        continue
                    key = "on_same_family_gold" if hit else "not_on_gold"
                    for bk in (qmeta[q]["bucket"], "all"):
                        tot[bk][str(b)][key] += 1
                        if need:
                            tot[bk][str(b)]["on_needed"] += 1
                    pq = perq.setdefault(q, {str(x): [0, 0] for x in BARS})
                    pq[str(b)][0 if hit else 1] += 1
                    if b == BARS[0] and hit:
                        res.setdefault("runs_on_gold_examples", []).append(
                            {"clip": st, "query": q, "run": [round(t0, 2), round(t1, 2)], "peak": round(peak, 3),
                             "gold": [(g["label"], g["start"], g["needed"]) for g in hit]})
    res["runs"] = tot
    res["per_query"] = {q: {"bucket": qmeta[q]["bucket"], **v} for q, v in sorted(perq.items(), key=lambda kv: -sum(kv[1]["0.4"]))}
    res["per_query_note"] = "per bar: [runs on same-family gold, runs not on gold]"
    OUTJ.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for u in res["unheard"]:
        print(u["clip"], u["label"], u["onset"], "new same-family", u["new_same_family"], "top", u["new_top5"][:3],
              "| existing", u["existing_nearest"], u["existing_top3"])
    for bk, v in tot.items():
        print(bk, v)
    print("->", OUTJ)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["queries", "run", "screen"])
    ap.add_argument("--set", default="dev", choices=["dev", "test"])
    ap.add_argument("--batch", type=int, default=24)
    a = ap.parse_args()
    {"queries": build_queries, "run": lambda: run(a.set, a.batch), "screen": screen}[a.mode]()
