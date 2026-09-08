"""Threshold-free evaluation of the visibility signal, plus the trivial baselines.

Why this exists. Reported gating accuracy (49.3%) mixes two very different failures:
"the model carries no signal" and "the model carries signal but the cut-off is in the
wrong place". Worse, the visibility threshold had never been swept at all, because the
evaluation cached the post-threshold list of visible entities rather than the scores.

This reports, on the same clips and the same human labels:

  1. TRIVIAL BASELINES -- always-augment (which IS the blind audio-to-image baseline
     the thesis claims to beat) and always-silent. A gate that cannot beat these has
     not demonstrated the contribution, and this table belongs in the thesis stated
     by us rather than discovered by an examiner.
  2. AUROC of the visibility score at the EVENT level: given a sound event whose
     source a human says is visible, and one they say is not, how often does the
     score rank them correctly? 0.5 is chance, and it is independent of any cut-off.
  3. A sweep of the visibility threshold, which the previous evaluation could not do.

Event-level ground truth is derived from the clip tag: in a seen_ambient clip every
detected source is visible; in unseen_ambient none are. "mixed" clips are excluded
here precisely because the clip label does not say WHICH source was visible -- an
honest limitation of clip-level labelling, and one worth stating.

Usage: python -m benchmark.auroc
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import config
from src.labels import is_salient_nonspeech, consolidate_families, min_confidence
from src.types import AudioEvent

TAGS = _ROOT / "benchmark" / "tags.json"


def auroc(pos, neg):
    """P(a random positive scores above a random negative); ties count as half."""
    if not pos or not neg:
        return float("nan"), 0, 0
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg)), len(pos), len(neg)


def load(backend):
    sfx = "" if backend == "clip" else f"_{backend}"
    f = _ROOT / "benchmark" / f"eval_cache{sfx}.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}


def main():
    tags = json.loads(TAGS.read_text(encoding="utf-8"))
    tag_of = {k.split("/", 1)[1]: v["tag"] for k, v in tags.items()}

    # ---------- 1. trivial baselines -------------------------------------------
    POS = {"unseen_ambient", "mixed"}
    NEG = {"seen_ambient", "no_ambient"}
    scored = [t for t in tag_of.values() if t in POS | NEG]
    n, pos = len(scored), sum(1 for t in scored if t in POS)
    print(f"benchmark: {n} labelled clips  ({pos} positive / {n-pos} negative)\n")
    print("TRIVIAL BASELINES (no model at all)")
    f1_aug = 2 * (pos / n) / ((pos / n) + 1)
    print(f"  always augment (= blind audio-to-image)  acc {pos/n:.1%}  F1 {f1_aug:.1%}")
    print(f"  always stay silent                       acc {(n-pos)/n:.1%}  F1  0.0%")
    print("  -> any gate must beat BOTH of these to have demonstrated anything\n")

    # ---------- 2. event-level AUROC of the visibility score --------------------
    for backend in ("clip", "siglip", "owlv2", "vlm"):
        cache = load(backend)
        if not cache:
            continue
        pos_s, neg_s, no_scores = [], [], 0
        for base, entry in cache.items():
            tag = tag_of.get(base)
            if tag not in ("seen_ambient", "unseen_ambient"):
                continue                      # mixed: clip label doesn't say which source
            scores = entry.get("vis_scores") or {}
            if not scores:
                no_scores += 1
                continue
            evs = [AudioEvent(*e) for e in entry["events"]]
            salient = [e for e in consolidate_families(
                           [x for x in evs if is_salient_nonspeech(x.label)])
                       if e.confidence >= min_confidence(e.label, config.DISPLAY_THRESHOLD)]
            for e in salient:
                s = scores.get(e.label)
                if s is None:
                    continue
                (pos_s if tag == "seen_ambient" else neg_s).append(s)
        a, np_, nn = auroc(pos_s, neg_s)
        print(f"{backend.upper():8} visibility AUROC = {a:.3f}   "
              f"({np_} visible events vs {nn} off-screen events)"
              + (f"   [{no_scores} clips cached without scores]" if no_scores else ""))
        if a == a:                            # not NaN
            verdict = ("signal is usable -- the cut-off is the problem" if a >= 0.65 else
                       "weak signal" if a >= 0.55 else
                       "no usable signal: off-the-shelf grounding cannot answer this")
            print(f"{'':8} -> {verdict}")

    if not any(load(b) for b in ("clip", "siglip", "owlv2", "vlm")):
        print("no eval caches found -- run benchmark.evaluate first")


if __name__ == "__main__":
    main()
