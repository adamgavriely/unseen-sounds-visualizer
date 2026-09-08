"""Choose demo clips where the augmentation is VISIBLY SELECTIVE.

Adam's requirement: a clip whose augmented sound runs the whole duration produces a
panel that is on from first frame to last, which looks like an image was appended to
the video rather than like a system deciding moment by moment. A convincing demo has
the panel appear, change, and go away.

So clips are ranked by how dynamic their augmentation timeline is:

  coverage      fraction of the clip during which ANY augmentation is on screen.
                Wanted in a middle band -- too low and nothing happens, too high and
                it is a static overlay. Best around 0.25-0.65.
  transitions   how many times the panel changes (appears, swaps source, clears).
                More transitions means the selectivity is visible to the eye.
  n_sources     distinct sounds augmented; two or three shows the slot layout working.

Clips are read from the evaluation cache, which already holds each clip's detected
events, their time spans and the visibility decision, so nothing needs re-running.

Usage:
    python -m benchmark.select_demo              # print the ranking
    python -m benchmark.select_demo --write 12   # write benchmark/demo_set.json
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
OUT = _ROOT / "benchmark" / "demo_set.json"
# a panel on for less than this is invisible; more than this is a static overlay
COVERAGE_LO, COVERAGE_HI = 0.20, 0.70
IDEAL = 0.45


def load_cache():
    for name in ("eval_cache_siglip.json", "eval_cache.json"):
        f = _ROOT / "benchmark" / name
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8"))
    return {}


def timeline(entry):
    """Augmented spans for a clip, as the compositor would build them."""
    evs = [AudioEvent(*e) for e in entry["events"]]
    salient = [e for e in consolidate_families(
                   [x for x in evs if is_salient_nonspeech(x.label)])
               if e.confidence >= min_confidence(e.label, config.DISPLAY_THRESHOLD)]
    visible = {v.lower() for v in entry.get("visible", [])}
    return [e for e in salient if e.label.lower() not in visible]


def dynamics(entry):
    """(coverage, transitions, n_sources) for the augmentation timeline."""
    dur = float(entry.get("duration") or 0) or 1.0
    offs = timeline(entry)
    if not offs:
        return 0.0, 0, 0
    # union of the augmented spans, clipped to the clip
    spans = sorted((max(0.0, e.start), min(dur, e.end)) for e in offs if e.end > e.start)
    merged = []
    for s, t in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], t))
        else:
            merged.append((s, t))
    covered = sum(t - s for s, t in merged)
    # a transition is each edge of a covered block that falls inside the clip
    transitions = sum((s > 0.3) + (t < dur - 0.3) for s, t in merged)
    return covered / dur, transitions, len({e.label for e in offs})


def score(cov, trans, nsrc):
    """Prefer mid-range coverage, then visible change, then a couple of sources."""
    if cov <= 0 or cov > 0.95:
        return -1.0
    band = 1.0 - abs(cov - IDEAL) / IDEAL          # 1.0 at IDEAL, falls off either side
    in_band = 1.0 if COVERAGE_LO <= cov <= COVERAGE_HI else 0.35
    return band * in_band + 0.25 * min(trans, 4) + 0.30 * min(nsrc, 3)


def main():
    write_n = None
    if "--write" in sys.argv:
        i = sys.argv.index("--write")
        write_n = int(sys.argv[i + 1]) if i + 1 < len(sys.argv) else 12

    cache = load_cache()
    if not cache:
        sys.exit("no eval cache found -- run benchmark.evaluate first")
    tags = json.loads(TAGS.read_text(encoding="utf-8")) if TAGS.exists() else {}
    tag_of = {k.split("/", 1)[1]: v["tag"] for k, v in tags.items()}
    folder_of = {k.split("/", 1)[1]: k.split("/", 1)[0] for k in tags}

    rows = []
    for base, entry in cache.items():
        tag = tag_of.get(base)
        if tag not in ("unseen_ambient", "mixed"):
            continue                     # the demo should show it augmenting something
        cov, trans, nsrc = dynamics(entry)
        s = score(cov, trans, nsrc)
        if s > 0:
            rows.append((s, base, tag, cov, trans, nsrc,
                         f"{folder_of.get(base, 'unsorted')}/{base}"))
    rows.sort(reverse=True)

    print(f"{'clip':38}{'tag':16}{'cover':>7}{'trans':>7}{'srcs':>6}{'score':>7}")
    for s, base, tag, cov, trans, nsrc, _ in rows[:20]:
        print(f"{base[:38]:38}{tag:16}{cov:>6.0%}{trans:>7}{nsrc:>6}{s:>7.2f}")
    print(f"\n{len(rows)} clips augment something without covering the whole duration")

    if write_n:
        chosen = [r[6] for r in rows[:write_n]]
        OUT.write_text(json.dumps(chosen, indent=2), encoding="utf-8")
        print(f"wrote {len(chosen)} clips -> {OUT}")


if __name__ == "__main__":
    main()
