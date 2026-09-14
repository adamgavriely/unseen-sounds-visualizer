"""Set the gate's silence rule once, on the development split, with the rubric's costs.

Why. Under the evaluation rubric a picture withheld where one was due scores 0 where a
picture would have scored up to 4; a picture shown where none was due scores about 3
("conveys most of it, minor omission or vagueness") where silence scores 4. So a wrong
silence costs four points and a redundant picture one, and a gate tuned for symmetric
accuracy is cautious in the direction that costs the most. On the v3 run (2026-09-14) the
gated system loses to the blind baseline on "picture needed" clips by -0.24 and half of
those losses are wrong silences (church bell, siren over a crowd, traffic).

What. The gate's visibility verdict is three VLM votes per sound per 5 s stretch
(reason._sound_is_visible): open naming + world-knowledge check, an a/b question in both
orderings, a description + world-knowledge check. Majority = visible; silent only if
visible in every stretch; kinds of a visible source are silenced too; display bar 0.35.
This script caches the raw votes for every labelled clip (--cache), then sweeps a small
grid on the DEV clips only (--sweep):

    silence rule     majority of 3 (current) | unanimous 3 of 3
    display bar      0.25 | 0.30 | 0.35 | 0.40
    kinds rule       on (current) | off

scoring each cell by the rubric-derived cost (missed picture -4, redundant picture -1)
and by plain accuracy, both reported. The chosen cell is written to
benchmark/gate_setting.json with a timestamp BEFORE the test clips' cache is opened
(--apply), which then lists the test clips whose clip-level decision flips.

Dev = every clip with a human tag that is not one of the 100 protocol test clips
(174 clips: 107 seen, 44 no_ambient, 23 unseen; mixed clips are all in the test set).

    python -m benchmark.gate_dev_sweep --cache --split dev      # GPU, ~3 h
    python -m benchmark.gate_dev_sweep --cache --split test     # GPU, ~2 h, in parallel
    python -m benchmark.gate_dev_sweep --sweep                  # CPU, prints the grid
    python -m benchmark.gate_dev_sweep --apply                  # CPU, flips on test
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.labels import consolidate_families, is_salient_nonspeech, min_confidence, same_source

TAGS = _ROOT / "benchmark" / "tags.json"
TEST_DESC = _ROOT / "benchmark" / "protocol_descriptions_v3.json"
CACHE_DIR = _ROOT / "benchmark" / "gate_votes"
SETTING = _ROOT / "benchmark" / "gate_setting.json"
CANDIDATE_BAR = 0.25             # votes are collected for every sound at or above the lowest bar swept
BARS = (0.25, 0.30, 0.35, 0.40)
RULES = ("majority", "unanimous")
SHOULD_SHOW = {"unseen_ambient", "mixed"}
SILENT_RIGHT = {"seen_ambient", "no_ambient"}
COST_MISS = 4.0                  # rubric: withheld picture scores 0 where one would score up to 4
COST_REDUNDANT = 1.0             # rubric: a redundant picture scores ~3 ("minor omission") where silence scores 4


def _find_clip(basename: str):
    for p in (_ROOT / "data" / "input" / "benchmark").rglob(basename):
        return p
    return None


def clip_sets():
    tags = json.loads(TAGS.read_text(encoding="utf-8"))
    test = {r["clip"] for r in json.loads(TEST_DESC.read_text(encoding="utf-8"))}
    labelled = {k.split("/", 1)[1]: v["tag"] for k, v in tags.items()
                if v.get("tag") in SHOULD_SHOW | SILENT_RIGHT}
    dev = {b: t for b, t in labelled.items() if b not in test}
    tst = {b: t for b, t in labelled.items() if b in test}
    return dev, tst


# ----------------------------------------------------------------------------- cache
def cache_votes(split: str):
    from src.stage1_audio_extraction import extract_audio
    from src.stage4_audio_event_detection import detect_events
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason as R
    dev, tst = clip_sets()
    clips = dev if split == "dev" else tst
    out_dir = CACHE_DIR / split
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [b for b in sorted(clips) if not (out_dir / (b + ".json")).exists()]
    print(f"[votes] {split}: {len(clips)} clips, {len(todo)} to do", flush=True)
    if not todo:
        return
    mdl, proc = R._load(config.VLM_MODEL, config.DEVICE)
    STRETCH = float(getattr(config, "VISIBILITY_STRETCH", 5.0))
    for i, base in enumerate(todo, 1):
        video = _find_clip(base)
        if video is None:
            print(f"[votes] missing clip {base}", flush=True)
            continue
        try:
            with tempfile.TemporaryDirectory() as td:
                media = extract_audio(video, Path(td) / "a.wav", config.SAMPLE_RATE)
                events = detect_events(Path(media.wav_path), threshold=config.AED_THRESHOLD,
                                       min_dur=config.AED_MIN_DUR, model=config.AED_MODEL,
                                       device=config.DEVICE)
            cands = consolidate_families([e for e in events if is_salient_nonspeech(e.label)],
                                         threshold=min_confidence("", CANDIDATE_BAR))
            rec = {"clip": base, "tag": clips[base], "duration": media.duration,
                   "events": [e.to_dict() for e in events], "sounds": []}
            for ev in cands:
                if ev.confidence < min_confidence(ev.label, CANDIDATE_BAR):
                    continue
                bursts = list(getattr(ev, "spans", None) or [(ev.start, ev.end)])
                pieces = []
                for a, b in bursts:
                    k = max(1, int(round((b - a) / STRETCH)))
                    edges = [a + (b - a) * j / k for j in range(k + 1)]
                    pieces += list(zip(edges, edges[1:]))
                stretches = []
                for a, b in pieces:
                    n = 6
                    lo, hi = a - 1.0, b + 1.0
                    times = [lo + (hi - lo) * t / (n - 1) for t in range(n)]
                    win = _sample_frames_at(video, times)
                    seen, named = R._sound_is_visible(ev.label, win, mdl, proc, config.DEVICE)
                    v = dict(R.LAST_VOTES)
                    stretches.append({"start": a, "end": b, "verdict": bool(seen),
                                      "name": v.get("name"), "ab": v.get("ab"),
                                      "desc": v.get("desc"), "named": v.get("named", "")})
                rec["sounds"].append({"label": ev.label, "confidence": ev.confidence,
                                      "start": ev.start, "end": ev.end, "stretches": stretches})
            (out_dir / (base + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"[votes] {i}/{len(todo)} {base}: {len(rec['sounds'])} sound(s)", flush=True)
        except Exception as e:
            print(f"[votes] ! {base}: {type(e).__name__}: {e}", flush=True)
    R.unload()


# ----------------------------------------------------------------------------- decide
def _visible(stretch: dict, rule: str) -> bool:
    votes = [stretch.get("name"), stretch.get("ab"), stretch.get("desc")]
    yes = sum(1 for v in votes if v is True)
    no = sum(1 for v in votes if v is False)
    if rule == "unanimous":
        return yes == 3
    return yes > no


def decide(rec: dict, bar: float, rule: str, kinds: bool) -> dict:
    """Clip-level decision from cached votes: which sounds would be shown."""
    live = [s for s in rec["sounds"] if s["confidence"] >= min_confidence(s["label"], bar)]
    silenced, shown = [], []
    for s in live:
        if s["stretches"] and all(_visible(st, rule) for st in s["stretches"]):
            silenced.append(s["label"])
        else:
            shown.append(s["label"])
    if kinds and silenced:
        shown = [l for l in shown if not any(same_source(l, g) for g in silenced)]
    return {"show": bool(shown), "shown": shown, "silenced": silenced}


def score(recs, bar, rule, kinds):
    cost = 0.0; correct = 0; miss = 0; redundant = 0
    for rec in recs:
        d = decide(rec, bar, rule, kinds)
        due = rec["tag"] in SHOULD_SHOW
        if d["show"] == due:
            correct += 1
        elif due:
            miss += 1; cost -= COST_MISS
        else:
            redundant += 1; cost -= COST_REDUNDANT
    n = len(recs)
    return {"bar": bar, "rule": rule, "kinds": kinds, "n": n, "accuracy": correct / n,
            "cost_per_clip": cost / n, "missed": miss, "redundant": redundant}


def load(split: str):
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted((CACHE_DIR / split).glob("*.json"))]


def sweep():
    recs = load("dev")
    print(f"[sweep] {len(recs)} dev clips with cached votes")
    rows = [score(recs, b, r, k) for k in (True, False) for r in RULES for b in BARS]
    print(f"{'bar':>5} {'rule':10s} {'kinds':5s} {'acc':>6} {'cost/clip':>9} {'missed':>6} {'redund':>6}")
    for r in rows:
        print(f"{r['bar']:>5.2f} {r['rule']:10s} {str(r['kinds']):5s} {r['accuracy']:6.1%} "
              f"{r['cost_per_clip']:9.3f} {r['missed']:6d} {r['redundant']:6d}")
    current = next(r for r in rows if r["bar"] == 0.35 and r["rule"] == "majority" and r["kinds"])
    best = max(rows, key=lambda r: (r["cost_per_clip"], r["accuracy"]))
    best_acc = max(rows, key=lambda r: (r["accuracy"], r["cost_per_clip"]))
    print(f"[sweep] current  : {current}")
    print(f"[sweep] best cost: {best}")
    print(f"[sweep] best acc : {best_acc}")
    SETTING.write_text(json.dumps({"chosen": best, "current": current, "best_accuracy": best_acc,
                                   "grid": rows, "objective": "cost_per_clip (miss -4, redundant -1)",
                                   "frozen_at": datetime.now().isoformat(timespec="seconds"),
                                   "dev_clips": len(recs)}, indent=1), encoding="utf-8")
    print(f"[sweep] frozen -> {SETTING}")


def apply():
    setting = json.loads(SETTING.read_text(encoding="utf-8"))
    ch, cur = setting["chosen"], setting["current"]
    recs = load("test")
    flips = []
    for rec in recs:
        a = decide(rec, cur["bar"], cur["rule"], cur["kinds"])
        b = decide(rec, ch["bar"], ch["rule"], ch["kinds"])
        if a["show"] != b["show"] or set(a["shown"]) != set(b["shown"]):
            flips.append({"clip": rec["clip"], "tag": rec["tag"], "before": a, "after": b})
    print(f"[apply] {len(recs)} test clips, {len(flips)} change under the frozen setting "
          f"(bar {ch['bar']}, {ch['rule']}, kinds={ch['kinds']}):")
    for f in flips:
        print(f"   {f['clip']:45s} {f['tag']:15s} {f['before']['shown']} -> {f['after']['shown']}")
    out = _ROOT / "benchmark" / "gate_flips_test.json"
    out.write_text(json.dumps({"setting": ch, "flips": flips}, indent=1), encoding="utf-8")
    print(f"[apply] -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--split", choices=("dev", "test"), default="dev")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()
    if a.cache:
        cache_votes(a.split)
    if a.sweep:
        sweep()
    if a.apply:
        apply()


if __name__ == "__main__":
    main()
