"""Build the gold-set annotation tool: one self-contained HTML page (benchmark/gold/index.html)
that plays each test clip from the local benchmark folder and records, per sound, whether it
is heard, whether its source is visible while it sounds, whether speech or music covers it,
and its rough start/end; per clip, one human sentence "what a hearing viewer gets that a deaf
viewer misses". The detector's candidates are hidden until the annotator asks for them, so a
first free listening pass is recorded before any anchoring.

Output of an annotator: one JSON file (Export button) -> benchmark/gold/annotations/<name>.json.
Merge and agreement: benchmark/gold/merge.py.

    python benchmark/gold/build_tool.py                    # writes benchmark/gold/index.html
    python benchmark/gold/build_tool.py --slice audioset   # writes benchmark/gold/index_audioset.html
                                                           # (pre-fill from AudioSet-Strong human labels,
                                                           #  read from benchmark/gold/audioset_slice.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gate_dev_sweep import load, _find_clip, min_confidence, decide

HERE = Path(__file__).resolve().parent
DUE = {"unseen_ambient", "mixed"}
NOT_SOUNDS = {"Speech", "Music"}


def _detail(rec, label):
    """the most specific raw detection under this family, for a friendlier default name"""
    best = None
    for e in rec.get("events", []):
        from src.labels import canonical
        if canonical(e["label"]) == label and (best is None or e["confidence"] > best["confidence"]):
            best = e
    return best["label"] if best else label


def _masked(rec, start, end, conf):
    """the sound is quiet (below 0.5) while speech or music above 0.5 covers most of its span"""
    if conf >= 0.5:
        return False
    for e in rec.get("events", []):
        if e["label"] in ("Speech", "Music") and e["confidence"] >= 0.5 and min(e["end"], end) - max(e["start"], start) >= 0.5 * (end - start):
            return True
    return False


def build_default():
    """the 100 benchmark test clips, pre-filled from the detector, the gate and the proposed system's sentence"""
    gate = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
    results = {}
    for rec in json.loads((_ROOT / "benchmark" / "protocol_results_v3_grounded.json").read_text(encoding="utf-8")):
        if rec["system"] == "proposed":
            results[rec["clip"]] = rec
    clips = []
    for rec in load("test"):
        video = _find_clip(rec["clip"])
        if video is None:
            continue
        rel = Path(video).resolve().relative_to(_ROOT).as_posix()
        d = decide(rec, gate["bar"], gate["rule"], gate["kinds"])
        cands = []
        for s in rec["sounds"]:
            if s["confidence"] < min_confidence(s["label"], gate["bar"]):
                continue
            visible_all = bool(s["stretches"]) and all(st.get("verdict") for st in s["stretches"])
            cands.append({"label": _detail(rec, s["label"]).replace(" (siren)", " siren").lower(), "family": s["label"],
                          "conf": round(s["confidence"], 2), "start": round(s["start"], 1), "end": round(s["end"], 1),
                          "visible": visible_all, "masked": _masked(rec, s["start"], s["end"], s["confidence"]),
                          "gate": "silenced" if s["label"] in d["silenced"] else ("shown" if s["label"] in d["shown"] else "dropped")})
        cands.sort(key=lambda c: -c["conf"])
        pr = results.get(rec["clip"], {})
        ref = pr.get("reference", "")
        clips.append({"id": rec["clip"], "src": "../../" + rel, "duration": round(rec["duration"], 1), "candidates": cands[:8],
                      "tag": rec["tag"], "picture_due": rec["tag"] in DUE,
                      "sentence": ref if ref and ref != "nothing beyond the picture" else "nothing beyond the picture"})
    return clips


def build_audioset():
    """slice B: AudioSet-Strong clips, pre-filled from the human-timed labels (no detector, no sentence, picture-due open)"""
    src = HERE / "audioset_slice.json"
    if not src.exists():
        sys.exit(f"{src} not found — run benchmark/gold/audioset_slice.py first")
    clips, missing = [], 0
    for c in json.loads(src.read_text(encoding="utf-8"))["clips"]:
        if not (_ROOT / "data" / "input" / "audioset_strong" / f"{c['id']}.mp4").exists():
            missing += 1
            continue
        cands = [{"label": e["label"].lower(), "family": e["label"], "conf": 1.0, "start": round(e["start"], 1), "end": round(e["end"], 1),
                  "visible": False, "masked": bool(e.get("masked")), "gate": "human"}
                 for e in c.get("events", []) if e["label"] not in NOT_SOUNDS]
        clips.append({"id": c["id"], "src": c["src"], "duration": round(c["duration"], 1), "candidates": cands,
                      "tag": "audioset_strong", "picture_due": None, "sentence": ""})
    if missing:
        print(f"skipped {missing} clips whose mp4 is not under data/input/audioset_strong/")
    return clips


def write(clips, name):
    clips.sort(key=lambda c: c["id"])
    html = (HERE / "tool_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__CLIPS__*/[]", json.dumps(clips, ensure_ascii=False))
    (HERE / name).write_text(html, encoding="utf-8")
    print(f"-> {HERE / name}: {len(clips)} clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", choices=["audioset"], default=None, help="build the AudioSet-Strong page instead of the benchmark one")
    args = ap.parse_args()
    if args.slice == "audioset":
        write(build_audioset(), "index_audioset.html")
    else:
        write(build_default(), "index.html")
