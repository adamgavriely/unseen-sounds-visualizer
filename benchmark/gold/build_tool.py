"""Build the gold-set annotation tool: one self-contained HTML page (benchmark/gold/index.html)
that plays each test clip from the local benchmark folder and records, per sound, whether it
is heard, whether its source is visible while it sounds, whether speech or music covers it,
and its rough start/end; per clip, one human sentence "what a hearing viewer gets that a deaf
viewer misses". The detector's candidates are hidden until the annotator asks for them, so a
first free listening pass is recorded before any anchoring.

Output of an annotator: one JSON file (Export button) -> benchmark/gold/annotations/<name>.json.
Merge and agreement: benchmark/gold/merge.py.

    python benchmark/gold/build_tool.py            # writes benchmark/gold/index.html
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gate_dev_sweep import load, _find_clip, min_confidence

HERE = Path(__file__).resolve().parent
gate = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]

clips = []
for rec in load("test"):
    video = _find_clip(rec["clip"])
    if video is None:
        continue
    rel = Path(video).resolve().relative_to(_ROOT).as_posix()
    cands = [{"label": s["label"], "conf": round(s["confidence"], 2), "start": round(s["start"], 1), "end": round(s["end"], 1)}
             for s in rec["sounds"] if s["confidence"] >= min_confidence(s["label"], gate["bar"])]
    cands.sort(key=lambda c: -c["conf"])
    clips.append({"id": rec["clip"], "src": "../../" + rel, "duration": round(rec["duration"], 1), "candidates": cands[:8]})
clips.sort(key=lambda c: c["id"])

html = (HERE / "tool_template.html").read_text(encoding="utf-8")
html = html.replace("/*__CLIPS__*/[]", json.dumps(clips, ensure_ascii=False))
(HERE / "index.html").write_text(html, encoding="utf-8")
print(f"-> {HERE / 'index.html'}: {len(clips)} clips")
