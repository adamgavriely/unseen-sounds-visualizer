"""Decision trail: one record per (candidate span, decision step), for the Decision Inspector (docs/inspector2).

Every decision point of the shipped pipeline calls `decide(...)` with the span it judged, the result, the measured value,
the bar it was compared with, and, when a model was asked, the exact question and the exact answer. The trail is reset
per clip and written next to onset_trace.json as trail.json. Logging only: nothing here changes a decision, so the
scores must stay identical (benchmark/gold/parity_check.py).

Hook sites (step ids match docs/inspector2/sample_data.py STEPS; file:line as of main 2 Oct 2026, see
docs/inspector2/HOOKS.md): stage 4 beats_extract ... rescue_once, stage 5 label_filter ... dedup, stage 6 depict_event ...
max_slots.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TRAIL: List[Dict[str, Any]] = []
RESULTS = {"pass", "drop", "move", "merge", "relabel", "rescue", "skip"}


def reset() -> None:
    TRAIL.clear()


def _span(e) -> Dict[str, Any]:
    if isinstance(e, (tuple, list)):
        label, a, b = e[0], e[1], e[2]
        return {"label": str(label), "start": round(float(a), 3), "end": round(float(b), 3)}
    return {"label": str(getattr(e, "label", getattr(e, "event_label", ""))),
            "start": round(float(getattr(e, "start", 0.0)), 3), "end": round(float(getattr(e, "end", 0.0)), 3),
            "conf": round(float(getattr(e, "confidence", 0.0) or 0.0), 3)}


def _fmt(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".")
    return str(v)


def decide(step: str, span, res: str, value: Any = None, bar: Any = None, note: str = "",
           asks: Optional[List[Dict[str, str]]] = None, new_span=None, **extra) -> None:
    """Record one decision.

    step   : step id (e.g. "dasm_local_veto")
    span   : the AudioEvent / AugmentationSpec / (label, start, end) judged, BEFORE the decision
    res    : pass | drop | move | merge | relabel | rescue | skip
    value  : what was measured, e.g. 0.21 or "DASM 0.21; Qwen names it, AF does not"
    bar    : what it was compared with, e.g. ">= 0.35"
    asks   : [{"who": "Qwen3-Omni V4 on 3.2-4.4 s", "q": <exact prompt>, "a": <exact reply>, "vote": "yes|no|seen|..."}]
    new_span: the span after a move / merge / relabel
    """
    if res not in RESULTS:
        raise ValueError(f"trail.decide: unknown result {res!r}")
    rec = {"step": step, "res": res, **_span(span)}
    for k, v in (("value", _fmt(value)), ("bar", _fmt(bar)), ("note", note or None)):
        if v is not None:
            rec[k] = v
    if asks:
        rec["asks"] = [{k: str(v) for k, v in a.items() if v is not None} for a in asks]
    if new_span is not None:
        rec["to"] = _span(new_span)
    if extra:
        rec["extra"] = {k: _fmt(v) for k, v in extra.items()}
    TRAIL.append(rec)


def dump(work_dir) -> Path:
    p = Path(work_dir) / "trail.json"
    p.write_text(json.dumps(TRAIL, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
