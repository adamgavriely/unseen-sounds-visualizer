"""Decision trail: one record per (candidate span, decision step), for the Decision Inspector (docs/inspector2).

Every decision point of the shipped pipeline calls `decide(...)` with the span it judged, the result, the measured value,
the bar it was compared with, and, when a model was asked, the exact question and the exact answer. The trail is reset
per clip and written next to onset_trace.json as trail.json. Logging only: nothing here changes a decision, so the
scores must stay identical (parity check: benchmark/gold/inspector_trail_export.py --expect).

Step ids are those of docs/inspector2/sample_data.py STEPS (+ a few display-only ids, see
benchmark/gold/inspector_trail_export.py STEPS_EXTRA). Hook sites: docs/inspector2/HOOKS.md (branch
claude/project-thread-4rwrnp), put in on main at the detector freeze (tag detector-frozen-2026-10-02).

Keying: a record's span is (label, start, end) BEFORE the decision; a move / relabel / merge / rescue / pass with
`new_span` links it to the span it became (the exporter follows these links). A record with extra burst=... is about one
burst / picture of a stage-5 sound, not about the whole sound.

`decide` never raises: a logging error is recorded as a "skip" row with the error text, so a hook can never change what
the pipeline does.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

TRAIL: List[Dict[str, Any]] = []
RESULTS = {"pass", "drop", "move", "merge", "relabel", "rescue", "skip"}


def reset() -> None:
    TRAIL.clear()


def snapshot() -> List[Dict[str, Any]]:
    return json.loads(json.dumps(TRAIL, ensure_ascii=False))


def extend(records) -> None:
    TRAIL.extend(records or [])


def _num(v):
    try:
        return float(v)
    except Exception:
        return 0.0


def _span(e) -> Dict[str, Any]:
    if isinstance(e, (tuple, list)):
        label, a, b = e[0], e[1], e[2]
        out = {"label": str(label), "start": round(_num(a), 3), "end": round(_num(b), 3)}
        if len(e) > 3 and e[3] is not None:
            out["conf"] = round(_num(e[3]), 3)
        return out
    lab = getattr(e, "label", None)
    if lab is None:
        lab = getattr(e, "event_label", "")
    return {"label": str(lab), "start": round(_num(getattr(e, "start", 0.0)), 3), "end": round(_num(getattr(e, "end", 0.0)), 3),
            "conf": round(_num(getattr(e, "confidence", 0.0) or 0.0), 3)}


def _fmt(v: Any) -> Optional[str]:
    if v is None:
        return None
    if hasattr(v, "dtype") and getattr(v, "shape", ()) == ():
        v = v.item()
    if isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        return f"{v:.3f}".rstrip("0").rstrip(".") if v == v else "nan"
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
    try:
        if res not in RESULTS:
            raise ValueError(f"unknown result {res!r}")
        rec = {"step": step, "res": res, **_span(span)}
        for k, v in (("value", _fmt(value)), ("bar", _fmt(bar)), ("note", note or None)):
            if v is not None:
                rec[k] = v
        if asks:
            rec["asks"] = [{k: str(v) for k, v in a.items() if v is not None} for a in asks]
        if new_span is not None:
            rec["to"] = _span(new_span)
        if extra:
            rec["extra"] = {k: _fmt(v) for k, v in extra.items() if v is not None}
        TRAIL.append(rec)
    except Exception as ex:                                  # logging only: never let a hook change the run
        try:
            TRAIL.append({"step": str(step), "res": "skip", "label": "", "start": 0.0, "end": 0.0,
                          "note": f"trail error: {type(ex).__name__}: {ex}"})
        except Exception:
            pass


def dump(work_dir, records=None) -> Path:
    p = Path(work_dir) / "trail.json"
    p.write_text(json.dumps(TRAIL if records is None else records, ensure_ascii=False, indent=1), encoding="utf-8")
    return p
