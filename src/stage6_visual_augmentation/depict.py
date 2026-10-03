"""Round 57 DEPICT-EVENT (design review and shipped, 1 Oct 2026).

A drawn picture is dropped when (a) its depicted event (the spec's `subject`) is visibly happening at the picture start
+-1 s (VLM: E1 "yes" AND the opposite twin E2 "no") AND (b) text only, the visibility gate's own named maker B "could be
mistaken for" the picture's sound A in both orders. The viewer already sees the thing that makes this sound. Prompts,
frames and model are those of benchmark/gold/depict_screen.py (release v1.2.0; the DEV screen: 29/58, 17, 2.113 vs 29/58, 18, 2.141).
Honest reading: the event question said "no" on 42 of 43 DEV pictures; the rule rarely fires.

Answers are asked once per clip by a separate step (one model on the GPU at a time) and cached; `_display_spans` reads the
cache by clip, so the scorer and the renderer see the same pictures. A clip without cached answers keeps every picture.

    python -m src.stage6_visual_augmentation.depict <arm work root> [cache.json]
"""
from __future__ import annotations

import json
from pathlib import Path

import config

E1 = 'Is "{event}" visibly happening in these frames? Answer yes or no.'
E2 = 'Do these frames show no "{event}" happening? Answer yes or no.'
L1 = "Could the sound of {B} be mistaken for the sound of {A}? Answer yes or no."
L2 = "Could the sound of {A} be mistaken for the sound of {B}? Answer yes or no."
_CACHE = {}


def key(label: str, start: float) -> str:
    return f"{label}|{float(start):.2f}"


def yn(reply: str):
    t = reply.strip().lower().lstrip("*\"'([ ").split()
    w = t[0].strip(".,!*\"')]") if t else ""
    return w if w in ("yes", "no") else None


def _path():
    return Path(getattr(config, "DEPICT_CACHE", None) or (Path(config.WORK_DIR) / "depict_answers.json"))


def silenced(clip: str):
    """{label|start} of the clip's specs the rule drops (empty when the clip was never asked)"""
    p = _path()
    if str(p) not in _CACHE:
        _CACHE[str(p)] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return {k for k, v in _CACHE[str(p)].get(clip or "", {}).items() if v.get("silenced")}


def filter_specs(specs, clip: str):
    if not getattr(config, "DEPICT_EVENT", False):
        return specs
    drop = silenced(clip or getattr(config, "GROUP_CLIP", None))
    _trail(specs, clip or getattr(config, "GROUP_CLIP", None), drop)
    return [s for s in specs if key(s.event_label, s.start) not in drop] if drop else specs


def _trail(specs, clip, drop):
    """Decision Inspector (src/trail.py): the cached DEPICT-EVENT answers of each drawn spec (logging only)"""
    try:
        from src import trail as _T
        recs = _CACHE.get(str(_path()), {}).get(clip or "", {})
        for s in specs:
            if not s.augment:
                continue
            r = recs.get(key(s.event_label, s.start))
            if r is None:
                _T.decide("depict_event", s, "pass", note="no cached DEPICT answers for this picture: kept")
                continue
            subj, mk = r.get("subject", ""), r.get("makers") or []
            asks = []
            rep = r.get("event_replies") or []
            if rep:
                asks.append({"who": "Qwen3.8-27B on 6 frames, start -1 .. +1 s", "q": E1.format(event=subj), "a": rep[0],
                             "vote": str(yn(rep[0]))})
                if len(rep) > 1:
                    asks.append({"who": "Qwen3.8-27B (twin)", "q": E2.format(event=subj), "a": rep[1], "vote": str(yn(rep[1]))})
            for B in mk:
                for q in (L1, L2):
                    asks.append({"who": "Qwen3.8-27B (text)", "q": q.format(B=B, A=s.event_label.lower()),
                                 "a": "(reply not stored; any maker answering yes twice: " + str(bool(r.get("lookalike_yes"))) + ")",
                                 "vote": "-"})
            gone = key(s.event_label, s.start) in drop
            _T.decide("depict_event", s, "drop" if gone else "pass", asks=asks,
                      value="event visibly happening: " + str(bool(r.get("event"))) + "; gate-named makers: "
                            + (", ".join(mk) or "none") + "; a maker could sound like it: " + str(bool(r.get("lookalike_yes"))),
                      bar="dropped iff E1 yes AND E2 no, AND a gate-named maker answers yes in both look-alike orders",
                      note="picture subject: " + subj + ("" if mk else " (no maker named by the gate: not asked)"))
    except Exception:
        pass


def ask_clip(reason, mdl, proc, d: Path):
    """answers for every drawn spec of one rendered clip folder (augmentations.json + gate_votes.json + media.json)"""
    from src.stage2_video_understanding import _sample_frames_at
    sp = json.loads((d / "augmentations.json").read_text(encoding="utf-8"))
    vf = d / "gate_votes.json"
    votes = json.loads(vf.read_text(encoding="utf-8")) if vf.exists() else []
    vp = Path(json.loads((d / "media.json").read_text(encoding="utf-8"))["video_path"])
    out = {}
    for s in sp:
        if not (s.get("augment") and s.get("image_path")):
            continue
        lab, a0 = s["event_label"], float(s["start"])
        subj = (s.get("subject") or s.get("image_prompt") or "").strip()
        bs = []
        for v in votes:
            if v["label"] == lab and abs(float(v["start"]) - a0) < 0.011:
                n = str(v.get("named") or "").strip()
                if n and n.lower() != "nothing" and n not in bs:
                    bs.append(n)
        rec = {"subject": subj, "makers": bs, "event": False, "lookalike_yes": False}
        if subj and bs:                                   # both halves are needed; skip the VLM when (b) cannot hold
            fr = _sample_frames_at(vp, [a0 - 1.0 + 2.0 * t / 5 for t in range(6)])
            if len(fr) >= 2:
                r1 = reason._ask(mdl, proc, E1.format(event=subj), images=fr, max_new=4)
                r2 = reason._ask(mdl, proc, E2.format(event=subj), images=fr, max_new=4)
                rec["event_replies"] = [r1, r2]
                rec["event"] = [yn(r1), yn(r2)] == ["yes", "no"]
            if rec["event"]:
                A = lab.lower()
                for B in bs:
                    x1 = reason._ask(mdl, proc, L1.format(B=B, A=A), images=None, max_new=4)
                    x2 = reason._ask(mdl, proc, L2.format(B=B, A=A), images=None, max_new=4)
                    if yn(x1) == "yes" and yn(x2) == "yes":
                        rec["lookalike_yes"] = True
                        break
        rec["silenced"] = bool(rec["event"] and rec["lookalike_yes"])
        out[key(lab, a0)] = rec
    return out


def ensure_subprocess(clip: str, work: Path) -> None:
    """pipeline.run: ask for this clip in a fresh process (only the VLM on the GPU there) unless already cached"""
    import subprocess
    import sys
    if not getattr(config, "DEPICT_EVENT", False):
        return
    p = _path()
    if p.exists() and clip in json.loads(p.read_text(encoding="utf-8")):
        return
    code = ("import config, json; config.use_shipped(); config.DEPICT_CACHE = %r; from pathlib import Path; "
            "from src.stage6_visual_augmentation import depict as D; from src.stage5_cross_modal_analysis import reason; "
            "mdl, proc = reason._load(config.VLM_MODEL, 'cuda'); p = Path(%r); "
            "allc = json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}; "
            "allc[%r] = D.ask_clip(reason, mdl, proc, Path(%r)); p.parent.mkdir(parents=True, exist_ok=True); "
            "p.write_text(json.dumps(allc, indent=1), encoding='utf-8')" % (str(p), str(p), clip, str(work)))
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(Path(__file__).resolve().parents[2]))
    _CACHE.pop(str(p), None)


def main(argv=None):
    import sys
    a = list(sys.argv[1:] if argv is None else argv)
    root = Path(a[0])
    config.use_shipped()
    if len(a) > 1:
        config.DEPICT_CACHE = a[1]
    p = _path()
    allc = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    todo = [d for d in sorted(root.iterdir()) if (d / "augmentations.json").exists() and d.name not in allc]
    if not todo:
        return
    from src.stage5_cross_modal_analysis import reason
    mdl, proc = reason._load(getattr(config, "VLM_MODEL", "Qwen/Qwen3.8-27B"), "cuda")
    for d in todo:
        allc[d.name] = ask_clip(reason, mdl, proc, d)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(allc, indent=1), encoding="utf-8")
        print("depict", d.name, sorted(k for k, v in allc[d.name].items() if v["silenced"]), flush=True)


if __name__ == "__main__":
    main()
