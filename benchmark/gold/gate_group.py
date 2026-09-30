"""Round 31 gate group (docs/prereg_round13_detector_push.md): trace every VISIBLE wrong picture and every gate-silenced
needed sound of the shipped SHIP5 (= SHIP4+BTP) pictures on merged DEV, with the stage-5 votes the pipeline stored in
gate_votes.json, and save a 6-frame contact sheet of what the gate saw (same frames: stretch +- 1 s). CPU only, nothing
in src/ or config.py is edited.

    python benchmark/gold/gate_group.py [--arm SHIP4+BTP] [--frames DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import btp_screen as B
from benchmark.gold import score_per_sound as S
from benchmark.gold import round13_dev as R
from benchmark.gold.detector_dry import clip_path

OUT = _ROOT / "benchmark" / "gold" / "gate_group.json"


def video_of(part, stem, work_root):
    m = work_root / stem / "media.json"
    if m.exists():
        d = json.loads(m.read_text(encoding="utf-8"))
        for k in ("video", "path", "input", "source"):
            if d.get(k) and Path(d[k]).exists():
                return Path(d[k])
    if part == "dev":
        for ext in (".mp4", ".webm", ".mkv", ".mov"):
            p = clip_path(stem + ext)
            if p:
                return p
    for d in (_ROOT / "data" / "input" / "tagger_dev2", _ROOT / "data" / "input" / "tagger_set"):
        for p in d.glob(stem + ".*"):
            return p
    return None


def votes_for(work_root, stem, lab, a, b):
    f = work_root / stem / "gate_votes.json"
    if not f.exists():
        return None
    rows = json.loads(f.read_text(encoding="utf-8"))
    out = [r for r in rows if S.same_family(r["label"], lab) and r["start"] <= b + 0.01 and r["end"] >= a - 0.01]
    return [{k: r.get(k) for k in ("label", "start", "end", "confidence", "stretch", "seen", "name", "ab", "desc", "named")} for r in out]


def specs_for(work_root, stem, lab, a, b):
    f = work_root / stem / "augmentations.json"
    if not f.exists():
        return []
    rows = json.loads(f.read_text(encoding="utf-8"))
    def near(r):                                          # any burst (span) of the spec touches [a, b] (advisor: not only start/end)
        sp = r.get("spans") or [[r["start"], r["end"]]]
        return any(x <= b + 1.0 and y >= a - 1.0 for x, y in sp)
    return [{k: r.get(k) for k in ("event_label", "start", "end", "confidence", "augment", "rescued", "reason", "spans")}
            for r in rows if S.same_family(r["event_label"], lab) and near(r)]


def sheet(video, a, b, out_png):
    from src.stage2_video_understanding import _sample_frames_at
    from PIL import Image
    n = 6; lo, hi = a - 1.0, b + 1.0
    times = [max(0.0, lo + (hi - lo) * t / (n - 1)) for t in range(n)]
    ims = _sample_frames_at(video, times)
    if not ims:
        return False
    w = 320; ims = [im.resize((w, int(im.height * w / im.width))) for im in ims]
    h = max(im.height for im in ims)
    canvas = Image.new("RGB", (w * 3, h * 2), "black")
    for i, im in enumerate(ims):
        canvas.paste(im, ((i % 3) * w, (i // 3) * h))
    canvas.save(out_png, quality=80)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="SHIP4+BTP")
    ap.add_argument("--frames", default=None)
    a = ap.parse_args()
    B.ARM = a.arm
    dev_root = R.R13 / f"{a.arm}_{B.SYS}"                # before parts(): tagger_prep re-points R.R13 at dev2
    P = B.parts()
    from benchmark.gold import tagger_prep as T          # after parts(): its import patches load_gold
    roots = {"dev": dev_root, "dev2": T.out("dev2") / f"{a.arm}_{B.SYS}"}
    fdir = Path(a.frames) if a.frames else None
    if fdir:
        fdir.mkdir(parents=True, exist_ok=True)
    cases = []
    for pt, st, g, pics in P:
        root = roots[pt]
        pics3 = [p[:3] for p in pics]
        full = S.score_clip(g, pics3)
        # each picture's class: remove it and see which count changes
        for p in pics:
            r = S.score_clip(g, [q[:3] for q in pics if q is not p])
            typ = [k for k in ("hit", "visible", "cross", "phantom", "dup") if full[k] != r[k]]
            if "visible" not in typ:
                continue
            lab, pa, pb, resc = p
            gm = [x for x in g if S.same_family(lab, x["label"]) and S.in_window(pa, x["start"], S.EARLY, S.LATE)]
            cases.append({"kind": "visible_wrong", "part": pt, "clip": st, "pic": [lab, round(pa, 2), round(pb, 2)], "rescued": resc,
                          "gold": [{k: x.get(k) for k in ("label", "start", "end", "visible", "obvious", "importance", "needed")} for x in gm],
                          "votes": votes_for(root, st, lab, pa, pb), "specs": specs_for(root, st, lab, pa, pb)})
        # misses whose family spec was silenced by the gate
        for x in g:
            if not x["needed"] or x["importance"] < S.MIN_IMPORTANCE:
                continue
            if any(S.same_family(l, x["label"]) and S.in_window(pa, x["start"], S.EARLY, S.LATE) for l, pa, _b, _r in pics):
                continue
            sp = specs_for(root, st, x["label"], x["start"] - 0.5, x["start"] + 1.0)
            if not sp:
                continue
            if any(s["augment"] for s in sp):
                kind = "miss_drawn_elsewhere"
            elif any("visible" in (s["reason"] or "") for s in sp):
                kind = "miss_gate_visible"
            else:
                kind = "miss_other"
            if kind != "miss_gate_visible":
                continue
            cases.append({"kind": kind, "part": pt, "clip": st, "gold": {k: x.get(k) for k in ("label", "start", "end", "visible", "obvious", "importance")},
                          "votes": votes_for(root, st, x["label"], x["start"] - 0.5, x["start"] + 1.0), "specs": sp})
    for i, c in enumerate(cases):
        if fdir:
            v = video_of(c["part"], c["clip"], roots[c["part"]])
            st = (c["votes"] or [{}])[0].get("stretch") if c["votes"] else None
            if st is None:
                st = [c["pic"][1], c["pic"][2]] if "pic" in c else [c["gold"]["start"], c["gold"]["end"]]
            if v and "pic" in c and c["pic"][2] - c["pic"][1] > 6:   # long pictures: also the last stretch
                c["frames_end"] = str(fdir / f"{i:02d}_{c['clip']}_end.jpg")
                sheet(v, c["pic"][2] - 5.0, c["pic"][2], c["frames_end"])
            if v:
                c["frames"] = str(fdir / f"{i:02d}_{c['clip']}_{(c.get('pic') or [c['gold']['label']])[0].replace(' ', '_').replace(',', '')}.jpg")
                sheet(v, st[0], max(st[1], st[0] + 0.5), c["frames"])
        print(json.dumps(c, default=float))
    OUT.write_text(json.dumps(cases, indent=1, default=float), encoding="utf-8")
    print(len(cases), "cases:", sum(c["kind"] == "visible_wrong" for c in cases), "visible wrong,",
          sum(c["kind"] == "miss_gate_visible" for c in cases), "gate-silenced misses")


if __name__ == "__main__":
    main()
