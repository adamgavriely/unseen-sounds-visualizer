"""Stage-5 gate accuracy on the per-sound gold (amendment 5, docs/prereg_v4.md): for every sound
the annotator timed, the visibility check is asked exactly as the pipeline asks it (the sound's
family label, six frames per 5-s stretch, three votes: open naming / a-b in both orderings /
description), and its verdict is compared with the annotator's visible-or-obvious tick.

SOTA vs others (Adam 2026-09-22): Qwen3.8-27B (v4 gate), Qwen2.5-VL-7B (v3 gate), and the
whole-clip object detector OWLv2 with the concept table (v2 gate). Votes are cached per clip so
the silence rule (majority / unanimous) is re-decided on CPU; selection on DEV only.

    python benchmark/gold/gate_gold.py --model Qwen/Qwen3.8-27B            # GPU
    python benchmark/gold/gate_gold.py --model Qwen/Qwen2.5-VL-7B-Instruct # GPU
    python benchmark/gold/gate_gold.py --owl                                # GPU (small)
    python benchmark/gold/gate_gold.py --score                              # CPU
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.detector_dry import clip_path, JUDGE100

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
OUT_DIR = _ROOT / "benchmark" / "gold" / "gate_gold"
STRETCH = 5.0


def gold_sounds():
    """[(clip name, stem, [sound dicts with resolved label])]"""
    d = json.loads(GOLD.read_text(encoding="utf-8"))
    out = []
    for c in d["clips"]:
        if not isinstance(c, dict) or not c.get("done") or c.get("bad"):
            continue
        snds = []
        for s in c.get("sounds", []):
            lab = s.get("family") or s.get("label")
            if not lab or s.get("start") is None or s.get("end") is None:
                continue
            snds.append({"label": S.resolve_label(lab), "raw": lab, "start": float(s["start"]), "end": float(s["end"]),
                         "seen": bool(s.get("visible") or s.get("obvious")), "visible": bool(s.get("visible")),
                         "obvious": bool(s.get("obvious")), "importance": int(s.get("importance") or 2)})
        out.append((c["clip"], Path(c["clip"]).stem, snds))
    return out


def _short(model: str) -> str:
    return model.split("/")[-1].replace("-Instruct", "").replace(".", "")


def run_vlm(model: str, device: str = "cuda"):
    from src.stage5_cross_modal_analysis import reason
    from src.stage2_video_understanding import _sample_frames_at
    out_dir = OUT_DIR / _short(model); out_dir.mkdir(parents=True, exist_ok=True)
    mdl, proc = reason._load(model, device)
    for name, stem, snds in gold_sounds():
        f = out_dir / f"{stem}.json"
        if f.exists():
            continue
        p = clip_path(name)
        if p is None:
            print("missing", name); continue
        rows = []
        for s in snds:
            a0, b0 = s["start"], max(s["end"], s["start"] + 0.5)
            k = max(1, int(round((b0 - a0) / STRETCH)))
            edges = [a0 + (b0 - a0) * i / k for i in range(k + 1)]
            stretches = []
            for a, b in zip(edges, edges[1:]):
                n = 6; lo, hi = a - 1.0, b + 1.0
                times = [lo + (hi - lo) * t / (n - 1) for t in range(n)]
                win = _sample_frames_at(p, times)
                seen, named = reason._sound_is_visible(s["label"], win, mdl, proc, device)
                stretches.append({"start": a, "end": b, "seen_majority": bool(seen), **dict(reason.LAST_VOTES)})
            rows.append({**s, "stretches": stretches})
        f.write_text(json.dumps({"clip": name, "sounds": rows}, indent=1), encoding="utf-8")
        print(stem, len(rows), "sounds", flush=True)


def run_owl(device: str = "cuda"):
    from src.stage2_video_understanding.owl import analyze_video_owl
    from src.stage2_video_understanding import VISIBLE_CONCEPTS
    from src.labels import canonical, FAMILY
    out_dir = OUT_DIR / "owlv2"; out_dir.mkdir(parents=True, exist_ok=True)
    for name, stem, snds in gold_sounds():
        f = out_dir / f"{stem}.json"
        if f.exists():
            continue
        p = clip_path(name)
        if p is None:
            continue
        keys = []
        for s in snds:
            fam = FAMILY.get(s["label"], canonical(s["label"]))
            s["concept"] = fam if fam in VISIBLE_CONCEPTS else (s["label"] if s["label"] in VISIBLE_CONCEPTS else None)
            if s["concept"]:
                keys.append(s["concept"])
        scene = analyze_video_owl(p, num_frames=6, model=config.OWL_MODEL, device=device,
                                  threshold=config.OWL_THRESHOLD, candidates=sorted(set(keys)) or None)
        vis = {e.lower() for e in scene.visible_entities}
        rows = [{**s, "seen_owl": bool(s.get("concept") and s["concept"].lower() in vis)} for s in snds]
        f.write_text(json.dumps({"clip": name, "sounds": rows, "visible_entities": sorted(vis)}, indent=1), encoding="utf-8")
        print(stem, sorted(vis), flush=True)


def decide(stretches, rule: str) -> bool:
    """the pipeline's clip-level verdict: silent only if the source is visible in EVERY stretch"""
    for st in stretches:
        votes = [st.get("name"), st.get("ab"), st.get("desc")]
        yes = sum(1 for v in votes if v is True); no = sum(1 for v in votes if v is False)
        seen = (yes == 3) if rule == "unanimous" else yes > no
        if not seen:
            return False
    return True


def score():
    judge = set(JUDGE100.read_text().split())
    table = {}
    for arm in sorted(d.name for d in OUT_DIR.iterdir() if d.is_dir()):
        files = list((OUT_DIR / arm).glob("*.json"))
        rules = ["owl"] if arm == "owlv2" else ["majority", "unanimous"]
        for rule in rules:
            for sub in ("dev54", "test85", "all"):
                sel = [f for f in files if sub == "all" or ((f.stem in judge) == (sub == "dev54"))]
                n_needed_kept = n_needed = n_seen_sil = n_seen = 0
                for f in sel:
                    d = json.loads(f.read_text(encoding="utf-8"))
                    for s in d["sounds"]:
                        if s["importance"] < 2:
                            continue
                        pred_seen = s["seen_owl"] if rule == "owl" else decide(s["stretches"], rule)
                        if s["seen"]:
                            n_seen += 1; n_seen_sil += int(pred_seen)
                        else:
                            n_needed += 1; n_needed_kept += int(not pred_seen)
                n = n_seen + n_needed
                r_sil = n_seen_sil / n_seen if n_seen else 0.0
                r_kept = n_needed_kept / n_needed if n_needed else 0.0
                acc = (n_seen_sil + n_needed_kept) / n if n else 0.0
                t = {"clips": len(sel), "n": n, "seen": n_seen, "needed": n_needed, "seen_silenced": r_sil,
                     "needed_kept": r_kept, "balanced_acc": (r_sil + r_kept) / 2, "acc": acc}
                table[f"{arm}|{rule}|{sub}"] = t
                print(f"[{arm:14s} {rule:9s} {sub:6s}] clips {t['clips']:3d} sounds {n:3d} (seen {n_seen}, needed {n_needed}) | "
                      f"seen silenced {r_sil:.2f} | needed kept {r_kept:.2f} | balanced {t['balanced_acc']:.2f} | acc {acc:.2f}")
    (OUT_DIR / "summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--owl", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    if a.model:
        run_vlm(a.model, a.device)
    if a.owl:
        run_owl(a.device)
    if a.score:
        score()


if __name__ == "__main__":
    main()
