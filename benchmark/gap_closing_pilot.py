"""Gap-closing score, kill-criterion pilot (docs/prereg_gap_closing.md, commit 189c357).

The same audio-visual model answers four fixed questions about a clip three times: with
the soundtrack (D_sound), muted on the same frames (D_mute), and muted on frames taken a
quarter-slot earlier (D_mute2, the wording-noise floor). gap = 1 - cos(D_sound, D_mute);
noise = 1 - cos(D_mute, D_mute2); MiniLM cosine on each answer, averaged over questions.

PASS iff AUROC(gap: 40 ambient vs 20 no-ambient) >= 0.75 AND median gap on ambient clips
> 90th percentile of noise. Sample: first 20 clips by sorted name of each dev label.
Nothing here is tuned after a run.

    python -m benchmark.gap_closing_pilot            # GPU: the three runs per clip
    python -m benchmark.gap_closing_pilot --eval     # login node
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.av_reference_pilot import Omni, frames_and_audio

OUT_DIR = _ROOT / "benchmark" / "gap_closing_pilot"
SUMMARY = _ROOT / "benchmark" / "gap_closing_pilot.json"
PER_LABEL = 20
QUESTIONS = ["what_happening", "danger", "mood", "out_of_view"]
PROMPT = (
    "You are watching a short video clip. Answer these four questions about it, each in one or two "
    "sentences. Do not quote what people say and do not describe music.\n"
    "1. What is happening?\n"
    "2. Is anything dangerous or urgent happening? If so, what?\n"
    "3. What is the mood of the scene?\n"
    "4. Is anything happening that you cannot see? If so, what?\n"
    "Answer with JSON only, in this exact form: "
    "{\"what_happening\": \"...\", \"danger\": \"...\", \"mood\": \"...\", \"out_of_view\": \"...\"}"
)
BAR = {"auroc_ambient_vs_none": 0.75, "noise_percentile": 90}


def parse(text: str):
    m = re.search(r"\{.*\}", text, re.S)
    out = {q: "" for q in QUESTIONS}
    if m:
        try:
            d = json.loads(m.group(0))
            for q in QUESTIONS:
                out[q] = str(d.get(q, "")).strip()
        except Exception:
            pass
    if not any(out.values()):                       # loose fallback: numbered lines
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        for q, l in zip(QUESTIONS, lines[:4]):
            out[q] = re.sub(r"^\d+[.)]\s*", "", l)
    return out


def sample():
    from benchmark.gate_dev_sweep import CACHE_DIR, _find_clip, load
    tags = {r["clip"]: r["tag"] for r in load("dev")}
    by = {}
    for f in sorted((CACHE_DIR / "dev").glob("*.json")):
        v = _find_clip(f.stem); t = tags.get(f.stem)
        if v is None or t is None:
            continue
        by.setdefault(t, []).append((f.stem, v, t))
    chosen = []
    for t in ("unseen_ambient", "seen_ambient", "no_ambient"):
        chosen += by.get(t, [])[:PER_LABEL]
    return chosen


def run():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    todo = [x for x in sample() if not (OUT_DIR / (x[0] + ".json")).exists()]
    print(f"[gap] {len(todo)} clips to do", flush=True)
    if not todo:
        return
    bk = Omni(config.DEVICE)
    with tempfile.TemporaryDirectory() as td:
        for i, (stem, video, tag) in enumerate(todo, 1):
            try:
                frames, audio, sr, dur = frames_and_audio(Path(video), Path(td), phase=0.5)
                frames2, _, _, _ = frames_and_audio(Path(video), Path(td), phase=0.25)
                raw = {"sound": bk.ask(frames, audio, sr, PROMPT),
                       "mute": bk.ask(frames, None, sr, PROMPT),
                       "mute2": bk.ask(frames2, None, sr, PROMPT)}
            except Exception as e:
                print(f"  ! {stem}: {type(e).__name__}: {e}", flush=True); continue
            rec = {"clip": stem, "tag": tag, "duration": dur,
                   "answers": {k: parse(v) for k, v in raw.items()},
                   "raw": {k: v[:1500] for k, v in raw.items()}}
            (OUT_DIR / (stem + ".json")).write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"[gap] {i}/{len(todo)} {stem} [{tag}] danger(sound)='{rec['answers']['sound']['danger'][:60]}'", flush=True)


def auroc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).sum(); eq = (pos[:, None] == neg[None, :]).sum()
    return float((gt + 0.5 * eq) / (len(pos) * len(neg)))


def evaluate():
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    def cos(a: str, b: str) -> float:
        if not a or not b:
            return 0.0
        e = m.encode([a, b], normalize_embeddings=True)
        return float(e[0] @ e[1])

    recs = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(OUT_DIR.glob("*.json"))]
    rows = []
    for r in recs:
        a = r["answers"]
        gq = {q: 1 - cos(a["sound"][q], a["mute"][q]) for q in QUESTIONS}
        nq = {q: 1 - cos(a["mute"][q], a["mute2"][q]) for q in QUESTIONS}
        rows.append({"clip": r["clip"], "tag": r["tag"], "gap": float(np.mean(list(gq.values()))),
                     "noise": float(np.mean(list(nq.values()))), "gap_q": gq, "noise_q": nq,
                     "danger_sound": a["sound"]["danger"], "danger_mute": a["mute"]["danger"]})
    amb = [x["gap"] for x in rows if x["tag"] in ("unseen_ambient", "seen_ambient")]
    none = [x["gap"] for x in rows if x["tag"] == "no_ambient"]
    unseen = [x["gap"] for x in rows if x["tag"] == "unseen_ambient"]
    rest = [x["gap"] for x in rows if x["tag"] != "unseen_ambient"]
    noise = [x["noise"] for x in rows]
    a1 = auroc(amb, none); med = float(np.median(amb)) if amb else float("nan")
    p90 = float(np.percentile(noise, BAR["noise_percentile"])) if noise else float("nan")
    passed = a1 >= BAR["auroc_ambient_vs_none"] and med > p90
    per_q = {q: {"gap_ambient_median": float(np.median([x["gap_q"][q] for x in rows if x["tag"] != "no_ambient"])),
                 "gap_none_median": float(np.median([x["gap_q"][q] for x in rows if x["tag"] == "no_ambient"])),
                 "noise_median": float(np.median([x["noise_q"][q] for x in rows]))} for q in QUESTIONS}
    siren = [x for x in rows if re.search(r"siren|alarm|fire_engine|fire_truck|police|ambulance", x["clip"], re.I)]
    summary = {"when": datetime.now().isoformat(timespec="minutes"), "model": "Qwen/Qwen2.5-Omni-7B", "clips": len(rows),
               "n_ambient": len(amb), "n_none": len(none), "auroc_ambient_vs_none": a1,
               "median_gap_ambient": med, "median_gap_none": float(np.median(none)) if none else None,
               "noise_p90": p90, "median_noise": float(np.median(noise)) if noise else None,
               "auroc_unseen_vs_rest_(info_only)": auroc(unseen, rest), "per_question": per_q,
               "siren_alarm_clips": [{"clip": x["clip"], "tag": x["tag"], "gap": x["gap"], "noise": x["noise"],
                                      "danger_sound": x["danger_sound"], "danger_mute": x["danger_mute"]} for x in siren],
               "bar": BAR, "passed": passed, "rows": rows}
    SUMMARY.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"[eval] {len(rows)} clips | AUROC gap ambient({len(amb)}) vs none({len(none)}) = {a1:.3f} (bar >= 0.75) | "
          f"median gap ambient {med:.3f} vs none {np.median(none):.3f}; noise p90 {p90:.3f}, median {np.median(noise):.3f} | "
          f"unseen-vs-rest AUROC (info) {auroc(unseen, rest):.3f} | {'PASSED' if passed else 'FAILED'} -> {SUMMARY}")
    for q, v in per_q.items():
        print(f"   {q:15s} gap ambient {v['gap_ambient_median']:.3f} | none {v['gap_none_median']:.3f} | noise {v['noise_median']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    evaluate() if a.eval else run()


if __name__ == "__main__":
    main()
