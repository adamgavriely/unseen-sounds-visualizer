"""PREPARATION for the one TEST exposure. No gold is read, nothing is scored.

Builds, for the 60 TEST clips (benchmark/gold/test_stems.txt), the same feature caches the DEV harness
(benchmark/gold/dev_harness.py, dev_candidates_check.py, dev_listener.py) has for DEV, into parallel folders, and runs the
two plumbing gates on B0r (flags off = the scored config) against the scored TEST render (tag test_final_v33, the one the
frozen amendment-21 TEST table used; config.use_scored()):
  D0  stage 4 rebuilt by the pipeline's own code (extract, flexsed_raw, union, veto) == the scored onset_trace, per clip
  D5  stage 5 rebuilt (gate answers reused, other questions asked live and memoised) == the scored augmentations.json and
      on-screen spans (label, start, end), per clip and system. Structural equality only: no picture is opened.

It is a thin wrapper: dev_harness / dev_candidates_check are imported unchanged and their module globals are redirected.
score_per_sound.load_gold is replaced by a function that raises, so no step here can touch gold.

    python benchmark/gold/test_harness_prep.py check    # CPU: what exists for TEST; trace / FlexSED format preflight
    python benchmark/gold/test_harness_prep.py beats    # GPU: BEATs framewise (shipped infer_beats) -> data/work/j2_test_beats
    python benchmark/gold/test_harness_prep.py wav16    # CPU: ffmpeg of each mp4, 16 kHz mono -> data/work/r13test/wav16
    python benchmark/gold/test_harness_prep.py panns    # GPU: PANNs CNN14 clip peaks (the veto input) -> data/work/r13test/panns
    python benchmark/gold/test_harness_prep.py stage4   # GPU only for live onsets: B0r spans, gate D0 -> data/work/r13test/stage4.json
    python benchmark/gold/test_harness_prep.py stage5   # GPU: stage 5 for B0r, both systems -> data/work/r13test/B0r_<system>
    python benchmark/gold/test_harness_prep.py d5       # CPU: gate D5 (structural), writes data/work/r13test/gates.json
An arm's TEST run later uses the same redirect: `python benchmark/gold/test_harness_prep.py stage4 --arms B0r R13-1` etc.

(design record: release v1.2.0)
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S


def _no_gold(*a, **k):
    raise RuntimeError("test_harness_prep: gold must not be read (TEST preparation only)")


S.load_gold = _no_gold                                   # hard guard, before anything else imports it by name

from benchmark.gold import dev_candidates_check as DCC   # noqa: E402
from benchmark.gold import dev_harness as R              # noqa: E402

WORK = DCC.WORK
TEST_TAG = "test_final_v33"                              # the scored TEST render (amendment 21 table, config.use_scored)
STEMS = sorted(x.strip() for x in (_ROOT / "benchmark" / "gold" / "test_stems.txt").read_text(encoding="utf-8").splitlines()
               if not x.strip().startswith("tg_")                # the 60 first-batch clips (tg_ = test2, clip_prep.py)
               if x.strip())
assert len(STEMS) == 60, len(STEMS)
OUT = WORK / "r13test"
BEATS_TEST = WORK / "j2_test_beats"
GATES = OUT / "gates.json"

# ---- redirect the DEV harness to TEST (module globals; the functions read them at call time)
DCC.dev_stems = lambda: (None, list(STEMS))
DCC.TAG = TEST_TAG
DCC.BEATS_DIR = BEATS_TEST
DCC.WAV16 = OUT / "wav16"
DCC.DC = OUT / "dc"                                      # empty on purpose: no devcand stage4 ref, no memo copied
DCC.STAGE4, DCC.MEMO = DCC.DC / "stage4.json", DCC.DC / "ask_memo.json"
R.R13 = OUT
R.STAGE4, R.MEMO, R.PANNS_DIR = OUT / "stage4.json", OUT / "ask_memo.json", OUT / "panns"


def dump(p, obj):
    DCC.dump(p, obj)


def check():
    """what exists for TEST, plus a format preflight of the scored traces and the FlexSED cache (CPU, small)"""
    rep = {"tag": TEST_TAG, "stems": len(STEMS), "exists": {}, "preflight": {}}
    for name, f in (("beats j2", lambda s: BEATS_TEST / f"{s}.npz"),
                    ("flexsed", lambda s: DCC.FLEX_DIR / f"{s}.npz"),
                    ("panns_fw (benchmark/gold)", lambda s: _ROOT / "benchmark" / "gold" / "panns_fw" / f"{s}.npz"),
                    ("r13test panns peaks", lambda s: R.PANNS_DIR / f"{s}.npz"),
                    ("wav16", lambda s: DCC.WAV16 / f"{s}.wav")):
        rep["exists"][name] = sum(f(s).exists() for s in STEMS)
    for sysn in DCC.SYSTEMS:
        d = DCC.scored_dir(sysn)
        for fn in ("audio.wav", "onset_trace.json", "media.json", "scene.json", "segments.json", "augmentations.json",
                   "gate_votes.json"):
            rep["exists"][f"{sysn}/{fn}"] = sum((d / s / fn).exists() for s in STEMS)
    # preflight: every trace has the five steps used by D0 and len(veto) == len(refine) (dev_harness asserts it)
    bad = []
    for sysn in DCC.SYSTEMS:
        for s in STEMS:
            p = DCC.scored_dir(sysn) / s / "onset_trace.json"
            if not p.exists():
                bad.append([sysn, s, "no trace"]); continue
            tr = json.loads(p.read_text(encoding="utf-8"))
            steps = {x["step"] for x in tr}
            nv = sum(x["step"] == "veto" for x in tr); nr = sum(x["step"] == "refine" for x in tr)
            if nv != nr or "extract" not in steps:
                bad.append([sysn, s, f"steps {sorted(steps)} veto {nv} refine {nr}"])
    rep["preflight"]["trace_bad"] = bad
    # FlexSED npz format equal to a DEV one (keys, labels, fps)
    dev_one = sorted(p for p in DCC.FLEX_DIR.glob("*.npz") if p.stem not in set(STEMS))
    fb = []
    if dev_one:
        z0 = np.load(dev_one[0])
        k0, l0 = sorted(z0.files), [str(x) for x in z0["labels"]]
        f0 = float(z0["fps"]) if "fps" in z0.files else None
        for s in STEMS:
            p = DCC.FLEX_DIR / f"{s}.npz"
            if not p.exists():
                fb.append([s, "missing"]); continue
            z = np.load(p)
            if sorted(z.files) != k0 or [str(x) for x in z["labels"]] != l0 or \
                    (f0 is not None and float(z["fps"]) != f0):
                fb.append([s, f"keys {sorted(z.files)}"])
    rep["preflight"]["flexsed_bad"] = fb
    print(json.dumps(rep, indent=1), flush=True)
    g = json.loads(GATES.read_text(encoding="utf-8")) if GATES.exists() else {}
    g["check"] = rep
    dump(GATES, g)


def beats():
    """exactly benchmark/gold/j2_dev_check.beats() (release v1.2.0), on the scored TEST render's audio.wav"""
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    BEATS_TEST.mkdir(parents=True, exist_ok=True)
    for st in STEMS:
        dst = BEATS_TEST / f"{st}.npz"
        wav = DCC.scored_dir("proposed") / st / "audio.wav"
        if dst.exists() or not wav.exists():
            continue
        fw, t, labs = infer_beats(wav, "cuda")
        np.savez_compressed(dst, fw=fw.astype(np.float32), times=np.asarray(t, np.float64), labels=np.array(labs))
        print(f"[beats] {st} {fw.shape}", flush=True)


def wav16():
    """as dev_candidates_check.paralist(): ffmpeg of the clip's mp4, mono 16 kHz"""
    DCC.WAV16.mkdir(parents=True, exist_ok=True)
    for st in STEMS:
        wav = DCC.WAV16 / f"{st}.wav"
        if wav.exists():
            continue
        video = json.loads((DCC.scored_dir("proposed") / st / "media.json").read_text(encoding="utf-8"))["video_path"]
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-ac", "1", "-ar", "16000", str(wav)], check=True)
    print(f"[wav16] {sum((DCC.WAV16 / f'{s}.wav').exists() for s in STEMS)} / {len(STEMS)}", flush=True)


def d5():
    """gate D5 only (dev_harness.score's `bad` line): rebuilt B0r vs the scored render, structural, no gold"""
    g = json.loads(GATES.read_text(encoding="utf-8")) if GATES.exists() else {}
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    d0 = s4["d0"]
    g["d0"] = {"pass": sum(v["pass"] for v in d0.values()), "of": len(d0), "fails": [k for k, v in d0.items() if not v["pass"]]}
    live = sum(1 for k, v in s4["arms"].items() if k.startswith("B0r|") for rr in v.values() for r in rr if r["refine"] == "live")
    g["b0r_live_refinements"] = live
    g["d5"] = {}
    for sysn in DCC.SYSTEMS:
        root = OUT / f"B0r_{sysn}"
        if not DCC.complete(root, STEMS):
            g["d5"][sysn] = "incomplete"; continue
        bad = [st for st in STEMS
               if DCC.pics_sig(S.load_pictures(root, st, sysn) or []) != DCC.pics_sig(S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [])
               or DCC.spec_sig(root, st) != DCC.spec_sig(DCC.scored_dir(sysn), st)]
        lg = root / "_stage5_log.json"
        st5 = json.loads(lg.read_text(encoding="utf-8")) if lg.exists() else {}
        tot = {}
        for v in st5.values():
            for k in ("gate_reused", "gate_live", "ask_memo", "ask_live"):
                tot[k] = tot.get(k, 0) + v.get(k, 0)
        g["d5"][sysn] = {"pass": len(STEMS) - len(bad), "of": len(STEMS), "differ": bad, "stage5": tot}
    print(f"[D0 TEST] {g['d0']['pass']} / {g['d0']['of']}; fails {g['d0']['fails']}; B0r live refinements {live}", flush=True)
    for sysn, v in g["d5"].items():
        msg = v if isinstance(v, str) else "%d / %d; differ %s; %s" % (v["pass"], v["of"], v["differ"], v["stage5"])
        print(f"[D5 TEST {sysn}] {msg}", flush=True)
    dump(GATES, g)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("check", "beats", "wav16", "panns", "stage4", "stage5", "d5"))
    ap.add_argument("--arms", nargs="+", default=["B0r"])
    a = ap.parse_args()
    {"check": check, "beats": beats, "wav16": wav16, "panns": R.panns,
     "stage4": lambda: R.stage4(a.arms), "stage5": lambda: R.stage5(a.arms), "d5": d5}[a.step]()


if __name__ == "__main__":
    main()
