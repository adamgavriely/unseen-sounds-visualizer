"""Outside one-model baselines (5 Oct 2026, report review): ONE ready-made model hears the clip and a picture is drawn
for every sound it reports. Nothing from our detector stack, gate, display rules or label filter.

  m2d   PretrainedSED M2D-strong (Schmid et al., CP-JKU, ICASSP 2025; MIT): M2D fine-tuned frame by frame on
        AudioSet-Strong, 447 classes, a probability every 40 ms, PSDS1 46.3. Cache data/work/psed_ens_cache/M2D
        (psed_infer.cache_clips, backbone "M2D") + m2d_cache/ for the clips it lacked (cache_m2d.py, same code).
  panns PANNs CNN14 (Kong et al. 2020; MIT): AudioSet-2M, 527 classes, framewise. Cache benchmark/gold/panns_fw +
        panns_cache/ for the clips it lacked (cache_panns.py, same call).
  qwen  Qwen3-Omni-30B-A3B-Instruct (Apache-2.0): lists the sounds with start seconds (qwen_omni.py). No bar.
  qwen_av  the same model watching video + audio, listing only sounds whose source it cannot see (qwen_omni_av.py).

Frame models: standard double-threshold (hysteresis) spans, low = bar / 2, minimum 0.5 s; one picture per span. The
bar is chosen ONCE on the merged DEV set (71 clips) by the report's viewer cost (4 x miss + 2 x wrong per clip), then
the merged TEST set (88 clips) is scored ONCE at that bar. For every model only speech and music are never drawn.
Same per-sound scorer, clip lists and paired clip bootstrap (100,000 draws, seed 0) as
benchmark/gold/final_vs_baselines.py; its rows for the final system and direct audio-to-image are rebuilt here and
checked against final_vs_baselines.json.

    python benchmark/gold/one_model_baseline/run.py --model m2d      -> result_m2d.json   (also writes stems.json)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(HERE))
from benchmark.gold import final_vs_baselines as FVB
from benchmark.gold import dev_candidates_check as DCC
from benchmark.gold import score_per_sound as S
from src.labels import SPEECH_LABELS, is_music, is_descendant
from src.stage4_audio_event_detection import _extract_events

CACHES = {"m2d": (_ROOT / "data" / "work" / "psed_ens_cache" / "M2D", HERE / "m2d_cache"),
          "panns": (_ROOT / "benchmark" / "gold" / "panns_fw", HERE / "panns_cache")}
NAMES = {"m2d": "PretrainedSED M2D_strong_1 (MIT)", "panns": "PANNs CNN14 (MIT)",
         "qwen": "Qwen3-Omni-30B-A3B-Instruct (Apache-2.0), audio only",
         "qwen_av": "Qwen3-Omni-30B-A3B-Instruct (Apache-2.0), video + audio, off-screen sounds only"}
GRID = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7]
MIN_DUR = 0.5
MODEL = "m2d"
TEST_LIST = _ROOT / "benchmark" / "gold" / "test_stems.txt"     # the report's TEST set (87 clips)
TEST_KEEP = set(TEST_LIST.read_text(encoding="utf-8").split())
TEST_SET = len(TEST_KEEP)

REC = []                                  # [(stem, gold sounds)] in the exact row order of final_vs_baselines
_orig_part = FVB._part


def _part(R, out_dir, gold, stems, sysn):
    rows = _orig_part(R, out_dir, gold, stems, sysn)
    if sysn == "proposed":
        REC.append([(st, gold[st]) for st in stems if st in gold])
    return rows


FVB._part = _part


def never_drawn(label):
    return label in SPEECH_LABELS or is_descendant(label, "Speech") or is_music(label)


def cache_of(stem):
    a, b = CACHES[MODEL]
    return a / f"{stem}.npz" if (a / f"{stem}.npz").exists() else b / f"{stem}.npz"


_FW = {}
_Q = {"dropped": 0}


def pictures(stem, bar):
    if MODEL.startswith("qwen"):
        import qwen_omni as Q
        if "vocab" not in _Q:
            _Q["vocab"] = Q.vocabulary()
            _Q["replies"] = json.loads(replies_file().read_text(encoding="utf-8"))
        r = _Q["replies"][stem]
        if MODEL == "qwen_av":
            import qwen_omni_av as QA
            pics, dropped = QA.parse(r["reply"], _Q["vocab"], r["duration"])
        else:
            pics, dropped = Q.parse(r["reply"], _Q["vocab"], r["duration"])
        _Q["dropped"] += dropped
        return [p for p in pics if not never_drawn(p[0])]
    if stem not in _FW:
        z = np.load(cache_of(stem), allow_pickle=False)
        _FW[stem] = (z["fw"].astype(np.float32), z["times"].astype(np.float64), [str(x) for x in z["labels"]])
    fw, ts, labs = _FW[stem]
    ev = _extract_events(fw, ts, labs, bar, None, MIN_DUR, low=bar * 0.5)
    return [(e.label, float(e.start), float(e.end)) for e in ev if not never_drawn(e.label)]


def replies_file():
    import qwen_omni as Q
    return Q.OUT if MODEL == "qwen" else HERE / "qwen_av_replies.json"


def rows_at(clips, bar):
    return [S.score_clip(g, pictures(st, bar)) for st, g in clips]


def keep(m):
    return {k: m[k] for k in ("hits", "misses", "wrong", "visible", "cross", "phantom", "viewer_cost")}


def compare(rows, base):
    cost = {s: np.array([DCC.clip_cost(r) for r in rows[s]]) for s in rows}
    nothing = np.array([4.0 * FVB.S_hits_misses(r) for r in rows["proposed"]])
    out = {"rows": {s: keep(DCC.metrics(rows[s])) for s in rows}, "diff": {}}
    out["rows"]["show_nothing"] = {"viewer_cost": float(nothing.mean())}
    out["diff"]["one_model_vs_show_nothing"] = FVB.boot2(cost["one_model"] - nothing)
    out["diff"]["one_model_vs_audio_to_image"] = FVB.boot2(cost["one_model"] - cost["blind_a2i"])
    out["diff"]["final_vs_one_model"] = FVB.boot2(cost["proposed"] - cost["one_model"])
    out["clips"] = len(nothing)
    out["parity_with_final_vs_baselines"] = all(
        abs(out["rows"][s]["viewer_cost"] - base["rows"][s]["viewer_cost"]) < 1e-9 for s in ("proposed", "blind_a2i"))
    return out


def main():
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=tuple(NAMES), default="m2d")
    MODEL = ap.parse_args().model
    base = json.loads((_ROOT / "benchmark" / "gold" / "final_vs_baselines.json").read_text(encoding="utf-8"))
    REC.clear(); dev = FVB.dev_rows(); dev_clips = [c for part in REC for c in part]
    REC.clear(); test = FVB.test_rows(); test_clips = [c for part in REC for c in part]
    assert len(dev_clips) == len(dev["proposed"]) and len(test_clips) == len(test["proposed"]), "row order mismatch"
    (HERE / "stems.json").write_text(json.dumps({"dev": [s for s, _ in dev_clips], "test": [s for s, _ in test_clips]}))
    if MODEL.startswith("qwen"):
        have = json.loads(replies_file().read_text(encoding="utf-8")) if replies_file().exists() else {}
        missing = [st for st, _ in dev_clips + test_clips if st not in have]
    else:
        missing = [st for st, _ in dev_clips + test_clips if not cache_of(st).exists()]
    if missing:
        (HERE / f"missing_{MODEL}.txt").write_text(" ".join(missing))
        print(f"missing {MODEL}:", len(missing), f"-> missing_{MODEL}.txt"); sys.exit(1)

    out = {"model": NAMES[MODEL], "rule": "speech and music never drawn; one picture per span / listed start"}
    if MODEL.startswith("qwen"):
        bar = None
        out["rule"] += "; no bar (greedy reply parsed as written)"
    else:
        grid = {}
        for b in GRID:
            m = DCC.metrics(rows_at(dev_clips, b))
            grid[b] = keep(m)
            print(f"DEV bar {b:.2f}: hits {m['hits']}/{m['hits'] + m['misses']} wrong {m['wrong']} cost {m['viewer_cost']:.3f}")
        bar = min(GRID, key=lambda t: grid[t]["viewer_cost"])
        print(f"bar chosen on DEV: {bar}  (TEST scored once, at this bar only)")
        out.update({"bar_chosen_on_dev": bar, "dev_grid": grid})
        out["rule"] += "; hysteresis low = bar/2, min 0.5 s"
    for name, rows, clips in (("development", dev, dev_clips), ("test", test, test_clips)):
        rows = dict(rows); rows["one_model"] = rows_at(clips, bar)
        parity = compare(rows, base[name])["parity_with_final_vs_baselines"]
        if name == "test":
            keep_i = [i for i, (st, _) in enumerate(clips) if st in TEST_KEEP]
            rows = {k: [v[i] for i in keep_i] for k, v in rows.items()}
            assert len(keep_i) == TEST_SET, len(keep_i)
        out[name] = compare(rows, base[name])
        out[name]["parity_with_final_vs_baselines"] = parity
        kept = [c for c in clips if name != "test" or c[0] in TEST_KEEP]
        out[name]["per_clip"] = [{"clip": st, **{s: {"cost": float(DCC.clip_cost(rows[s][i])),
                                                     **{k: int(rows[s][i][k]) for k in ("hit", "miss", "visible", "cross", "phantom", "dup")}}
                                                 for s in rows}} for i, (st, _) in enumerate(kept)]
    if MODEL.startswith("qwen"):
        out["reply_lines_dropped_unknown_name"] = _Q["dropped"]
    dst = HERE / f"result_{MODEL}.json"
    dst.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    for name in ("development", "test"):
        r = out[name]
        print(name, r["clips"], "clips  parity", r["parity_with_final_vs_baselines"])
        for s, x in r["rows"].items():
            print("  ", s, x)
        for k, v in r["diff"].items():
            print("  ", k, "d %+.3f [%+.3f, %+.3f] p %.4f" % tuple(v))
    print("->", dst)


if __name__ == "__main__":
    main()
