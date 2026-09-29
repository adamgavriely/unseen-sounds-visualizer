"""Round 14 amendment J4 feature cache (docs/prereg_round13_detector_push.md, "Round 14 amendment J"): audio-LLM onset
timestamps from Qwen3-Omni and Audio Flamingo Next on the weak / rescued runs. TEST: features only (no gold is read by
pool / run / merge; score_per_sound.load_gold raises). The DEV screen reads DEV gold only (the 49 DEV stems).

Items (per split): the items of dev_listener.json / test_listener.json with pool P2 and peak >= 0.4, or pool P3; depictable.
Audio: [peak_t - 2, peak_t + 2] s clipped to the clip (not shifted), at least 1 s (listener_variants' cut()), from the same
wav16 files as the listener caches (listener_variants.SPLITS[split]["wav"]).
Question (greedy, max_new_tokens 8), family lower case as in the other listeners:
    "At which second of this recording does the {family} sound start? Reply with a number, or none."
Qwen3-Omni: as listener_variants.score (thinker.generate, talker disabled). AF Next: listener_afnext.load_model()'s gen().
Parse: first number in the reply (listener_variants' V3 regex) = t_rel (window seconds); t_clip = win_start + t_rel;
no number -> none. out_of_window flags t_rel < 0 or > window length (kept as given).

    python benchmark/gold/listener_timestamps.py pool                 # CPU: item lists -> *_listener_ts_items.json
    python benchmark/gold/listener_timestamps.py run --model qwen     # GPU: DEV then TEST (resumable partial file)
    python benchmark/gold/listener_timestamps.py run --model afn      # GPU
    python benchmark/gold/listener_timestamps.py merge                # CPU: -> dev_listener_ts.json, test_listener_ts.json
    python benchmark/gold/listener_timestamps.py screen               # CPU: DEV screen (J3 + J4) -> motion_ts_screen.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import listener_variants as LV
from src.labels import canonical

GOLD = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
SPLITS = ("dev", "test")
CACHE = {"dev": GOLD / "dev_listener.json", "test": GOLD / "test_listener.json"}
ITEMS = {s: GOLD / f"{s}_listener_ts_items.json" for s in SPLITS}
PART = {(s, m): GOLD / f"{s}_listener_ts_{m}.json" for s in SPLITS for m in ("qwen", "afn")}
OUT = {s: GOLD / f"{s}_listener_ts.json" for s in SPLITS}
SCREEN = GOLD / "motion_ts_screen.json"
Q = "At which second of this recording does the {} sound start? Reply with a number, or none."
NEW, HALF, P2_PEAK = 8, 2.0, 0.4
NUM = re.compile(r"-?\d+(?:\.\d+)?")
SR = LV.SR
MODELS = {"qwen": "Qwen/Qwen3-Omni-30B-A3B-Instruct", "afn": "nvidia/audio-flamingo-next-hf"}


def _no_gold(*a, **k):
    raise RuntimeError("listener_timestamps: gold must not be read in pool/run/merge")


def key(x):
    return (x["clip"], x["family"], round(float(x["start"]), 3), round(float(x["end"]), 3))


def skey(x):
    return "|".join(str(v) for v in key(x))


# ============================================================================= items
def pool():
    import soundfile as sf
    S.load_gold = _no_gold
    for split in SPLITS:
        d = json.loads(CACHE[split].read_text(encoding="utf-8"))
        src = [x for x in d["items"] if x.get("depictable") and
               ((x["pool"] == "P2" and float(x["peak"]) >= P2_PEAK) or x["pool"] == "P3")]
        dur, out, seen = {}, [], {}
        for x in src:
            k = key(x)
            if k in seen:                                             # identical (clip, family, start, end): ask once
                seen[k]["pools"] = sorted(set(seen[k]["pools"]) | {x["pool"]})
                continue
            st = x["clip"]
            if st not in dur:
                dur[st] = float(sf.info(str(LV.SPLITS[split]["wav"] / f"{st}.wav")).duration)
            pk = float(x["peak_t"])
            a, b = max(0.0, pk - HALF), min(dur[st], pk + HALF)
            it = {"clip": st, "family": x["family"], "label": x["label"], "pool": x["pool"], "pools": [x["pool"]],
                  "start": float(x["start"]), "end": float(x["end"]), "run_len": float(x["run_len"]),
                  "cut_start": float(x["cut_start"]), "cut_end": float(x["cut_end"]), "peak": float(x["peak"]),
                  "peak_t": pk, "origin": x.get("origin"), "clip_dur": dur[st], "win": [a, b]}
            seen[k] = it
            out.append(it)
        meta = {"amendment": "round 14 J4 (docs/prereg_round13_detector_push.md)", "split": split,
                "source_cache": str(CACHE[split].relative_to(_ROOT)),
                "items": f"pool P2 with peak >= {P2_PEAK} or pool P3; depictable; identical (clip, family, start, end) asked once",
                "window": f"[peak_t - {HALF}, peak_t + {HALF}] s clipped to the clip (not shifted), >= 1 s (listener_variants cut())",
                "wav": str(LV.SPLITS[split]["wav"]), "question": Q, "family_text": "family.lower()",
                "decoding": f"greedy (do_sample=False), max_new_tokens {NEW}"}
        C.dump(ITEMS[split], {"_meta": meta, "items": out})
        n = {p: sum(p in x["pools"] for x in out) for p in ("P2", "P3")}
        print(f"[pool] {split}: {len(src)} source items -> {len(out)} asked {n}; clips {len(dur)}; "
              f"window < 4 s: {sum(x['win'][1] - x['win'][0] < 3.999 for x in out)}", flush=True)


# ============================================================================= models
def load_qwen():
    """listener_variants.score(): processor, model, prep() and gen() unchanged (no mpnet)"""
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODELS["qwen"])
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODELS["qwen"], dtype=torch.bfloat16,
                                                                 device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass

    def prep(w, q):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        return inp.to(model.thinker.device).to(torch.bfloat16) if hasattr(inp, "to") else inp

    def gen(w, q, n):
        inp = prep(w, q)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=n, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    info = {"model": MODELS["qwen"], "load_seconds": round(time.time() - t0, 1),
            "call": "listener_variants.score prep()/gen(): thinker.generate, talker disabled, use_audio_in_video=False"}
    print(f"[qwen] loaded {info}", flush=True)
    return gen, info


def load_afn():
    from benchmark.gold import listener_afnext as AF
    M = AF.load_model()
    info = dict(M["info"], call="listener_afnext.load_model() gen(): model.generate, use_cache=True, no repetition_penalty")
    return M["gen"], info


def parse(txt, win_len, a):
    m = NUM.search(txt)
    t_rel = float(m.group(0)) if m else None
    return {"text": txt, "t_rel": t_rel, "t_clip": (a + t_rel) if t_rel is not None else None, "none": t_rel is None,
            "says_none": "none" in txt.lower(), "out_of_window": bool(t_rel is not None and (t_rel < 0 or t_rel > win_len + 1e-6))}


def run(model):
    import soundfile as sf
    S.load_gold = _no_gold
    gen, info = load_qwen() if model == "qwen" else load_afn()
    for split in SPLITS:
        outp = PART[(split, model)]
        base = json.loads(ITEMS[split].read_text(encoding="utf-8"))
        if outp.exists():
            d = json.loads(outp.read_text(encoding="utf-8"))
        else:
            d = {"_meta": dict(base["_meta"], **info), "answers": {}}
        todo = sorted([x for x in base["items"] if skey(x) not in d["answers"]], key=lambda x: (x["clip"], x["win"][0]))
        print(f"[{model}] {split}: {len(todo)} / {len(base['items'])} to do", flush=True)
        cache, t1 = {}, time.time()
        for n, it in enumerate(todo, 1):
            st = it["clip"]
            if st not in cache:
                w, sr = sf.read(str(LV.SPLITS[split]["wav"] / f"{st}.wav"), dtype="float32")
                assert sr == SR and w.ndim == 1, (st, sr, w.shape)
                cache.clear(); cache[st] = w
            w = cache[st]
            a, b = it["win"]
            i, j = int(a * SR), int(b * SR)
            seg = w[i:max(j, i + SR)]                                  # listener_variants cut(), unchanged
            txt = gen(seg, Q.format(it["family"].lower()), NEW)
            d["answers"][skey(it)] = parse(txt, len(seg) / SR, a)
            if n <= 3 or n == 10:
                print(f"[{model}] {split} {n} {st} {it['family']} win {a:.2f}-{b:.2f}: {txt!r} ({time.time() - t1:.0f} s)",
                      flush=True)
            if n % 100 == 0:
                C.dump(outp, d)
                print(f"[{model}] {split} {n}/{len(todo)} ({time.time() - t1:.0f} s)", flush=True)
        d["_meta"]["seconds_" + split] = round(time.time() - t1, 1)
        C.dump(outp, d)
        A = list(d["answers"].values())
        print(f"[{model}] {split} done -> {outp}: {len(A)} answers, number {sum(not x['none'] for x in A)}, "
              f"out of window {sum(x['out_of_window'] for x in A)}", flush=True)


def merge():
    S.load_gold = _no_gold
    for split in SPLITS:
        base = json.loads(ITEMS[split].read_text(encoding="utf-8"))
        parts = {m: json.loads(PART[(split, m)].read_text(encoding="utf-8")) for m in ("qwen", "afn")}
        items = []
        for x in base["items"]:
            k = skey(x)
            assert all(k in parts[m]["answers"] for m in parts), (split, k)
            items.append(dict(x, **{m: parts[m]["answers"][k] for m in parts}))
        meta = dict(base["_meta"], models={m: {kk: v for kk, v in parts[m]["_meta"].items() if kk not in base["_meta"]}
                                           for m in parts},
                    fields="per model: text (raw), t_rel (first number, window seconds), t_clip (win_start + t_rel), none "
                    "(no number), says_none, out_of_window", key="(clip, family, start, end)")
        C.dump(OUT[split], {"_meta": meta, "items": items})
        both = sum(1 for x in items if not x["qwen"]["none"] and not x["afn"]["none"])
        agree = sum(1 for x in items if not x["qwen"]["none"] and not x["afn"]["none"]
                    and abs(x["qwen"]["t_rel"] - x["afn"]["t_rel"]) <= 1.0)
        print(f"[merge] {split}: {len(items)} items; qwen number {sum(not x['qwen']['none'] for x in items)}, afn number "
              f"{sum(not x['afn']['none'] for x in items)}, both {both}, within 1 s {agree} -> {OUT[split]}", flush=True)


# ============================================================================= DEV screen (DEV gold only)
IMPULSIVE = ("Gunshot", "Gasp", "Explosion", "Knock", "Hammer", "Glass", "Door", "Whack", "Clang", "Slam")
NMS_S, MULT, MED_FLOOR = 0.5, 3.0, 0.05


def impulsive(label):
    names = {label.lower(), canonical(label).lower()}
    return any(re.search(r"\b" + w.lower() + r"\b", n) for n in names for w in IMPULSIVE)


def motion_peaks(z):
    """local maxima of energy (largest within +-0.5 s) that are >= 3 x the clip median (median floored at 0.05)"""
    e, t, fps = z["energy"].astype(np.float64), z["times"], float(z["fps"])
    med = max(float(np.median(e)), MED_FLOOR) if len(e) else MED_FLOOR
    r = int(round(NMS_S * fps))
    pk = [i for i in range(len(e)) if e[i] >= MULT * med and e[i] == e[max(0, i - r):i + r + 1].max()
          and (i == 0 or e[i] > e[i - 1])]
    return t[pk], e, t, med


def screen():
    stems = sorted(x.strip() for x in (GOLD / "dev_stems.txt").read_text(encoding="utf-8").splitlines() if x.strip())
    assert len(stems) == 49
    allg = S.load_gold([GOLD / "annotations" / "gold_AG.json"])
    gold = {st: allg[st] for st in stems}                             # DEV only; the full dict is not used after this line
    del allg
    need = {st: [g for g in gold[st] if g["needed"] and g["importance"] >= S.MIN_IMPORTANCE] for st in stems}
    E, L = S.EARLY, S.LATE
    # ---- J3
    mot = {st: np.load(WORK / "motion_dev" / f"{st}.npz") for st in stems}
    pk = {st: motion_peaks(mot[st]) for st in stems}
    rows = []
    for st in stems:
        ptimes, e, t, med = pk[st]
        for g in need[st]:
            if not impulsive(g["label"]):
                continue
            on = float(g["start"])
            inwin = [float(p) for p in ptimes if on - E <= p <= on + L]
            anyf = bool(((t >= on - E) & (t <= on + L) & (e >= MULT * med)).any())
            rows.append({"clip": st, "label": g["label"], "onset": on, "peak_in_window": bool(inwin), "peaks_in_window": inwin,
                         "any_frame_3x_in_window": anyf, "clip_median": med, "clip_peaks": int(len(ptimes))})
    per_clip = {}
    for st in stems:
        ptimes, e, t, med = pk[st]
        dur = float(t[-1]) if len(t) else 0.0
        grid = np.arange(0.0, dur, 0.04)
        cov = float(np.mean([any(g - E <= p <= g + L for p in ptimes) for g in grid])) if len(grid) else 0.0
        per_clip[st] = {"peaks": int(len(ptimes)), "per_10s": 10 * len(ptimes) / dur if dur else 0.0, "median": med,
                        "raw_median": float(np.median(e)) if len(e) else 0.0, "duration": dur,
                        "chance_window_has_peak": cov}
    npk = np.array([v["peaks"] for v in per_clip.values()])
    j3 = {"rule": f"motion peak = local max (largest within +-{NMS_S} s, rising) of energy >= {MULT} x clip median "
                  f"(median floored at {MED_FLOOR}); hit window [onset - {E}, onset + {L}]; needed = gold needed and importance "
                  f">= {S.MIN_IMPORTANCE}; impulsive = label or canonical family contains one of {list(IMPULSIVE)} (word)",
          "n_sounds": len(rows), "peak_in_window": sum(r["peak_in_window"] for r in rows),
          "any_frame_3x_in_window": sum(r["any_frame_3x_in_window"] for r in rows),
          "chance_same_clips": float(np.mean([per_clip[r["clip"]]["chance_window_has_peak"] for r in rows])) if rows else None,
          "peaks_per_clip": {"mean": float(npk.mean()), "median": float(np.median(npk)), "max": int(npk.max()),
                             "min": int(npk.min()), "per_10s_mean": float(np.mean([v["per_10s"] for v in per_clip.values()]))},
          "chance_window_has_peak_mean_all_clips": float(np.mean([v["chance_window_has_peak"] for v in per_clip.values()])),
          "clips_raw_median_below_floor": [st for st, v in per_clip.items() if v["raw_median"] < MED_FLOOR],
          "sounds": rows, "per_clip": per_clip}
    # ---- J4
    ts = json.loads(OUT["dev"].read_text(encoding="utf-8"))["items"]
    runs = []
    for x in ts:
        ov = [g for g in need[x["clip"]] if S.same_family(x["label"], g["label"])
              and min(x["end"], float(g["end"])) - max(x["start"], float(g["start"])) > 0]
        if not ov:
            continue
        ons = [float(g["start"]) for g in ov]
        hit = lambda s: s is not None and any(S.in_window(s, o, E, L) for o in ons)
        q, a = x["qwen"], x["afn"]
        both = not q["none"] and not a["none"]
        agree = both and abs(q["t_clip"] - a["t_clip"]) <= 1.0
        mean = (q["t_clip"] + a["t_clip"]) / 2 if agree else None
        eff = x["start"] if x["run_len"] >= 0.5 else x["cut_start"]   # the harness' effective picture start for a run
        runs.append({"clip": x["clip"], "family": x["family"], "label": x["label"], "pools": x["pools"], "start": x["start"],
                     "end": x["end"], "eff_start": eff, "onsets": ons, "qwen": q["t_clip"], "afn": a["t_clip"],
                     "qwen_text": q["text"], "afn_text": a["text"], "both_number": both, "agree_1s": agree, "mean": mean,
                     "mean_hit": hit(mean), "qwen_hit": hit(q["t_clip"]), "afn_hit": hit(a["t_clip"]), "start_hit": hit(eff),
                     "rule_start": mean if agree else eff, "rule_hit": hit(mean if agree else eff)})
    n = len(runs)
    ag = [r for r in runs if r["agree_1s"]]
    j4 = {"rule": "runs (J4 items) of the same family as a DEV needed gold sound (score_per_sound.same_family) whose "
                  "[start, end] overlaps it; hit = inside [onset - 0.5, onset + 1.0] of any overlapped sound; run start = "
                  "start if run_len >= 0.5 else cut_start (the harness' effective start); rule = mean if both within 1 s else run start",
          "n_runs": n, "qwen_number": sum(r["qwen"] is not None for r in runs), "afn_number": sum(r["afn"] is not None for r in runs),
          "both_number": sum(r["both_number"] for r in runs), "agree_1s": len(ag),
          "agree_mean_hit": sum(r["mean_hit"] for r in ag), "agree_start_hit": sum(r["start_hit"] for r in ag),
          "all_start_hit": sum(r["start_hit"] for r in runs), "all_rule_hit": sum(r["rule_hit"] for r in runs),
          "qwen_hit": sum(r["qwen_hit"] for r in runs), "afn_hit": sum(r["afn_hit"] for r in runs),
          "distinct_sounds_start_hit": len({(r["clip"], o) for r in runs if r["start_hit"] for o in r["onsets"]}),
          "runs": runs}
    # false-alarm side of J4: agreement rate on runs that overlap no gold sound of their family (the answers still move a picture)
    other = [x for x in ts if not any(S.same_family(x["label"], g["label"]) and min(x["end"], float(g["end"])) -
                                      max(x["start"], float(g["start"])) > 0 for g in gold[x["clip"]])]
    j4["no_gold_runs"] = {"n": len(other), "both_number": sum(not x["qwen"]["none"] and not x["afn"]["none"] for x in other),
                          "agree_1s": sum(not x["qwen"]["none"] and not x["afn"]["none"] and
                                          abs(x["qwen"]["t_clip"] - x["afn"]["t_clip"]) <= 1.0 for x in other)}
    C.dump(SCREEN, {"_meta": {"amendment": "round 14 J (DEV screen)", "split": "DEV 49 clips", "gold": "gold_AG.json, DEV stems only"},
                    "J3": j3, "J4": j4})
    print(f"[J3] impulsive needed sounds {j3['n_sounds']}: motion peak in window {j3['peak_in_window']}, any 3x frame "
          f"{j3['any_frame_3x_in_window']}; chance on the same clips {j3['chance_same_clips']}; peaks/clip {j3['peaks_per_clip']}; "
          f"chance all clips {j3['chance_window_has_peak_mean_all_clips']:.2f}; low-median clips {j3['clips_raw_median_below_floor']}")
    for r in rows:
        print(f"   {r['clip']} {r['label']} {r['onset']:.2f}: in {r['peak_in_window']} {r['peaks_in_window']} "
              f"(clip peaks {r['clip_peaks']}, median {r['clip_median']:.2f})")
    print(f"[J4] runs overlapping needed {n}: qwen number {j4['qwen_number']}, afn number {j4['afn_number']}, both "
          f"{j4['both_number']}, within 1 s {j4['agree_1s']}; of those mean hit {j4['agree_mean_hit']} vs start hit "
          f"{j4['agree_start_hit']}; all runs start hit {j4['all_start_hit']} vs rule hit {j4['all_rule_hit']}; qwen hit "
          f"{j4['qwen_hit']}, afn hit {j4['afn_hit']}; no-gold runs {j4['no_gold_runs']}")
    for r in runs:
        print(f"   {r['clip']} {r['label']} run {r['start']:.2f}-{r['end']:.2f} on {r['onsets']}: q {r['qwen_text']!r} "
              f"a {r['afn_text']!r} mean {r['mean']} hit {r['mean_hit']} start_hit {r['start_hit']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("pool", "run", "merge", "screen"))
    ap.add_argument("--model", choices=("qwen", "afn"))
    a = ap.parse_args()
    if a.step == "pool":
        pool()
    elif a.step == "run":
        run(a.model)
    elif a.step == "merge":
        merge()
    else:
        screen()


if __name__ == "__main__":
    main()
