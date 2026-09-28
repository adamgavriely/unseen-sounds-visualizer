"""Setup audit of the stage-4 detector tests (2026-09-28). Looks for BUGS in the test setup, not for new detector ideas.

Nine rounds of detector ideas failed on the held-out 415. Before a tenth, this checks the parts every round shares:
the downloaded audio, the gold labels, the scorer, the harness stack vs the real stage-4 code, and BEATs' time stamps.

    python benchmark/setup_audit.py probe      # 280 / 415 / fresh: ffprobe + decoded-audio stats (no scores)
    python benchmark/setup_audit.py align      # audio onset vs gold onset at isolated sharp events (280 / 415 / fresh;
                                               #   audio and gold only, no detector output, no cost)
    python benchmark/setup_audit.py gold       # MID -> name mapping, consequential rule, runs (280 / 415 gold only)
    python benchmark/setup_audit.py scorer     # clip_cost sanity on the 280 with gold-made predictions
    python benchmark/setup_audit.py caches     # cluster, CPU: BEATs stamps, the bark-run check, baseline re-scores (280 ONLY)
    python benchmark/setup_audit.py pipeline --stage4-file F --n 20   # cluster, GPU: real detect_events vs harness (280)

Every step merges its keys into benchmark/setup_audit.json. The fresh set is never scored (no cost, no recall, no
detector output is read for it). Nothing on DEV/TEST.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.labels import canonical, is_salient_nonspeech, is_music, ancestors, GENERIC_LABELS
from src.types import AudioEvent

OUT = _ROOT / "benchmark" / "setup_audit.json"
GOLD = {s: _ROOT / "benchmark" / "gold" / f"audioset_{s}.json" for s in ("calib", "heldout", "fresh")}
VID = {s: _ROOT / "data" / "input" / f"audioset_{s}" for s in ("calib", "heldout", "fresh")}
LABELS = _ROOT / "data" / "audioset_strong_labels"
SEED = 0

# impulsive gold labels (strong-set display names): a clear energy / spectral-flux jump at the onset
SHARP = {"Gunshot, gunfire", "Bark", "Yip", "Bow-wow", "Slam", "Knock", "Tap", "Glass shatter", "Shatter", "Breaking",
         "Smash, crash", "Explosion", "Bang", "Firecracker", "Cap gun", "Hammer", "Chop", "Chopping (food)", "Thump, thud",
         "Clang", "Ding", "Crack", "Whip", "Slap, smack", "Wood block", "Snap", "Cough", "Sneeze", "Glass chink, clink",
         "Chink, clink", "Door", "Clapping", "Camera", "Cupboard open or close", "Drawer open or close", "Dishes, pots, and pans",
         "Cutlery, silverware", "Stomp, stamp", "Bouncing", "Burst, pop", "Squeak", "Tick", "Beep, bleep", "Honk",
         "Vehicle horn, car horn, honking, toot", "Doorbell", "Ding-dong", "Bicycle bell", "Keys jangling", "Coin (dropping)",
         "Hiccup", "Meow", "Caw", "Quack", "Cluck", "Oink", "Moo", "Bleat", "Neigh, whinny", "Hoot", "Crow"}


def write(key, val):
    d = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    d[key] = val
    OUT.write_text(json.dumps(d, indent=1, default=float), encoding="utf-8")


def gold(set_name):
    return json.loads(GOLD[set_name].read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- names
def strong_names():
    with (LABELS / "mid_to_display_name.tsv").open(encoding="utf-8") as f:
        return {mid: name for mid, name in csv.reader(f, delimiter="\t")}


def ontology():
    o = json.loads((_ROOT / "src" / "audioset_ontology.json").read_text(encoding="utf-8"))
    byid = {x["id"]: x["name"] for x in o}
    par = defaultdict(list)
    for x in o:
        for c in x["child_ids"]:
            par[byid[c]].append(x["name"])
    return byid, dict(par)


def strong_to_beats():
    """strong-set display name -> the name BEATs (and the ontology) uses for the SAME MID; only where they differ"""
    beats = json.loads((_ROOT / "src" / "audioset_mid_names.json").read_text(encoding="utf-8"))
    return {n: beats[m] for m, n in strong_names().items() if m in beats and beats[m] != n}


def remap_clip(c, m):
    """the clip with gold names remapped by MID to BEATs' names; consequential recomputed with the builders' own rule"""
    from benchmark.gold.audioset_slice import CONSEQUENTIAL
    c = copy.deepcopy(c)
    for g in c["events"]:
        if g["label"] in m:
            g["label"] = m[g["label"]]
            g["consequential"] = bool(g["label"] in CONSEQUENTIAL or canonical(g["label"]) in CONSEQUENTIAL)
    return c


_ANC_ALL = {}


def anc_all(label, par):
    if label not in _ANC_ALL:
        out, todo = set(), list(par.get(label, []))
        while todo:
            p = todo.pop()
            if p not in out:
                out.add(p); todo += par.get(p, [])
        _ANC_ALL[label] = out
    return _ANC_ALL[label]


def same_multi(par):
    def f(a, b):
        return a == b or canonical(a) == canonical(b) or b in anc_all(a, par) or a in anc_all(b, par)
    return f


# ----------------------------------------------------------------------------- the scorer (a copy of D.clip_cost with hooks)
def clip_cost_v(c, ev, same=None, early=0.5, late=1.0, gold_events=None):
    """detector_round2.clip_cost with the family test and the gold list swappable; identical when both are None"""
    from benchmark import audioset_detector_eval as E
    same = same or E._same
    ok = lambda lab: is_salient_nonspeech(lab) and not is_music(lab)
    ev = [e for e in ev if ok(e.label)]
    gl = gold_events if gold_events is not None else c["events"]
    g_c = [g for g in gl if ok(g["label"]) and g["consequential"]]
    miss_ov = sum(not any(same(e.label, g["label"]) and E._overlap_ok(e.start, e.end, g["start"], g["end"]) for e in ev) for g in g_c)
    miss_on = sum(not any(same(e.label, g["label"]) and g["start"] - early <= e.start <= g["start"] + late for e in ev) for g in g_c)
    fp = sum(not any(same(e.label, g["label"]) and min(e.end, g["end"]) - max(e.start, g["start"]) > 0 for g in c["events"]) for e in ev)
    return {"C_overlap": 4 * miss_ov + 2 * fp, "C_onset": 4 * miss_on + 2 * fp, "fp": fp, "miss_overlap": miss_ov,
            "miss_onset": miss_on, "n_conseq": len(g_c)}


def summ(rows, n_clips):
    n = max(1, sum(r["n_conseq"] for r in rows))
    return {"C_overlap": float(np.mean([r["C_overlap"] for r in rows])), "C_onset": float(np.mean([r["C_onset"] for r in rows])),
            "recall_overlap": 1 - sum(r["miss_overlap"] for r in rows) / n, "recall_onset": 1 - sum(r["miss_onset"] for r in rows) / n,
            "fp": int(sum(r["fp"] for r in rows)), "fp_per_min": sum(r["fp"] for r in rows) / (n_clips * 10 / 60.0),
            "n_conseq": int(sum(r["n_conseq"] for r in rows))}


def merged_runs(c, gap=2.0):
    """consequential gold merged per family when the pause is <= gap (the project's own annotation rule: one row per
    continuous sound, split at pauses > 2 s); non-consequential events unchanged"""
    out = [g for g in c["events"] if not g["consequential"]]
    fam = defaultdict(list)
    for g in c["events"]:
        if g["consequential"]:
            fam[canonical(g["label"])].append(dict(g))
    for f, gs in fam.items():
        gs.sort(key=lambda g: g["start"])
        cur = gs[0]
        for g in gs[1:]:
            if g["start"] - cur["end"] <= gap:
                cur["end"] = max(cur["end"], g["end"]); cur["masked"] = cur["masked"] or g["masked"]
            else:
                out.append(cur); cur = g
        out.append(cur)
    return out


# ----------------------------------------------------------------------------- 1a. probe
def ffprobe(p):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration,start_time:stream=codec_type,codec_name,start_time,duration,sample_rate,channels,height",
                        "-of", "json", str(p)], capture_output=True, text=True)
    return json.loads(r.stdout or "{}")


def decode(p, sr=16000):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(p), "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
                       capture_output=True)
    return np.frombuffer(r.stdout, np.float32)


def probe(a):
    res = {}
    for s in ("calib", "heldout", "fresh"):
        d = gold(s)
        ids = [c["id"] for c in d["clips"]]
        on_disk = sorted(p.stem for p in VID[s].glob("*.mp4"))
        rows, flags = [], defaultdict(list)
        for cid in ids:
            p = VID[s] / f"{cid}.mp4"
            if not p.exists():
                flags["missing_file"].append(cid); continue
            j = ffprobe(p)
            st = {x["codec_type"]: x for x in j.get("streams", [])}
            au, vi = st.get("audio", {}), st.get("video", {})
            x = decode(p)
            rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
            row = {"id": cid, "dur": float(j.get("format", {}).get("duration", 0)), "a_start": float(au.get("start_time", "nan")),
                   "v_start": float(vi.get("start_time", "nan")) if vi else None, "a_dur": float(au.get("duration", "nan")),
                   "v_dur": float(vi.get("duration", "nan")) if vi else None, "sr": int(au.get("sample_rate", 0)),
                   "ch": int(au.get("channels", 0)), "codec": au.get("codec_name"), "height": vi.get("height"),
                   "decoded_s": len(x) / 16000.0, "rms_db": 20 * np.log10(rms + 1e-12),
                   "silent_frac": float((np.abs(x) < 1e-4).mean()) if len(x) else 1.0}
            rows.append(row)
            if not audio_ok(row):
                flags["no_audio_stream"].append(cid)
            if row["decoded_s"] < 9.9:
                flags["shorter_than_9.9s"].append((cid, round(row["decoded_s"], 3)))
            if row["decoded_s"] > 10.1:
                flags["longer_than_10.1s"].append((cid, round(row["decoded_s"], 3)))
            if abs(row["a_start"]) > 0.01:
                flags["audio_start_not_0"].append((cid, row["a_start"]))
            if row["v_start"] is not None and abs(row["v_start"]) > 0.1:
                flags["video_start_over_0.1s"].append((cid, row["v_start"]))
            # digital silence (0.1-s blocks, |x| < 1e-4) inside labelled sound: a muted / broken audio track
            c = next(cc for cc in d["clips"] if cc["id"] == cid)
            blk = x[: len(x) // 1600 * 1600].reshape(-1, 1600)
            sil = np.abs(blk).max(axis=1) < 1e-4
            tt = np.arange(len(sil)) * 0.1
            lab = np.zeros(len(sil), bool)
            for g in c["events"]:
                if g["label"] != "Silence":
                    lab |= (tt >= g["start"]) & (tt + 0.1 <= g["end"])
            if (sil & lab).sum() >= 10:
                flags["muted_audio_under_labelled_sound_ge_1s"].append((cid, round(float((sil & lab).sum()) * 0.1, 1)))
            if row["decoded_s"] < 9.9 and max((g["end"] for g in c["events"]), default=0) > row["decoded_s"] + 0.5:
                flags["gold_beyond_audio_end_by_0.5s"].append((cid, row["decoded_s"], max(g["end"] for g in c["events"])))
            if row["v_dur"] is not None and abs(row["a_dur"] - row["v_dur"]) > 0.1:
                flags["audio_video_len_differ"].append((cid, row["a_dur"], row["v_dur"]))
            if row["rms_db"] < -60:
                flags["near_silent_rms_below_-60dB"].append((cid, round(row["rms_db"], 1)))
            if row["silent_frac"] > 0.5:
                flags["over_half_digital_silence"].append((cid, round(row["silent_frac"], 2)))
        dur = np.array([r["decoded_s"] for r in rows])
        ytids = Counter(i.rsplit("_", 1)[0] for i in ids)
        res[s] = {"clips_in_json": len(ids), "files_on_disk": len(on_disk), "json_ids_unique": len(set(ids)) == len(ids),
                  "files_not_in_json": sorted(set(on_disk) - set(ids)), "duplicate_youtube_ids": {k: v for k, v in ytids.items() if v > 1},
                  "decoded_seconds": {"min": float(dur.min()), "median": float(np.median(dur)), "max": float(dur.max())},
                  "sample_rates": dict(Counter(r["sr"] for r in rows)), "channels": dict(Counter(r["ch"] for r in rows)),
                  "codecs": dict(Counter(r["codec"] for r in rows)), "video_heights": dict(Counter(r["height"] for r in rows)),
                  "rms_db": {"min": float(min(r["rms_db"] for r in rows)), "median": float(np.median([r["rms_db"] for r in rows]))},
                  "flags": {k: v for k, v in flags.items()}, "flag_counts": {k: len(v) for k, v in flags.items()}}
        print(f"[probe] {s}: {res[s]['clips_in_json']} clips, dur {res[s]['decoded_seconds']}, sr {res[s]['sample_rates']}, "
              f"ch {res[s]['channels']}, flags {res[s]['flag_counts']}", flush=True)
    cross = {}
    yt = {s: {c["id"].rsplit("_", 1)[0] for c in gold(s)["clips"]} for s in res}
    seg = {s: {c["id"] for c in gold(s)["clips"]} for s in res}
    for a_, b_ in (("calib", "heldout"), ("calib", "fresh"), ("heldout", "fresh")):
        cross[f"{a_}&{b_}"] = {"same_segment": sorted(seg[a_] & seg[b_]), "same_youtube_id": sorted(yt[a_] & yt[b_])}
    res["cross_set_overlap"] = cross
    print(f"[probe] cross-set overlap: { {k: (len(v['same_segment']), len(v['same_youtube_id'])) for k, v in cross.items()} }")
    fl = (_ROOT / "benchmark" / "gold" / "audioset_slice.py").read_text(encoding="utf-8")
    res["download_command"] = {"force_keyframes_at_cuts": "--force-keyframes-at-cuts" in fl,
                               "sections": "*{start:.1f}-{start + 10:.1f}" in fl,
                               "note": "one fetch() in audioset_slice.py serves slice B, the 280, the 415 and fresh"}
    write("probe", res)


def audio_ok(row):
    return row["sr"] > 0 and row["ch"] > 0


# ----------------------------------------------------------------------------- 1b. align
def onset_env(x, sr=16000, hop=160):
    import librosa
    env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop, n_fft=1024)
    env = env / (np.median(env) + 1e-9)
    return env, sr / hop


def isolated_sharp(c, win=0.5, max_len=1.5):
    """sharp gold events (start in [1, 9] s, <= 1.5 s long) with no OTHER gold event starting within +-win of the onset"""
    ev = c["events"]
    out = []
    for i, g in enumerate(ev):
        if g["label"] not in SHARP or g["end"] - g["start"] > max_len or g["start"] < 1.0 or g["start"] > 9.0:
            continue
        if any(j != i and abs(h["start"] - g["start"]) < win for j, h in enumerate(ev)):
            continue
        out.append(g)
    return out


def family_isolated(c, pad=2.5, max_len=3.0):
    """salient gold events (<= 3 s, inside [2, 8] s) with no other event of the same family within [start - pad, end + pad]
    (other sounds allowed): the family's BEATs curve near it can only come from this event"""
    out = []
    for i, g in enumerate(c["events"]):
        if not (is_salient_nonspeech(g["label"]) and not is_music(g["label"])) or g["end"] - g["start"] > max_len                 or g["start"] < 2.0 or g["end"] > 8.0 or g["label"] in GENERIC_LABELS:
            continue
        if any(j != i and canonical(h["label"]) == canonical(g["label"]) and h["start"] < g["end"] + pad and h["end"] > g["start"] - pad
               for j, h in enumerate(c["events"])):
            continue
        out.append(g)
    return out


def lag_at(env, fps, t, rng=1.0):
    a, b = int(round((t - rng) * fps)), int(round((t + rng) * fps))
    a, b = max(0, a), min(len(env), b + 1)
    if b - a < 5:
        return None
    return (a + int(np.argmax(env[a:b]))) / fps - t


def clip_lag(env, fps, onsets, rng=3.0):
    """one lag per clip: the shift that puts the most onset strength (max within +-20 ms) at the clip's gold onsets;
    z = how far that best shift stands above the other shifts (robust z-score)"""
    lags = np.arange(-rng, rng + 1e-9, 0.01)
    sc = []
    for L in lags:
        v = []
        for t in onsets:
            i = int(round((t + L) * fps))
            v.append(env[max(0, i - 2): i + 3].max() if 0 <= i < len(env) else 0.0)
        sc.append(np.mean(v))
    sc = np.array(sc)
    med = np.median(sc); mad = 1.4826 * np.median(np.abs(sc - med)) + 1e-9
    return float(lags[int(np.argmax(sc))]), float((sc.max() - med) / mad)


def align(a):
    rng = np.random.default_rng(SEED)
    res = {}
    for s in ("calib", "heldout", "fresh"):
        d = gold(s)
        ev_lag, ev_ctrl, per_clip = [], [], []
        for c in d["clips"]:
            x = decode(VID[s] / f"{c['id']}.mp4")
            env, fps = onset_env(x)
            for g in isolated_sharp(c):
                L = lag_at(env, fps, g["start"])
                if L is not None:
                    ev_lag.append({"id": c["id"], "label": g["label"], "onset": g["start"], "lag": round(L, 3)})
                Lc = lag_at(env, fps, float(rng.uniform(1.0, 9.0)))          # control: a random time in the same clip
                if Lc is not None:
                    ev_ctrl.append(Lc)
            # every labelled onset (AudioSet-Strong is exhaustive), except events that start the clip
            ons = sorted({g["start"] for g in c["events"] if 0.3 < g["start"] < min(9.7, len(x) / 16000 - 0.3)})
            if len(ons) >= 2:
                L, z = clip_lag(env, fps, ons)
                per_clip.append({"id": c["id"], "n_onsets": len(ons), "lag": round(L, 2), "z": round(z, 1)})
        lags = np.array([e["lag"] for e in ev_lag]); ctrl = np.array(ev_ctrl)
        conf = [p for p in per_clip if p["z"] >= 4]
        cl = np.array([p["lag"] for p in conf])
        within = lambda v, t: float((np.abs(v) <= t).mean()) if len(v) else None
        res[s] = {"sharp_events": {"n": len(lags), "clips": len({e["id"] for e in ev_lag}),
                                   "lag_median": float(np.median(lags)) if len(lags) else None,
                                   "lag_q1_q3": [float(np.percentile(lags, 25)), float(np.percentile(lags, 75))] if len(lags) else None,
                                   "within_0.1s": within(lags, 0.1), "within_0.25s": within(lags, 0.25),
                                   "control_random_time_within_0.1s": within(ctrl, 0.1), "control_within_0.25s": within(ctrl, 0.25)},
                  "clip_lag_all_onsets": {"clips_scored": len(per_clip), "clips_confident_z_ge_4": len(conf),
                                          "median": float(np.median(cl)) if len(cl) else None,
                                          "q1_q3": [float(np.percentile(cl, 25)), float(np.percentile(cl, 75))] if len(cl) else None,
                                          "within_0.1s": within(cl, 0.1), "within_0.25s": within(cl, 0.25),
                                          "confident_abs_over_0.5s": sorted([p for p in conf if abs(p["lag"]) > 0.5], key=lambda p: -p["z"])},
                  "events_detail": ev_lag if s == "calib" else None}
        r = res[s]
        print(f"[align] {s}: sharp events n={r['sharp_events']['n']} in {r['sharp_events']['clips']} clips, lag median "
              f"{r['sharp_events']['lag_median']}, within 0.1 s {r['sharp_events']['within_0.1s']} (random-time control "
              f"{r['sharp_events']['control_random_time_within_0.1s']}); clip lag (z>=4) {r['clip_lag_all_onsets']['clips_confident_z_ge_4']}"
              f"/{len(per_clip)} clips, median {r['clip_lag_all_onsets']['median']}, within 0.1 s {r['clip_lag_all_onsets']['within_0.1s']}, "
              f"|lag|>0.5 s: {len(r['clip_lag_all_onsets']['confident_abs_over_0.5s'])}", flush=True)
    write("align", res)


# ----------------------------------------------------------------------------- 2. gold mapping
def gold_check(a):
    from benchmark.gold.audioset_slice import CONSEQUENTIAL
    names = strong_names()
    byid, par = ontology()
    beats = json.loads((_ROOT / "src" / "audioset_mid_names.json").read_text(encoding="utf-8"))
    m = strong_to_beats()
    onto_names = set(byid.values())
    not_in_onto = sorted(n for mid, n in names.items() if mid not in byid)
    res = {"strong_classes": len(names), "renamed_vs_beats": m, "strong_only_labels_no_ontology": not_in_onto,
           "consequential_list_not_in_ontology": sorted(x for x in CONSEQUENTIAL if x not in onto_names),
           "consequential_list_not_in_strong_vocab": sorted(x for x in CONSEQUENTIAL if x not in set(names.values()))}
    for s in ("calib", "heldout"):
        d = gold(s)
        ev = [g for c in d["clips"] for g in c["events"]]
        lab = Counter(g["label"] for g in ev)
        con = Counter(g["label"] for g in ev if g["consequential"])
        per_clip = [sum(g["consequential"] for g in c["events"]) for c in d["clips"]]
        # the events the harness actually scores (salient, non-music, consequential)
        scored = [(c["id"], g) for c in d["clips"] for g in c["events"]
                  if g["consequential"] and is_salient_nonspeech(g["label"]) and not is_music(g["label"])]
        sc_clip = Counter(cid for cid, _ in scored)
        top = sc_clip.most_common(10)
        lost = {n: lab[n] for n in m if lab[n]}
        lost_conseq = {n: lab[n] for n in m if lab[n] and (m[n] in CONSEQUENTIAL or canonical(m[n]) in CONSEQUENTIAL)}
        # runs: consequential events of one family with a pause <= 2 s before the next (AudioSet-Strong labels each bark)
        n_run, clips_run = 0, set()
        for c in d["clips"]:
            fam = defaultdict(list)
            for g in c["events"]:
                if g["consequential"] and is_salient_nonspeech(g["label"]) and not is_music(g["label"]):
                    fam[canonical(g["label"])].append(g)
            for gs in fam.values():
                gs.sort(key=lambda g: g["start"])
                for p, q in zip(gs, gs[1:]):
                    if q["start"] - p["end"] <= 2.0:
                        n_run += 1; clips_run.add(c["id"])
        merged = sum(sum(1 for g in merged_runs(c) if g["consequential"] and is_salient_nonspeech(g["label"]) and not is_music(g["label"]))
                     for c in d["clips"])
        # consequential by family mapping only (the name itself is not in the list): what the rule sweeps in
        via_family = Counter(g["label"] for g in ev if g["consequential"] and g["label"] not in CONSEQUENTIAL)
        # consequential labels the scorer never keeps (filtered as non-salient) or BEATs cannot name
        filtered = Counter(g["label"] for g in ev if g["consequential"] and not (is_salient_nonspeech(g["label"]) and not is_music(g["label"])))
        not_beats = Counter(g["label"] for g in ev if g["consequential"] and g["label"] not in set(beats.values()))
        zero_len = [(c["id"], g["label"], g["start"]) for c in d["clips"] for g in c["events"] if g["end"] <= g["start"]]
        out_rng = [(c["id"], g["label"], g["start"], g["end"]) for c in d["clips"] for g in c["events"] if g["start"] < 0 or g["end"] > 10.0 + 1e-6]
        res[s] = {"clips": len(d["clips"]), "events": len(ev), "consequential_flagged": sum(con.values()),
                  "consequential_scored": len(scored), "clips_with_no_scored_consequential": sum(1 for c in d["clips"] if not sc_clip[c["id"]]),
                  "clips_with_no_gold_events": sum(1 for c in d["clips"] if not c["events"]),
                  "top10_clips_by_scored_consequential": top, "share_of_scored_in_top10_clips": sum(v for _, v in top) / max(1, len(scored)),
                  "consequential_by_label": dict(con.most_common()), "consequential_via_family_only": dict(via_family.most_common()),
                  "consequential_but_filtered_as_not_salient": dict(filtered), "consequential_label_not_in_beats_vocab": dict(not_beats),
                  "renamed_label_events": lost, "renamed_events_that_should_be_consequential": lost_conseq,
                  "strong_only_label_events": {n: lab[n] for n in not_in_onto if lab[n]},
                  "scored_consequential_in_runs_gap_le_2s": n_run, "clips_with_runs": len(clips_run),
                  "scored_consequential_after_merging_runs": merged, "zero_length_events": zero_len, "events_outside_0_10": out_rng,
                  "multi_parent_label_events": {k: lab[k] for k in par if len(par[k]) > 1 and lab[k]}}
        print(f"[gold] {s}: {len(ev)} events, {len(scored)} scored consequential in {len(d['clips']) - res[s]['clips_with_no_scored_consequential']} "
              f"clips; top-10 clips hold {res[s]['share_of_scored_in_top10_clips']:.0%}; runs {n_run} (merged -> {merged}); "
              f"renamed {lost}; should-be-consequential {lost_conseq}", flush=True)
    write("gold", res)


# ----------------------------------------------------------------------------- 3. scorer sanity (gold-made predictions, 280)
def as_events(gl, shift=0.0, label=lambda l: l):
    return [AudioEvent(label(g["label"]), min(10.0, max(0.0, g["start"] + shift)), min(10.0, max(0.0, g["end"] + shift)), 1.0) for g in gl]


def scorer(a):
    from benchmark import detector_round2 as D
    d = gold("calib")
    cl = d["clips"]
    byid, par = ontology()
    m = strong_to_beats()
    ok = lambda l: is_salient_nonspeech(l) and not is_music(l)
    rng = np.random.default_rng(SEED)
    beats_vocab = sorted(set(json.loads((_ROOT / "src" / "audioset_mid_names.json").read_text(encoding="utf-8")).values()))
    salient_vocab = [l for l in beats_vocab if ok(l)]

    def up(l):
        an = ancestors(l)
        return an[0] if an else l

    def up_nongeneric(l):
        for x in ancestors(l):
            if ok(x):
                return x
        return l
    tests = {
        "gold_all_salient": lambda c: as_events([g for g in c["events"] if ok(g["label"])]),
        "gold_consequential_only": lambda c: as_events([g for g in c["events"] if g["consequential"]]),
        "gold_all_in_beats_names": lambda c: as_events([g for g in c["events"] if ok(g["label"])], label=lambda l: m.get(l, l)),
        "shift_+0.5s": lambda c: as_events([g for g in c["events"] if ok(g["label"])], 0.5),
        "shift_+1.5s": lambda c: as_events([g for g in c["events"] if ok(g["label"])], 1.5),
        "shift_-0.5s": lambda c: as_events([g for g in c["events"] if ok(g["label"])], -0.5),
        "one_family_up": lambda c: as_events([g for g in c["events"] if ok(g["label"])], label=up),
        "one_family_up_skip_generic": lambda c: as_events([g for g in c["events"] if ok(g["label"])], label=up_nongeneric),
        "empty": lambda c: [],
        "random_spans": lambda c: [AudioEvent(salient_vocab[int(rng.integers(len(salient_vocab)))], s, min(10.0, s + float(rng.uniform(0.5, 3))), 1.0)
                                   for s in rng.uniform(0, 9.5, max(1, sum(g["consequential"] for g in c["events"])))],
    }
    res = {}
    for name, fn in tests.items():
        rows = []
        for c in cl:
            ev = fn(c)
            r = D.clip_cost(c, ev)
            r2 = clip_cost_v(c, ev)
            assert r["C_overlap"] == r2["C_overlap"] and r["C_onset"] == r2["C_onset"], "clip_cost_v differs from D.clip_cost"
            rows.append(r)
        S = summ(rows, len(cl))
        bad = [(c["id"], r["fp"], r["miss_overlap"]) for c, r in zip(cl, rows) if r["C_overlap"] > 0][:8]
        res[name] = {**S, "clips_with_C>0": sum(r["C_overlap"] > 0 for r in rows), "examples": bad}
        print(f"[scorer] {name:28s} C-ov {S['C_overlap']:.3f} C-on {S['C_onset']:.3f} rec {S['recall_overlap']:.1%} fp {S['fp']} "
              f"clips C>0 {res[name]['clips_with_C>0']}", flush=True)
    # which gold labels a perfect BEATs-named detector is punished for (FP over a renamed / strong-only gold label)
    fp_labels = Counter()
    for c in cl:
        for e in as_events([g for g in c["events"] if ok(g["label"])], label=lambda l: m.get(l, l)):
            r = clip_cost_v(c, [e])
            if r["fp"]:
                fp_labels[e.label] += 1
    res["gold_all_in_beats_names"]["fp_labels"] = dict(fp_labels)
    # multi-parent: gold events / labels affected
    multi = {k for k, v in par.items() if len(v) > 1}
    affected = Counter(g["label"] for c in cl for g in c["events"] if g["label"] in multi or (anc_all(g["label"], par) & multi))
    res["multi_parent"] = {"labels_with_more_than_one_parent": len(multi), "gold_280_events_on_multi_parent_path": sum(affected.values()),
                           "by_label": dict(affected.most_common()),
                           "one_parent_file_matches_official_first_parent": all(
                               json.loads((_ROOT / "src" / "audioset_parents.json").read_text(encoding="utf-8")).get(k) in par[k] for k in multi)}
    write("scorer", res)


# ----------------------------------------------------------------------------- 4/5. caches (cluster, CPU; 280 only)
def stack_prov(cc, b=0.1218):
    """round-5 run() for BEATs (slot 0) with b fixed, plus provenance: each shown span's BEATs start before the twin rule"""
    from benchmark import detector_round2 as D
    from benchmark import detector_round4 as R4
    from src.stage4_audio_event_detection import _extract_events
    key = lambda e: canonical(e.label)
    tag, f = cc[0], cc[1]
    events = _extract_events(tag[0], tag[1], tag[2], D.AED, None, config.AED_MIN_DUR, low=D.AED * D.HYS)
    orig = {id(e): e.start for e in events}
    fev = _extract_events(f[0], f[1], f[2], R4.FBAR, None, config.AED_MIN_DUR, low=R4.FBAR * D.HYS)
    fresh = []
    for e in fev:
        tw = [x for x in events if key(x) == key(e) and x.start - 1.0 <= e.end and e.start - 1.0 <= x.end]
        if tw:
            for x in tw:
                x.start = min(x.start, e.start)
        else:
            fresh.append(e)
    fids = {id(e) for e in fresh}
    events = events + fresh
    fpk = D.clip_peak(f)
    events = [e for e in events if fpk.get(key(e), 1.0) >= R4.FVETO]
    tpk = D.clip_peak(tag)
    events = [e for e in events if id(e) not in fids or tpk.get(key(e), 1.0) >= b]
    shown = [e for e in events if e.confidence >= D.DISP]
    prov = {id(e): ("flex_only" if id(e) in fids else ("twin_moved" if orig[id(e)] > e.start + 1e-9 else "beats")) for e in shown}
    return shown, prov, orig


def fam_curve(fr, label):
    fw, ts, labs = fr
    cols = [i for i, l in enumerate(labs) if canonical(l) == canonical(label) or l == label]
    return (fw[:, cols].max(axis=1) if cols else np.zeros(len(ts))), ts


def caches(a):
    from benchmark import audioset_stage4_report as R
    from benchmark import detector_round2 as D
    from benchmark import detector_round4 as R4
    from benchmark import audioset_detector_eval as E
    import benchmark.detector_round5 as R5
    R.use_set("calib")
    cl = D.usable()
    assert len(cl) == 280, len(cl)
    cc = [R4.load3(c["id"]) for c in cl]
    res = {"clips": len(cl)}
    # --- baseline reproduction (round 5 gate-1 baseline, the audit doc's 3.036)
    evs5, rows5, S5 = R5.run(cl, cc, 0, D.AED, D.DISP, b=0.1218)
    per = [stack_prov(x) for x in cc]
    sig = lambda ev: sorted((e.label, round(e.start, 6), round(e.end, 6)) for e in ev)
    assert all(sig(p[0]) == sig(e) for p, e in zip(per, evs5)), "stack_prov differs from round-5 run"
    base_rows = [clip_cost_v(c, p[0]) for c, p in zip(cl, per)]
    S0 = summ(base_rows, len(cl))
    assert abs(S0["C_overlap"] - S5["C_overlap"]) < 1e-9 and abs(S0["C_overlap"] - 3.0357142857142856) < 1e-9, (S0, S5)
    res["baseline"] = S0
    print(f"[caches] baseline reproduced: C-ov {S0['C_overlap']:.4f} C-on {S0['C_onset']:.4f} rec {S0['recall_overlap']:.1%}", flush=True)

    # --- 5. BEATs stamps in the caches
    b0 = cc[0][0]
    ts_all = [x[0][1] for x in cc]
    dts = sorted({round(float(np.diff(t).min()), 4) for t in ts_all} | {round(float(np.diff(t).max()), 4) for t in ts_all})
    res["stamps_cache"] = {"first_stamp": sorted({round(float(t[0]), 4) for t in ts_all}), "dt": dts,
                           "n_frames": dict(Counter(len(t) for t in ts_all)), "last_stamp": sorted({round(float(t[-1]), 3) for t in ts_all}),
                           "formula": "stamp = window_end - 0.5 s  ->  a stamp t scores audio [t - 1.5, t + 0.5]",
                           "flexsed_fps": sorted({round(1.0 / float(np.diff(x[1][1][:2])[0]), 3) for x in cc})}
    # empirical: isolated sharp events -> first/last BEATs stamp for the family vs gold onset/end
    first, last, ctl, first_sharp, first_other = [], [], [], [], []
    for c, x in zip(cl, cc):
        for g in family_isolated(c):
            cur, ts = fam_curve(x[0], g["label"])
            if not cur.any():
                continue
            m = (ts >= g["start"] - 2.0) & (ts <= g["end"] + 2.5)
            hit = np.where(m & (cur >= D.DISP))[0]
            if not len(hit):
                continue
            i0 = hit[0]
            # the contiguous >= 0.175 run holding the first display-level stamp
            j0, j1 = i0, i0
            while j0 > 0 and cur[j0 - 1] >= D.AED:
                j0 -= 1
            while j1 + 1 < len(cur) and cur[j1 + 1] >= D.AED:
                j1 += 1
            first.append(float(ts[i0] - g["start"])); ctl.append(float(ts[j0] - g["start"]))
            last.append(float(ts[j1] + 0.25 - g["end"]))
            (first_sharp if g["label"] in SHARP else first_other).append(float(ts[j0] - g["start"]))
    q = lambda v: {"n": len(v), "median": float(np.median(v)) if v else None,
                   "q1_q3": [float(np.percentile(v, 25)), float(np.percentile(v, 75))] if v else None}
    res["stamps_empirical"] = {"first_stamp_ge_0.35_minus_gold_onset": q(first), "span_start_at_0.175_minus_gold_onset": q(ctl),
                               "span_start_minus_onset_sharp_labels": q(first_sharp), "span_start_minus_onset_other_labels": q(first_other),
                               "span_end_minus_gold_end": q(last),
                               "expected_if_stamp_is_window_end_minus_0.5": "start about -0.5..+0.25 s, end about +1.5..+1.75 s"}
    print(f"[caches] stamps {res['stamps_cache']}; empirical {res['stamps_empirical']}", flush=True)

    # --- the bark-run explanation of the early onsets (hit under overlap, missed under onset)
    rows = []
    for c, x, (shown, prov, orig) in zip(cl, cc, per):
        ev = [e for e in shown if is_salient_nonspeech(e.label) and not is_music(e.label)]
        gl = [g for g in c["events"] if g["consequential"] and is_salient_nonspeech(g["label"]) and not is_music(g["label"])]
        for g in gl:
            mo = [e for e in ev if E._same(e.label, g["label"]) and E._overlap_ok(e.start, e.end, g["start"], g["end"])]
            on = any(E._same(e.label, g["label"]) and g["start"] - 0.5 <= e.start <= g["start"] + 1.0 for e in ev)
            if not mo or on:
                continue
            e = min(mo, key=lambda e: e.start)
            prev = [h for h in gl if E._same(h["label"], g["label"]) and h["end"] <= g["start"] + 1e-9 and h is not g]
            prev_end = max((h["end"] for h in prev), default=None)
            covers_prev = any(min(e.end, h["end"]) - max(e.start, h["start"]) > 0 for h in prev)
            cur, ts = fam_curve(x[0], e.label)
            gap_min = None
            if prev_end is not None and g["start"] > prev_end:
                mm = (ts >= prev_end) & (ts <= g["start"])
                gap_min = float(cur[mm].min()) if mm.any() else None
            rows.append({"id": c["id"], "label": g["label"], "onset": g["start"], "span": [round(e.start, 2), round(e.end, 2)],
                         "err": round(e.start - g["start"], 2), "prov": prov.get(id(e)),
                         "beats_start_before_twin": round(orig.get(id(e), e.start), 2),
                         "prev_same_family_gap": None if prev_end is None else round(g["start"] - prev_end, 2),
                         "span_covers_an_earlier_same_family_event": covers_prev, "beats_min_in_gap": gap_min})
    errs = np.array([r["err"] for r in rows])
    gm = [r["beats_min_in_gap"] for r in rows if r["beats_min_in_gap"] is not None]
    gaps = [r["prev_same_family_gap"] for r in rows if r["prev_same_family_gap"] is not None]
    res["onset_only_misses"] = {
        "n": len(rows), "clips": len({r["id"] for r in rows}), "median_err": float(np.median(errs)) if len(rows) else None,
        "early": int((errs < -0.5).sum()), "late": int((errs > 1.0).sum()),
        "span_covers_earlier_same_family_event": sum(r["span_covers_an_earlier_same_family_event"] for r in rows),
        "prev_gap_median_s": float(np.median(gaps)) if gaps else None, "prev_gap_le_2s": sum(1 for x in gaps if x <= 2.0),
        "beats_stays_ge_0.175_through_gap": sum(1 for v in gm if v >= D.AED), "gap_checked": len(gm),
        "start_moved_by_twin_rule": sum(r["prov"] == "twin_moved" for r in rows),
        "early_without_any_earlier_same_family_event": sum(1 for r in rows if r["err"] < -0.5 and not r["span_covers_an_earlier_same_family_event"]),
        "per_clip": dict(Counter(r["id"] for r in rows).most_common()), "rows": rows}
    print(f"[caches] onset-only misses {res['onset_only_misses']['n']}: " + json.dumps({k: v for k, v in res['onset_only_misses'].items() if k not in ('rows', 'per_clip')}), flush=True)

    # --- baseline re-scores on the 280 (same spans, the scorer / gold setup changed one thing at a time)
    byid, par = ontology()
    m = strong_to_beats()
    same_m = same_multi(par)
    cl_rm = [remap_clip(c, m) for c in cl]
    shown = [p[0] for p in per]
    variants = {
        "baseline": lambda: [clip_cost_v(c, ev) for c, ev in zip(cl, shown)],
        "gold_names_by_MID": lambda: [clip_cost_v(c, ev) for c, ev in zip(cl_rm, shown)],
        "multi_parent_family": lambda: [clip_cost_v(c, ev, same=same_m) for c, ev in zip(cl, shown)],
        "names_by_MID+multi_parent": lambda: [clip_cost_v(c, ev, same=same_m) for c, ev in zip(cl_rm, shown)],
        "gold_runs_merged_2s": lambda: [clip_cost_v(c, ev, gold_events=merged_runs(c)) for c, ev in zip(cl, shown)],
    }
    out = {}
    for k, fn in variants.items():
        rr = fn(); out[k] = summ(rr, len(cl))
        out[k]["dC_vs_baseline"] = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rr, base_rows)]) if k != "baseline" else None
    old = config.LABEL_FILTER
    for k, cls in (("label_filter_depictable", cl), ("depictable+names_by_MID+multi_parent", cl_rm)):
        config.LABEL_FILTER = "depictable"
        try:
            rr = [clip_cost_v(c, ev, same=(same_m if cls is cl_rm else None)) for c, ev in zip(cls, shown)]
        finally:
            config.LABEL_FILTER = old
        out[k] = summ(rr, len(cl)); out[k]["dC_vs_baseline"] = D.boot([x["C_overlap"] - y["C_overlap"] for x, y in zip(rr, base_rows)])
    res["rescore_280"] = out
    for k, v in out.items():
        print(f"[rescore 280] {k:40s} C-ov {v['C_overlap']:.3f} C-on {v['C_onset']:.3f} rec {v['recall_overlap']:.1%} "
              f"fp {v['fp']} ({v['fp_per_min']:.2f}/min) n_conseq {v['n_conseq']} dC {v['dC_vs_baseline']}", flush=True)
    write("caches", res)


# ----------------------------------------------------------------------------- 4. real pipeline vs harness (cluster GPU, 280)
def pipeline(a):
    import importlib.util
    import tempfile
    from benchmark import audioset_stage4_report as R
    from benchmark import detector_round2 as D
    from benchmark import detector_round4 as R4
    from src.stage4_audio_event_detection import flexsed_infer as FX
    from src.stage4_audio_event_detection import beats_infer as B
    R.use_set("calib")
    cl = D.usable()
    rng = np.random.default_rng(SEED)
    if a.pick == "conseq":
        pool = [i for i, c in enumerate(cl) if any(g["consequential"] and is_salient_nonspeech(g["label"]) and not is_music(g["label"])
                                                   for g in c["events"])]
        pick = sorted(rng.choice(pool, min(a.n, len(pool)), replace=False).tolist())
    else:
        pick = sorted(rng.choice(len(cl), a.n, replace=False).tolist())
    config.use_shipped()
    spec = importlib.util.spec_from_file_location("stage4_local", a.stage4_file)
    S4 = importlib.util.module_from_spec(spec); spec.loader.exec_module(S4)
    assert "BEATS_SELF_VETO" in Path(a.stage4_file).read_text(encoding="utf-8"), "stage-4 file has no self-veto block"
    FX.CACHE_DIR = R.FLEX                                   # data/work/flexsed_calib/<id>.npz
    rows, cost_h, cost_p = [], [], []
    with tempfile.TemporaryDirectory() as td:
        for i in pick:
            c = cl[i]
            wd = Path(td) / c["id"]; wd.mkdir()
            wav = wd / "audio.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(R.E.VIDEOS / f"{c['id']}.mp4"), "-vn", "-ac", "1",
                            "-ar", "16000", str(wav)], check=True)
            # live BEATs vs the cache
            fw, ts, labs = B.infer_beats(wav, "cuda")
            cb = R.load(R.E.WIN / "beats" / f"{c['id']}.npz")
            beats_diff = float(np.abs(fw.astype(np.float32) - cb[0]).max()) if fw.shape == cb[0].shape else None
            times_same = bool(len(ts) == len(cb[1]) and np.allclose(ts, cb[1], atol=1e-4))
            ev = S4.detect_events(wav, threshold=config.AED_THRESHOLD, min_dur=config.AED_MIN_DUR, model=config.AED_MODEL, device="cuda")
            trace = list(S4.TRACE)
            pre = [t for t in trace if t["step"] == "veto"]
            pipe_shown = [e for e in ev if e.confidence >= config.DISPLAY_THRESHOLD]
            pre_shown = [t for t in pre if t["conf"] >= config.DISPLAY_THRESHOLD]
            cc = R4.load3(c["id"])
            harn = stack_prov(cc)[0]
            hs = sorted((e.label, round(e.start, 2), round(e.end, 2)) for e in harn)
            ps_pre = sorted((t["label"], round(t["start"], 2), round(t["end"], 2)) for t in pre_shown)
            ps = sorted((e.label, round(e.start, 2), round(e.end, 2)) for e in pipe_shown)
            moved = []
            for (l1, s1, e1) in ps_pre:
                for e in pipe_shown:
                    if e.label == l1 and abs(e.end - e1) < 0.01 and abs(e.start - s1) > 0.01:
                        moved.append(round(e.start - s1, 2))
            cost_h.append(clip_cost_v(c, harn)); cost_p.append(clip_cost_v(c, pipe_shown))
            rows.append({"id": c["id"], "beats_live_vs_cache_maxabs": beats_diff, "beats_times_same": times_same,
                         "harness": len(hs), "pipeline_before_refine": len(ps_pre), "pipeline_final": len(ps),
                         "same_before_refine": hs == ps_pre,
                         "only_harness": [x for x in hs if x not in ps_pre], "only_pipeline_before_refine": [x for x in ps_pre if x not in hs],
                         "refine_start_moves": moved, "any_refined_start_earlier": any(mv < -0.01 for mv in moved),
                         "labels_differ_final": sorted({x[0] for x in hs} ^ {x[0] for x in ps})})
            print(f"[pipeline] {c['id']}: harness {len(hs)} / pipe-pre {len(ps_pre)} / pipe {len(ps)} same-pre {hs == ps_pre} "
                  f"beats diff {beats_diff} moves {moved}", flush=True)
    mv = [x for r in rows for x in r["refine_start_moves"]]
    res = {"clips": len(rows), "pick": a.pick, "stage4_file": str(a.stage4_file),
           "same_spans_before_refine": sum(r["same_before_refine"] for r in rows),
           "beats_live_vs_cache_maxabs_max": max((r["beats_live_vs_cache_maxabs"] or 0) for r in rows),
           "beats_times_same_all": all(r["beats_times_same"] for r in rows),
           "refine_moved_starts": len(mv), "refine_move_median_s": float(np.median(mv)) if mv else None,
           "refine_moves_earlier": sum(1 for x in mv if x < -0.01),
           "cost_on_these_clips": {"harness": summ(cost_h, len(rows)), "pipeline": summ(cost_p, len(rows))}, "rows": rows,
           "config": {k: getattr(config, k, None) for k in ("FLEXSED_BAR", "FLEXSED_VETO", "PANNS_VETO", "BEATS_SELF_VETO", "ONSET_CAM",
                                                           "ONSET_MONOTONE", "LABEL_FILTER", "MAX_SPAN", "UNION_START", "UNION_WEAK_TWIN",
                                                           "AED_THRESHOLD", "AED_HYSTERESIS", "DISPLAY_THRESHOLD", "AED_MIN_DUR")}}
    print(f"[pipeline] {json.dumps({k: v for k, v in res.items() if k != 'rows'})}", flush=True)
    write("pipeline" if a.pick == "random" else "pipeline_conseq", res)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("probe", "align", "gold", "scorer", "caches", "pipeline"))
    ap.add_argument("--stage4-file", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--pick", choices=("random", "conseq"), default="random", help="conseq: only clips with a scored consequential event")
    a = ap.parse_args()
    {"probe": probe, "align": align, "gold": gold_check, "scorer": scorer, "caches": caches, "pipeline": pipeline}[a.step](a)


if __name__ == "__main__":
    main()
