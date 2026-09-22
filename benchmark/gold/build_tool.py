"""Build the gold-set annotation tool: one self-contained HTML page (benchmark/gold/index.html)
that plays each test clip from the local benchmark folder and records, per sound, whether it
is heard, whether its source is visible while it sounds, whether the picture alone already
makes it obvious, how much a deaf viewer would miss it (importance 1-3), whether speech or
music covers it, and its rough start/end; per clip, one human sentence "what a hearing viewer gets that a deaf
viewer misses". The detector's candidates are hidden until the annotator asks for them, so a
first free listening pass is recorded before any anchoring.

Output of an annotator: one JSON file (Export button) -> benchmark/gold/annotations/<name>.json.
Merge and agreement: benchmark/gold/merge.py.

    python benchmark/gold/build_tool.py                    # writes benchmark/gold/index.html
    python benchmark/gold/build_tool.py --slice audioset   # writes benchmark/gold/index_audioset.html
                                                           # (pre-fill from AudioSet-Strong human labels,
                                                           #  read from benchmark/gold/audioset_slice.json)
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gate_dev_sweep import load, _find_clip, min_confidence, decide
from src.labels import is_salient_nonspeech, is_music, SPEECH_LABELS

HERE = Path(__file__).resolve().parent
DUE = {"unseen_ambient", "mixed"}
NOT_SOUNDS = {"Speech", "Music"}

# default "importance" (how much a deaf viewer would miss the sound): 3 = safety or plot, 1 = background texture, else 2 = context
IMPORTANCE_3 = ["siren", "alarm", "glass", "shatter", "gunshot", "explosion", "baby", "doorbell", "telephone", "phone", "horn",
                "fireworks", "thunder", "scream", "crying", "smoke detector", "buzzer", "bell"]
IMPORTANCE_1 = ["traffic", "engine", "vehicle", "rain", "wind", "water", "stream", "waves", "crowd", "hubbub", "noise", "hum", "air conditioning"]


def _has(text, keywords):
    """whole-word match (plus -s/-es/-ing/-ed), so 'train' is not 'rain' and 'human' is not 'hum'"""
    return any(re.search(r"\b" + re.escape(k) + r"(s|es|ing|ed)?\b", text) for k in keywords)


def importance_of(label, family=""):
    """1, 2 or 3 from the lowercased label and family: a safety/plot keyword wins, then a background keyword, else 2"""
    text = f"{label or ''} {family or ''}".lower()
    return 3 if _has(text, IMPORTANCE_3) else 1 if _has(text, IMPORTANCE_1) else 2


def _detail(rec, label):
    """the most specific raw detection under this family, for a friendlier default name"""
    best = None
    for e in rec.get("events", []):
        from src.labels import canonical
        if canonical(e["label"]) == label and (best is None or e["confidence"] > best["confidence"]):
            best = e
    return best["label"] if best else label


def _masked(rec, start, end, conf):
    """the sound is quiet (below 0.5) while speech or music above 0.5 covers most of its span"""
    if conf >= 0.5:
        return False
    for e in rec.get("events", []):
        if e["label"] in ("Speech", "Music") and e["confidence"] >= 0.5 and min(e["end"], end) - max(e["start"], start) >= 0.5 * (end - start):
            return True
    return False


# 2026-09-20, Adam: "take the mixed I did not tag and move them; I want 30 more mixed candidates".
# The pre-tagged mixed clips he had not finished (per his last export) are parked (hidden unless
# done, or "show parked" is ticked); these 32 clips, model-suggested "mixed" at sorting time and
# never tagged, come in as split "extra" with tag "mixed" (a candidate tag; his ticks decide).
EXTRA_MIXED = ["mc_courtroom_verdict", "mc_airport_scene", "mc_beach_scene", "mc_car_chase_scene", "mc_siren_arrival",
               "mc_parade_scene", "mv_earthquake_scene", "gn_docudrama_reenactment", "oc_creaking_floorboards",
               "oc_pov_hunting_stalk", "mx_harbor_boat_engine", "mx_air_show_crowd", "mx_level_crossing_town",
               "mx_market_stall_frying", "mx_swan_river_boats", "mx_duck_pond_feeding", "mx_pedestrian_crossing_beep",
               "as_jackhammer_Gm7nLucM", "as_thunder_4gKvZMFU", "as_helicopter_i57RXxfJ", "as_siren_i4JvkuR0",
               "lx_chainsaw_and_power_tool_BBukw6J", "lx_crowd_and_applause_6ihRuUM", "ly_dog_4nNLrN8", "ly_dog_RUISGGA",
               "ly_truck_X-t-4sb", "ly_civil_defense_siren_3U2JPY2", "un_dog_barking_3doKyrCe", "un_dog_howling_8VZs-wTE",
               "un_skidding_LX5Y1jco", "un_wind_noise_CiSmL4nm", "un_vacuum_cleaner_cle_7bHrke6z", "un_people_shouting_0SL91QFy",
               "un_thunder_QISybl0-"]
_LAST_EXPORT = HERE / "annotations" / "gold_AG.json"
# 2026-09-21 night, Adam: "I need at least 30 more candidates to mixed ... clearer mixed style".
# Second wave (scripts/source_mixed2.py, m2_): each query = a visible actor doing something
# audible + a named off-screen event; kept by an audio-only rule (BEATs, no visibility model).
# Tier 1 = two drawable families of importance >= 2 heard, tier 2 = one. Same split "extra".
_MIXED2 = HERE / "mixed2_audio.json"
MIXED2 = json.loads(_MIXED2.read_text(encoding="utf-8")) if _MIXED2.exists() else {}
# VLM-only pre-screen of the same clips (scripts/screen_mixed2.py: frames + the list of sounds heard,
# one question, no pipeline gate): per clip
# a verdict mixed / unseen / seen / empty. Adam (2026-09-21 night): "use the VLM to check which
# ones actually are candidates for mixed" -> only verdict "mixed" clips are shown; the rest are
# parked (visible with "also the parked ones"). A pre-screen: which videos might contain two or more sounds, some seen and some unseen. Every tag is Adam's.
_VLM2 = HERE / "mixed2_vlm.json"
VLM2 = json.loads(_VLM2.read_text(encoding="utf-8")) if _VLM2.exists() else {}
# same pre-screen on the wave-5 film cuts (unseen candidates): verdict unseen / mixed shown first, seen / empty parked
_VLM5 = HERE / "movies2_vlm.json"
VLM5 = json.loads(_VLM5.read_text(encoding="utf-8")) if _VLM5.exists() else {}
_AUD5 = HERE / "movies2_audio.json"
AUD5 = json.loads(_AUD5.read_text(encoding="utf-8")) if _AUD5.exists() else {}


def _done_ids():
    """clip ids finished or marked bad in the annotator's last export (parked clips stay visible if done)"""
    if not _LAST_EXPORT.exists():
        return set()
    return {c["clip"] for c in json.loads(_LAST_EXPORT.read_text(encoding="utf-8")).get("clips", []) if c.get("done") or c.get("bad")}


def _duration(path):
    import subprocess
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True).stdout.strip()
    return round(float(out), 1) if out else 0.0


def build_extra(done=frozenset()):
    """the extra mixed candidates: wave 1 (EXTRA_MIXED, parked unless done) and wave 2 (m2_, from mixed2_audio.json)"""
    clips = []
    _rank = {"mixed": 0, "unseen": 1, "seen": 2, "empty": 3}
    m2 = sorted(MIXED2, key=lambda s: (_rank.get(VLM2.get(s, {}).get("verdict"), 1), MIXED2[s]["tier"], s))
    # wave 3 (scripts/source_mixed3.py): chosen by genre only, no screening at all -> shown as they are
    m3 = sorted(q.stem for q in (_ROOT / "data" / "input" / "benchmark" / "unsorted").glob("m3_*.mp4"))
    # wave 4 (scripts/source_mixed4.py): by structure (kennels, barns, colonies, ranges; dog + fireworks,
    # helmet cams, milsim; named film scenes), no screening. Waves 2 and 3 were dropped (Adam, 2026-09-22:
    # none mixed) -> their files sit in _dropped and are not listed.
    m4 = sorted(q.stem for q in (_ROOT / "data" / "input" / "benchmark" / "unsorted").glob("m4_*.mp4"))
    # wave 5 (scripts/source_movies2.py): film scenes chosen for off-screen sound -> UNSEEN candidates,
    # tag "unseen_ambient"; wave 4 parked (Adam, 2026-09-22: "all non-tagged m4 you can drop")
    _rank5 = {"unseen": 0, "mixed": 0, "seen": 2, "empty": 3}
    m5 = sorted((q.stem for q in (_ROOT / "data" / "input" / "benchmark" / "unsorted").glob("m5_*.mp4")),
                key=lambda s: (_rank5.get(VLM5.get(s, {}).get("verdict"), 1), s))
    for stem in EXTRA_MIXED + m2 + m3 + m4 + m5:
        video = next((p for d in ("_dropped", "unsorted", "mixed", "seen_ambient", "unseen_ambient", "no_ambient")
                      for p in [_ROOT / "data" / "input" / "benchmark" / d / f"{stem}.mp4"] if p.exists()), None)
        if video is None:
            print(f"extra clip missing: {stem}"); continue
        rel = video.resolve().relative_to(_ROOT).as_posix()
        if stem in VLM2 and VLM2[stem].get("verdict") not in (None, "error"):
            v = VLM2[stem]
            cands = [{"label": f["label"].replace(" (siren)", " siren").lower(), "family": f["family"], "conf": f["conf"],
                      "start": f["start"], "end": f["end"], "visible": seen, "obvious": seen,
                      "importance": importance_of(f["label"].lower(), f["family"]), "masked": False,
                      "gate": "vlm " + ("seen" if seen else "unseen")}
                     for seen, rows in ((False, v["unseen"]), (True, v["seen"]), (False, v.get("unsure", []))) for f in rows]
        elif stem in VLM5 and VLM5[stem].get("verdict") not in (None, "error"):
            v = VLM5[stem]
            cands = [{"label": f["label"].replace(" (siren)", " siren").lower(), "family": f["family"], "conf": f["conf"],
                      "start": f["start"], "end": f["end"], "visible": seen, "obvious": seen,
                      "importance": importance_of(f["label"].lower(), f["family"]), "masked": False,
                      "gate": "vlm " + ("seen" if seen else "unseen")}
                     for seen, rows in ((False, v["unseen"]), (True, v["seen"]), (False, v.get("unsure", []))) for f in rows]
        elif stem in AUD5:
            cands = [{"label": f["detail"].replace(" (siren)", " siren").lower(), "family": f["family"], "conf": f["conf"],
                      "start": f["start"], "end": f["end"], "visible": False, "obvious": False,
                      "importance": importance_of(f["detail"].lower(), f["family"]), "masked": False, "gate": "beats audio-only"}
                     for f in AUD5[stem]]
        elif stem.startswith(("m3_", "m4_", "m5_")):
            cands = []
        elif stem in MIXED2:
            cands = [{"label": f["detail"].replace(" (siren)", " siren").lower(), "family": f["family"], "conf": f["conf"],
                      "start": f["start"], "end": f["end"], "visible": False, "obvious": False,
                      "importance": importance_of(f["detail"].lower(), f["family"]), "masked": False, "gate": "beats audio-only"}
                     for f in MIXED2[stem]["families"]]
        else:
            cands = [{"label": k["detail"].replace(" (siren)", " siren").lower(), "family": k["label"], "conf": k["conf"],
                      "start": k["start"], "end": k["end"], "visible": False, "obvious": False,
                      "importance": importance_of(k["detail"].lower(), k["label"]), "masked": False,
                      "gate": ("psed" if k["det"] == "psed" else "beats") + " " + k["gate"]} for k in DET_CANDS.get(stem, [])]
        cands.sort(key=lambda c: -c["conf"])
        wave = 5 if stem.startswith("m5_") else 4 if stem.startswith("m4_") else 3 if stem.startswith("m3_") else 2 if stem in MIXED2 else 1
        clip = {"id": f"{stem}.mp4", "src": "../../" + rel, "duration": _duration(video), "candidates": cands[:12],
                "tag": "unseen_ambient" if wave == 5 else "mixed", "split": "extra", "wave": wave,
                "picture_due": True, "sentence": "nothing beyond the picture"}
        # 2026-09-21 night, Adam: wave 1 parked ("park") -- hidden unless done, or "also the parked ones"
        if clip["wave"] in (1, 4) and stem + ".mp4" not in done:
            clip["parked"] = True          # the page still shows a parked clip you marked done
        if clip["wave"] == 5:
            clip["vlm"] = VLM5.get(stem, {}).get("verdict", "")
            if VLM5 and clip["vlm"] in ("seen", "empty") and stem + ".mp4" not in done:
                clip["parked"] = True      # VLM says nothing off screen: parked, still reachable
        if clip["wave"] == 2:
            clip["vlm"] = VLM2.get(stem, {}).get("verdict", "")
            if VLM2 and clip["vlm"] != "mixed" and stem + ".mp4" not in done:
                clip["parked"] = True       # VLM says not mixed: parked, still reachable
        clips.append(clip)
    return clips


_DC = HERE / "detector_candidates.json"
DET_CANDS = json.loads(_DC.read_text(encoding="utf-8")) if _DC.exists() else {}


def build_default():
    """the 100 benchmark test clips, pre-filled from the detector, the gate and the proposed system's sentence"""
    gate = json.loads((_ROOT / "benchmark" / "gate_setting.json").read_text(encoding="utf-8"))["chosen"]
    results = {}
    for rec in json.loads((_ROOT / "benchmark" / "protocol_results_v3_grounded.json").read_text(encoding="utf-8")):
        if rec["system"] == "proposed":
            results[rec["clip"]] = rec
    clips = []
    pool = [(r, "test") for r in load("test")] + [(r, "dev") for r in load("dev")]   # test = the 100 protocol clips; dev = the rest, to browse
    for rec, split in pool:
        video = _find_clip(rec["clip"])
        if video is None:
            continue
        rel = Path(video).resolve().relative_to(_ROOT).as_posix()
        d = decide(rec, gate["bar"], gate["rule"], gate["kinds"])
        cands = []
        for s in rec["sounds"]:
            if s["confidence"] < min_confidence(s["label"], gate["bar"]):
                continue
            visible_all = bool(s["stretches"]) and all(st.get("verdict") for st in s["stretches"])
            label = _detail(rec, s["label"]).replace(" (siren)", " siren").lower()
            cands.append({"label": label, "family": s["label"],
                          "conf": round(s["confidence"], 2), "start": round(s["start"], 1), "end": round(s["end"], 1),
                          "visible": visible_all, "obvious": visible_all, "importance": importance_of(label, s["label"]),
                          "masked": _masked(rec, s["start"], s["end"], s["confidence"]),
                          "gate": "silenced" if s["label"] in d["silenced"] else ("shown" if s["label"] in d["shown"] else "dropped")})
        # add what the two detectors of the v4 rows heard (BEATs from v4b, PSED from v4ab; all detections
        # above their bars, whether the gate showed them or not), so no detector's candidates are missing
        for k in DET_CANDS.get(Path(rec["clip"]).stem, []):
            if any(c["family"] == k["label"] and min(c["end"], k["end"]) - max(c["start"], k["start"]) > 0 for c in cands):
                continue
            cands.append({"label": k["detail"].replace(" (siren)", " siren").lower(), "family": k["label"], "conf": k["conf"],
                          "start": k["start"], "end": k["end"], "visible": False, "obvious": False,
                          "importance": importance_of(k["detail"].lower(), k["label"]), "masked": False,
                          "gate": ("psed" if k["det"] == "psed" else "beats") + " " + k["gate"]})
        cands.sort(key=lambda c: -c["conf"])
        pr = results.get(rec["clip"], {})
        ref = pr.get("reference", "")
        clips.append({"id": rec["clip"], "src": "../../" + rel, "duration": round(rec["duration"], 1), "candidates": cands[:12],
                      "tag": rec["tag"], "split": split, "picture_due": rec["tag"] in DUE,
                      "sentence": ref if ref and ref != "nothing beyond the picture" else "nothing beyond the picture"})
    # variety: the 111 slice-B clips (10-s YouTube: babies, dogs, doorbells, gunshots, alarms, movies...)
    # are in the same page, tag "audioset_strong"; both detectors' rows were rendered on them, so
    # they score exactly like the protocol clips
    for c in build_audioset():
        c["split"] = "test"
        clips.append(c)
    done = _done_ids()
    for c in clips:
        if c["tag"] == "mixed" and c["id"] not in done:
            c["parked"] = True
    clips += build_extra(done)
    return clips


def build_audioset():
    """slice B: AudioSet-Strong clips, pre-filled from the human-timed labels (no detector, no sentence, picture-due open)"""
    src = HERE / "audioset_slice.json"
    if not src.exists():
        sys.exit(f"{src} not found — run benchmark/gold/audioset_slice.py first")
    clips, missing = [], 0
    for c in json.loads(src.read_text(encoding="utf-8"))["clips"]:
        if not (_ROOT / "data" / "input" / "audioset_strong" / f"{c['id']}.mp4").exists():
            missing += 1
            continue
        cands = [{"label": e["label"].lower(), "family": e["label"], "conf": 1.0, "start": round(e["start"], 1), "end": round(e["end"], 1),
                  "visible": False, "obvious": False, "importance": importance_of(e["label"]),
                  "masked": bool(e.get("masked")), "gate": "human"}
                 for e in c.get("events", []) if e["label"] not in NOT_SOUNDS
                 and e["label"] not in SPEECH_LABELS and not is_music(e["label"])]
        clips.append({"id": c["id"], "src": c["src"], "duration": round(c["duration"], 1), "candidates": cands,
                      "tag": "audioset_strong", "picture_due": None, "sentence": ""})
    if missing:
        print(f"skipped {missing} clips whose mp4 is not under data/input/audioset_strong/")
    return clips


def write(clips, name):
    clips.sort(key=lambda c: c["id"])
    html = (HERE / "tool_template.html").read_text(encoding="utf-8")
    html = html.replace("/*__CLIPS__*/[]", json.dumps(clips, ensure_ascii=False))
    # distinct title per page: the browser store key is derived from it
    html = html.replace("<title>Gold set annotation</title>",
                        "<title>Gold set annotation" + (" (slice B)" if "audioset" in name else " (benchmark)") + "</title>")
    (HERE / name).write_text(html, encoding="utf-8")
    print(f"-> {HERE / name}: {len(clips)} clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slice", choices=["audioset"], default=None, help="build the AudioSet-Strong page instead of the benchmark one")
    args = ap.parse_args()
    if args.slice == "audioset":
        write(build_audioset(), "index_audioset.html")
    else:
        write(build_default(), "index.html")
