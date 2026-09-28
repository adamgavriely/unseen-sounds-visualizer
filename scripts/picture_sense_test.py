"""Picture sense test (plan: docs/picture_sense_test_2026-09-28.md, written before any picture was drawn).

Arms: A = hand table (shipped), B = VLM free text (PICTURE_LOOK_VLM + PICTURE_LOOKALIKE_VLM), C = PICTURE_SENSE
(src/stage6_visual_augmentation/sense.py). 7 known bad cases + 25 unseen labels. Every arm is checked with the same
fixed union question per sound (sense.check_fixed).

    python scripts/picture_sense_test.py --phase prep     # subjects (frozen), swaps, slots, B look-alikes (text only)
    python scripts/picture_sense_test.py --phase mine     # mistake mining (4 pictures per label), union options
    python scripts/picture_sense_test.py --phase arm --arm A|B|C
    python scripts/picture_sense_test.py --phase sheet    # (local, CPU) blind folder + contact sheet
Output: data/work/picture_sense_test/{items.json, union.json, <arm>/<tag>/aug_*.png, results_<arm>.json}
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))
import config

OUT = _ROOT / "data" / "work" / "picture_sense_test"
WORK = _ROOT / "data" / "work"
SEED = 20260928
# (case, split tag, clip, index) -- one real spec each (docs/picture_sense_test_2026-09-28.md)
KNOWN = [("fire alarm", "dev_monocap_v31", "as_fire_alarm_kGKZ0YK4", 0),
         ("crowd", "dev_monocap_v31", "mv_protest_scene_movie", 0),
         ("steam", "dev_monocap_v31", "b3_crossing_bells", 0),
         ("smoke detector", "test_final_v33", "w8_helmetcam_chainsaw_roof_2b", 1),
         ("car horn", "sliceB_v32", "97aoiaWwRVk_20000", 0),
         ("rattle", "sliceB_v32", "_U8kAFAm8tQ_30000", 0),
         ("typing", "test_final_v33", "ambient_nightlife_neon_97", 0)]
KNOWN_SOURCES = {"Alarm", "Crowd", "Steam", "Smoke detector, smoke alarm", "Vehicle horn, car horn, honking",
                 "Rattle (instrument)", "Typing", "Vehicle", "Computer keyboard"}
N_UNSEEN = 25


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def code_hashes() -> dict:
    fs = ["src/stage6_visual_augmentation/verify.py", "src/stage6_visual_augmentation/__init__.py",
          "src/stage6_visual_augmentation/sense.py", "src/stage5_cross_modal_analysis/reason.py", "src/labels.py",
          "config.py", "benchmark/gold/gen_screen.py", "scripts/picture_sense_test.py"]
    return {f: md5(_ROOT / f) for f in fs}


def spec_of(it: dict) -> SimpleNamespace:
    return SimpleNamespace(index=it["index"], event_label=it["event_label"], source=it["source"],
                           subject=it["subject"], start=float(it["start"]), augment=True, image_prompt="",
                           image_path="", backend="")


def drawn_subject(spec) -> str:
    from benchmark.gold.gen_screen import TEMPLATES
    src = spec.source or spec.event_label
    key = src if src in TEMPLATES else (spec.event_label if spec.event_label in TEMPLATES else None)
    return TEMPLATES[key] if key else spec.subject


def unseen_order() -> list:
    """The 215 families shuffled with the fixed seed, minus cards, templates, the known sources and table words."""
    from benchmark.gold.gen_screen import TEMPLATES, CARDS
    from src.stage6_visual_augmentation import verify as V
    fam = sorted(json.loads((_ROOT / "benchmark" / "gold" / "depictable_vocab.json").read_text("utf-8"))["families"])
    order = list(fam)
    random.Random(SEED).shuffle(order)
    out = []
    for L in order:
        sp = SimpleNamespace(event_label=L, source=L, start=0.0)
        if L in CARDS or L in TEMPLATES or L in KNOWN_SOURCES or V.ambiguous_entry(sp, ""):
            continue
        out.append(L)
    return out


# ---------------------------------------------------------------------------------------------------------------
def phase_prep():
    from src.stage5_cross_modal_analysis import reason as R
    from src.stage6_visual_augmentation import verify as V
    from src.stage6_visual_augmentation import sense as S
    mdl, proc = V._vlm()
    items = []
    for case, tag, clip, idx in KNOWN:
        d = WORK / f"shipped_{tag}" / clip
        a = next(x for x in json.loads((d / "augmentations.json").read_text("utf-8")) if x["index"] == idx)
        spec = SimpleNamespace(index=a["index"], event_label=a["event_label"], source=a.get("source", ""),
                               subject=a.get("subject", ""), start=float(a["start"]))
        stored = spec.subject
        spec.subject = R.with_maker(spec, stored, None, mdl, proc)
        items.append({"key": f"known/{case}", "group": "known", "case": case, "tag": clip, "split": tag,
                      "index": spec.index, "event_label": spec.event_label, "source": spec.source,
                      "start": spec.start, "stored_subject": stored, "subject": spec.subject})
    order = unseen_order()
    swaps, n = [], 0
    for L in order:
        if n >= N_UNSEEN:
            break
        spec = SimpleNamespace(index=0, event_label=L, source=L, subject="", start=0.0)
        phrase = R._depict_v31(spec, "", None, [], mdl, proc)
        subj = R.with_maker(spec, phrase, None, mdl, proc)
        spec.subject = subj
        if V.ambiguous_entry(spec, drawn_subject(spec)):
            swaps.append({"label": L, "subject": subj, "why": "subject matches an AMBIGUOUS entry"})
            print(f"[prep] swap {L!r}: subject {subj!r} matches the table", flush=True)
            continue
        n += 1
        items.append({"key": f"unseen/{L}", "group": "unseen", "case": L, "tag": "sense_" + slug(L), "split": "",
                      "index": 0, "event_label": L, "source": L, "start": 0.0, "stored_subject": phrase,
                      "subject": subj})
    for it in items:
        spec = spec_of(it)
        it["drawn"] = drawn_subject(spec)
        o = V.options_for(spec, it["drawn"])
        it["a_intended"], it["a_confusions"], it["entry"] = o["intended"], o["confusions"], o["entry"]
        e = V.ambiguous_entry(spec, it["drawn"])
        it["b_lookalikes"] = V.lookalikes(e["maker"], e["intended"]) if e and e.get("maker") else []
        sf = S.slot_form(spec)
        it["slots"] = {k: sf[k] for k in ("slots", "why", "sentence", "object", "raw")}
        print(f"[prep] {it['key']}: subject {it['subject']!r} drawn {it['drawn']!r} | C {sf['sentence'] or '(plain)'}",
              flush=True)
    (OUT / "items.json").write_text(json.dumps({"items": items, "swaps": swaps, "code": code_hashes(),
                                                "vlm": config.VLM_MODEL, "gen": config.GEN_MODEL},
                                               indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[prep] {len(items)} items, {len(swaps)} swaps", flush=True)


def _content(s: str) -> set:
    from src.stage6_visual_augmentation import sense as S
    return {w for w in S._wset(s) if w not in S.FILLER and w not in {"something", "else", "thing"}}


def phase_mine():
    from src.stage6_visual_augmentation import sense as S
    from src.labels import label_names
    d = json.loads((OUT / "items.json").read_text("utf-8"))
    fixed = {}
    for it in d["items"]:
        spec = spec_of(it)
        t0 = time.time()
        p = S.plan(spec, it["drawn"], config.GEN_MODEL, "cuda", config.RESOLUTION)
        own = S.own_options(spec, it["drawn"])
        it["c_plan"] = p
        it["c_own"] = own
        # the union question: A's intended; A's + B's + C's look-alikes (+ C's generics), de-duplicated; an option whose
        # content words are all words of the intended / label / plain subject is dropped (a second right answer)
        right = _content(it["a_intended"]) | _content(it["subject"]) | _content(it["drawn"])
        for n in label_names(it["source"]) + label_names(it["event_label"]):
            right |= _content(n)
        conf, seen = [], []
        for c in it["a_confusions"] + it["b_lookalikes"] + p["lookalikes"] + own["confusions"]:
            cw = _content(c)
            if not cw or cw <= right or cw in seen or c == it["a_intended"]:
                continue
            seen.append(cw)
            conf.append(c)
        it["union"] = {"intended": it["a_intended"], "confusions": conf}
        fixed[f"{it['tag']}|{it['source']}"] = it["union"]
        print(f"[mine] {it['key']} ({time.time() - t0:.0f}s): looks {p['lookalikes']} neg {p['neg']!r} | union "
              f"{len(conf)} confusions", flush=True)
    d["code_mine"] = code_hashes()
    (OUT / "items.json").write_text(json.dumps(d, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "union.json").write_text(json.dumps(fixed, indent=1, ensure_ascii=False), encoding="utf-8")


def _small(work_dir: Path, index: int):
    from PIL import Image
    for p in work_dir.glob(f"aug_{index:03d}*.png"):
        im = Image.open(p)
        if im.size[0] > 512:
            im.convert("RGB").resize((512, 512), Image.LANCZOS).save(p)


def phase_arm(arm: str):
    from src.stage6_visual_augmentation import _final_picture, VERIFY_LOG
    from src.stage6_visual_augmentation import verify as V
    from src.stage6_visual_augmentation import sense as S
    config.PICTURE_LOOK_VLM = arm == "B"
    config.PICTURE_LOOKALIKE_VLM = arm == "B"
    config.PICTURE_SENSE = arm == "C"
    print("[cfg] arm", arm, "LOOK_VLM", config.PICTURE_LOOK_VLM, "LOOKALIKE_VLM", config.PICTURE_LOOKALIKE_VLM,
          "SENSE", config.PICTURE_SENSE, "VERIFY", config.PICTURE_VERIFY, "MAKER", config.PICTURE_MAKER,
          "tries", config.PICTURE_VERIFY_TRIES, flush=True)
    d = json.loads((OUT / "items.json").read_text("utf-8"))
    S.FIXED.update(json.loads((OUT / "union.json").read_text("utf-8")))
    own_check = S.check
    V.check = S.check_fixed                  # every arm: the same fixed union question
    S.check = S.check_fixed
    res_p = OUT / f"results_{arm}.json"
    rows = json.loads(res_p.read_text("utf-8"))["rows"] if res_p.exists() else []
    done = {r["key"] for r in rows}
    for it in d["items"]:
        if arm == "B" and it["group"] != "known":
            continue                         # B = A on the unseen by construction (no table entry)
        if it["key"] in done:
            continue
        spec = spec_of(it)
        wd = OUT / arm / it["tag"]
        wd.mkdir(parents=True, exist_ok=True)
        path = wd / f"aug_{spec.index:03d}.png"
        t0 = time.time()
        n0 = len(VERIFY_LOG)
        _final_picture(spec, path, wd, it["subject"], config.RESOLUTION, config.GEN_MODEL, "cuda")
        log = VERIFY_LOG[-1] if len(VERIFY_LOG) > n0 else {}
        tries = log.get("tries", [])
        card = log.get("final") == "word card"
        row = {"arm": arm, "key": it["key"], "group": it["group"], "case": it["case"], "tag": it["tag"],
               "label": it["source"], "n_tries": len(tries), "pass1": bool(tries and tries[0]["ok"]),
               "passed": any(t["ok"] for t in tries), "card": card, "final_prompt": spec.image_prompt,
               "image": str(path.relative_to(OUT)), "log": log, "secs": round(time.time() - t0)}
        if arm == "C" and not card:
            o = own_check(path, spec, it["drawn"], salt=0, tag=it["tag"], see=False)
            row["own_check"] = {"ok": o["ok"], "picked": o["mc"]["picked"], "options": o["mc"]["options"]}
        _small(wd, spec.index)
        rows.append(row)
        print(f"[{arm}] {it['key']}: tries {len(tries)} pass1 {row['pass1']} passed {row['passed']} card {card} "
              f"({row['secs']}s) {spec.image_prompt[:90]!r}", flush=True)
        res_p.write_text(json.dumps({"arm": arm, "code": code_hashes(), "rows": rows}, indent=1, ensure_ascii=False),
                         encoding="utf-8")
    print(f"[{arm}] done: {len(rows)} rows", flush=True)


# ---------------------------------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["prep", "mine", "arm"], required=True)
    ap.add_argument("--arm", choices=["A", "B", "C"])
    a = ap.parse_args()
    print("[cfg]", config.use_shipped(), flush=True)
    config.DEVICE = "cuda"
    print("[cfg] VLM", config.VLM_MODEL, "GEN", config.GEN_MODEL, "RES", config.RESOLUTION, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.phase == "prep":
        phase_prep()
    elif a.phase == "mine":
        phase_mine()
    else:
        phase_arm(a.arm)


if __name__ == "__main__":
    main()
