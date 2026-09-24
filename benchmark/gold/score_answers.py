"""Score Adam's free-text answers from the blind picture page (docs/picture_v3_prereg.md, amendment).

Two steps, in this order, and the order is the point:

  1. `score`  — every answer is classified against the committed answer sheet using ONLY the sound
                key (which sound the picture was for), never the arm key. Anything the sheet cannot
                classify is listed; the author decides those by hand in an overrides file, still
                without the arm key, and runs `score` again until nothing is left.
  2. `unseal` — only then is the arm key opened: correct rate per arm, paired bootstrap over sounds
                (2000 draws, seed 0), Adam's own agreement on the repeats, the split by whether the
                source differed from the family, and the checker's calibration (frozen rule).

    python benchmark/gold/score_answers.py score  --answers picture_answers_2.json \
        --key data/work/rate_pictures2_SOUND_KEY_open_for_scoring.json \
        --sheet benchmark/gold/pictures/answer_sheet_picfresh_v32.json \
        [--overrides benchmark/gold/pictures/answer_overrides_picfresh_v32.json]
    python benchmark/gold/score_answers.py unseal --scored <scored json> \
        --arms data/work/rate_pictures2_ARM_KEY_do_not_open_until_scored.json \
        [--check <check_picfresh_v32_scored.json, written by `check` before Adam's file is opened>]
    python benchmark/gold/score_answers.py check --check <check_picfresh_v32.json> --sheet <sheet>
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

CANT = "__cant__"
ORDER = ["correct", "narrower", "wrong", "vague"]      # first list that matches decides


def _norm(text: str) -> str:
    t = re.sub(r"[^a-z0-9\- ]+", " ", text.lower().replace("-", " "))
    return " " + re.sub(r"\s+", " ", t).strip() + " "


def _has(text: str, phrase: str) -> bool:
    """whole-word match, tolerant of a plural s/es on the last word"""
    p = _norm(phrase).strip()
    if not p:
        return False
    return re.search(r"(?<![a-z0-9])" + re.escape(p) + r"(e?s)?(?![a-z0-9])", _norm(text)) is not None


def classify(text: str, row: dict) -> str:
    if not text or text == CANT:
        return "cant"
    for kind in ORDER:
        if any(_has(text, w) for w in row.get(kind, [])):
            return kind
    return "unclassified"


def cmd_score(a):
    answers = json.loads(Path(a.answers).read_text(encoding="utf-8"))["answers"]
    key = json.loads(Path(a.key).read_text(encoding="utf-8"))
    sheet = json.loads(Path(a.sheet).read_text(encoding="utf-8"))
    over = json.loads(Path(a.overrides).read_text(encoding="utf-8")) if a.overrides else {}
    out, left = {}, []
    for code, k in sorted(key.items()):
        text = answers.get(code, "")
        row = sheet[str(k["i"])]
        auto = classify(text, row) if text else "missing"
        final = over.get(code, {}).get("class", auto) if auto == "unclassified" or code in over else auto
        out[code] = {"i": k["i"], "text": text, "source": row["source"], "family": row["family"],
                     "auto": auto, "class": final, "why": over.get(code, {}).get("why", "")}
        if final == "unclassified":
            left.append(code)
    dst = Path(a.out or Path(a.answers).with_name(Path(a.answers).stem + "_scored.json"))
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    counts = {}
    for r in out.values():
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"{len(out)} cards -> {dst}   {counts}")
    for code in left:
        r = out[code]
        print(f"   UNCLASSIFIED {code}  sound: {r['source']} ({r['family']})  answer: {r['text']!r}")


def cmd_check(a):
    """The checker's answers, classified with the same sheet BEFORE Adam's file is opened (frozen rule).
    Printed as totals only, never per arm."""
    chk = json.loads(Path(a.check).read_text(encoding="utf-8"))
    sheet = json.loads(Path(a.sheet).read_text(encoding="utf-8"))
    out = {}
    for k, c in chk.items():
        i = k.split(":")[1]
        obj, snd = c.get("object", ""), c.get("sound", "")
        if not obj and c.get("raw"):                  # replies saved before the box-token parser fix
            from benchmark.gold.picture_bench import _two_lines
            obj, snd = _two_lines(c["raw"])
        txt = (obj + " ; " + snd).strip(" ;")
        out[k] = {"text": txt, "class": classify(txt, sheet[i])}
    dst = Path(a.out or Path(a.check).with_name(Path(a.check).stem + "_scored.json"))
    dst.write_text(json.dumps(out, indent=1), encoding="utf-8")
    counts = {}
    for r in out.values():
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    print(f"{len(out)} checker answers -> {dst}   {counts}")


def _boot(diff: np.ndarray, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), size=(n, len(diff)))
    m = diff[idx].mean(axis=1)
    return float(diff.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def _kappa(a: list, b: list) -> float:
    a, b = np.array(a, bool), np.array(b, bool)
    po = float((a == b).mean())
    pe = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


def cmd_unseal(a):
    scored = json.loads(Path(a.scored).read_text(encoding="utf-8"))
    arms = json.loads(Path(a.arms).read_text(encoding="utf-8"))
    good = {"correct", "narrower"}
    first, repeats = {}, []
    for code in sorted(scored):
        k = (arms[code]["arm"], arms[code]["i"])
        if k in first:
            repeats.append((first[k], code))
        else:
            first[k] = code
    right = {k: scored[c]["class"] in good for k, c in first.items()}
    arm_names = sorted({k[0] for k in first})
    print("per arm (repeats excluded): correct incl. narrower | narrower | vague | wrong | can't | unclassified")
    for arm in arm_names:
        cs = [scored[c]["class"] for k, c in first.items() if k[0] == arm]
        n = len(cs)
        print(f"   {arm:6s} {sum(x in good for x in cs):3d}/{n}  {cs.count('narrower'):3d}  {cs.count('vague'):3d}"
              f"  {cs.count('wrong'):3d}  {cs.count('cant'):3d}  {cs.count('unclassified'):3d}")
    sounds = sorted({k[1] for k in first})
    fam_diff = {int(scored[c]["i"]): scored[c]["source"] != scored[c]["family"] for c in scored}
    for x, y in [("N", "today"), ("N", "N0"), ("N0", "today")]:
        if x not in arm_names or y not in arm_names:
            continue
        both = [i for i in sounds if (x, i) in right and (y, i) in right]
        d = np.array([right[(x, i)] - right[(y, i)] for i in both], float)
        m, lo, hi = _boot(d)
        print(f"   {x} - {y}: {m:+.3f} [{lo:+.3f}, {hi:+.3f}] over {len(both)} sounds")
        for label, sel in [("source differs from family", True), ("source = family", False)]:
            sub = np.array([right[(x, i)] - right[(y, i)] for i in both if fam_diff.get(i) == sel], float)
            if len(sub):
                m, lo, hi = _boot(sub)
                print(f"        {label:28s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] n={len(sub)}")
    agree = [scored[p]["class"] == scored[q]["class"] for p, q in repeats]
    agree_b = [(scored[p]["class"] in good) == (scored[q]["class"] in good) for p, q in repeats]
    if agree:
        print(f"repeats: same class {sum(agree)}/{len(agree)}, same right/not-right {sum(agree_b)}/{len(agree_b)}"
              f" ({100 * sum(agree_b) / len(agree_b):.0f}%; below 80% the round is inconclusive)")
    wrong_n = [(k, scored[c]) for k, c in first.items() if k[0] == "N" and scored[c]["class"] == "wrong"]
    for k, r in wrong_n:
        print(f"   LOOK (possible false message) N:{k[1]} sound {r['source']}: {r['text']!r}")
    if a.check and Path(a.check).exists():
        chk = json.loads(Path(a.check).read_text(encoding="utf-8"))     # the cmd_check output
        adam, mach = [], []
        for k in first:
            c = chk.get(f"{k[0]}:{k[1]}")
            if not c:
                continue
            adam.append(right[k])
            mach.append(c["class"] in good)
        if adam:
            adam_a, mach_a = np.array(adam), np.array(mach)
            catch = float((~mach_a[~adam_a]).mean()) if (~adam_a).any() else float("nan")
            false_rej = float((~mach_a[adam_a]).mean()) if adam_a.any() else float("nan")
            kap = _kappa(adam, mach)
            ok = catch >= 0.70 and false_rej <= 0.15 and kap >= 0.5
            print(f"checker vs Adam over {len(adam)} pictures: catches {catch:.0%} of his not-right (bar 70%), "
                  f"rejects {false_rej:.0%} of his right (bar 15%), kappa {kap:.2f} (bar 0.5) -> "
                  f"{'EARNS A VOTE' if ok else 'stays a logged column'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["check", "score", "unseal"])
    ap.add_argument("--answers")
    ap.add_argument("--key")
    ap.add_argument("--sheet")
    ap.add_argument("--overrides", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--scored")
    ap.add_argument("--arms")
    ap.add_argument("--check", default="")
    a = ap.parse_args()
    {"check": cmd_check, "score": cmd_score, "unseal": cmd_unseal}[a.cmd](a)


if __name__ == "__main__":
    main()
