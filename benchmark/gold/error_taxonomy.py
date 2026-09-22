"""Where every wrong picture comes from (2026-09-23; the four reviewers' top thesis lever).

The system shows a picture; the picture is wrong. Which stage is to blame? Each false alarm of a
system is put in exactly one bucket, decided from the annotator's own gold:

  invented      no sound of that family is in the clip at all -> the DETECTOR named something that
                was never there
  gate-leak     the sound is real and sounding at that moment, but the annotator ticked it
                visible or obvious -> the GATE should have silenced it
  wrong-family  a real sound is there at that moment, but of another family -> the DETECTOR
                mislabelled what it heard
  late          the sound is real, needed, and the picture is of the right family, but it starts
                outside the [-0.5, +1.0] s window -> a TIMING loss, not a content error
  level-1       the sound is real and needed but the annotator rated it 1 ("the noise of the
                place") -> scored as a false alarm only because level-1 sounds are don't-care

A second, GPU pass asks a VLM whether the generated image actually shows what its label says, which
is the only way to tell a correct decision from a correct picture.

    python benchmark/gold/error_taxonomy.py --tag v4b4              # CPU
    python benchmark/gold/error_taxonomy.py --tag v4b4 --pictures   # + the VLM picture check (GPU)
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S

GOLD = _ROOT / "benchmark" / "gold" / "annotations" / "gold_AG.json"
EARLY, LATE = 0.5, 1.0


def bucket(lab, a, b, snds):
    """one of the five causes, decided from the gold alone"""
    fam = [s for s in snds if S.same_family(lab, s["label"])]
    if not fam:
        return "invented"
    # sounding at the moment the picture appears (a second of slack either way)
    now = [s for s in fam if s["start"] - 1.0 <= b and a <= s["end"] + 1.0]
    if not now:
        near = [s for s in fam if abs(s["start"] - a) <= 8.0]
        return "late" if near else "invented"
    if all(not s["needed"] for s in now):
        return "gate-leak"
    needed_now = [s for s in now if s["needed"]]
    if all(s["importance"] < 2 for s in needed_now):
        return "level-1"
    if any(S.in_window(a, s["start"], EARLY, LATE) for s in needed_now):
        return "counted-hit"                     # should not happen: it would have been scored a hit
    return "late"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v4b4")
    ap.add_argument("--systems", nargs="+", default=["proposed", "blind_a2i"])
    ap.add_argument("--subset", default="bench")
    ap.add_argument("--pictures", action="store_true", help="also ask a VLM whether each image shows its label (GPU)")
    ap.add_argument("--vlm", default=None)
    a = ap.parse_args()
    config.use_v4("59")
    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    w = _ROOT / "data" / "work"
    out = {}
    shown = defaultdict(list)
    for system in a.systems:
        root = w / f"protocol_{system}_{a.tag}"
        cnt, wrongfam = Counter(), Counter()
        for stem in sorted(subs[a.subset]):
            pics = S.load_pictures(root, stem, system)
            if pics is None:
                continue
            snds = gold[stem]
            for lab, x, y in pics:
                if any(S.same_family(lab, s["label"]) and S.in_window(x, s["start"], EARLY, LATE)
                       and s["needed"] and s["importance"] >= 2 for s in snds):
                    cnt["RIGHT"] += 1
                    shown[system].append((stem, lab, x, y, "RIGHT"))
                    continue
                # a real sound of another family at that moment?
                other = [s for s in snds if not S.same_family(lab, s["label"]) and s["start"] - 1.0 <= y and x <= s["end"] + 1.0]
                bk = bucket(lab, x, y, snds)
                if bk == "invented" and other:
                    bk = "wrong-family"
                cnt[bk] += 1
                if bk in ("invented", "wrong-family"):
                    wrongfam[lab] += 1
                shown[system].append((stem, lab, x, y, bk))
        total = sum(cnt.values())
        wrong = total - cnt["RIGHT"]
        print(f"== {system} ({a.tag}, {a.subset}): {total} pictures, {cnt['RIGHT']} right ({cnt['RIGHT'] / max(1, total):.0%}), {wrong} wrong")
        for k in ("invented", "wrong-family", "gate-leak", "late", "level-1", "counted-hit"):
            if cnt[k]:
                print(f"     {k:13s} {cnt[k]:3d}  = {cnt[k] / max(1, wrong):.0%} of the wrong ones")
        print(f"     most-invented labels: {wrongfam.most_common(8)}")
        out[system] = {"total": total, "right": cnt["RIGHT"], **{k: cnt[k] for k in cnt}}
    (_ROOT / "benchmark" / "gold" / f"error_taxonomy_{a.tag}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")

    if a.pictures:
        check_pictures(a, gold, subs, w, shown)


def check_pictures(a, gold, subs, w, shown):
    """does the generated image actually show what its label says? (the only test of the picture itself)"""
    from src.stage5_cross_modal_analysis import reason
    model = a.vlm or config.VLM_MODEL
    mdl, proc = reason._load(model, "cuda")
    res = defaultdict(Counter)
    detail = []
    for system, items in shown.items():
        root = w / f"protocol_{system}_{a.tag}"
        for stem, lab, x, y, bk in items:
            f = root / stem / "augmentations.json"
            if not f.exists():
                continue
            spec = next((s for s in json.loads(f.read_text(encoding="utf-8"))
                         if s.get("augment") and s.get("event_label") == lab and abs(float(s.get("start", -9)) - x) < 2.5), None)
            img = spec and spec.get("image_path")
            if not img or not Path(img).exists():
                p2 = root / stem / "augmentations" / Path(img).name if img else None
                img = str(p2) if p2 and p2.exists() else None
            if not img:
                continue
            from PIL import Image
            ans = reason._ask(mdl, proc, f"Does this picture show {lab}? Answer yes or no.",
                              images=[Image.open(img).convert("RGB")], max_new=6).strip().lower()
            ok = ans.startswith("y")
            res[system][f"{bk}:{'ok' if ok else 'bad'}"] += 1
            detail.append({"system": system, "clip": stem, "label": lab, "bucket": bk, "picture_shows_label": ok})
    for system, c in res.items():
        tot_ok = sum(v for k, v in c.items() if k.endswith(":ok"))
        tot = sum(c.values())
        print(f"== {system}: the image shows its own label in {tot_ok}/{tot} = {tot_ok / max(1, tot):.0%} of shown pictures")
        for bk in ("RIGHT", "invented", "wrong-family", "gate-leak", "late", "level-1"):
            ok, bad = c.get(f"{bk}:ok", 0), c.get(f"{bk}:bad", 0)
            if ok + bad:
                print(f"     {bk:13s} image matches its label in {ok}/{ok + bad} = {ok / (ok + bad):.0%}")
    (_ROOT / "benchmark" / "gold" / f"picture_check_{a.tag}.json").write_text(json.dumps(detail, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
