"""The per-picture judge, PP-1 (docs/prereg_per_picture_judge.md, frozen before any number).

Adam: score every picture, and count the bad ones. The judge sees each picture ALONE -- no video, no
sound name, no reference -- and answers the question Adam answers in his blind ratings. Everything after
that is mechanical and uses only the annotator's ticks: which gold sound the answer matches, whether that
sound was needed, and what the picture costs a deaf viewer.

    answers  (GPU)  python benchmark/gold/judge_per_picture.py answers --tags v4b4 dev_monocap_v31 --bench-fresh
    score    (CPU)  python benchmark/gold/judge_per_picture.py score --tag dev_monocap_v31 --subset dev
    repeat   (GPU)  python benchmark/gold/judge_per_picture.py answers --tags v4b4 --repeat 20

Captions are read mechanically (the tag text is the answer); nothing about them goes through the judge.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S           # noqa: E402
from benchmark.gold.error_taxonomy import GOLD           # noqa: E402
from benchmark.gold.answer_sheet import sheet_for, _children, SYN   # noqa: E402
from benchmark.gold.score_answers import classify        # noqa: E402

JUDGE = "google/gemma-4-31B-it"
PROMPT = "What is making a sound in this picture? Answer in a few words, or answer exactly: can't tell."
CACHE = _ROOT / "benchmark" / "judge_per_picture_answers.json"
SYSTEMS = ("proposed", "blind_a2i", "audio_caption")
BEST = ["correct", "narrower", "vague", "wrong", "cant", "unclassified"]   # a picture takes its best class


def _key(p: Path) -> str:
    return hashlib.sha1(Path(p).read_bytes()).hexdigest()[:16]


def pictures(tag: str, system: str, stem: str):
    """[(label, start, end, image path or None)] exactly as the scorer reconstructs the panel"""
    work = _ROOT / "data" / "work" / f"protocol_{system}_{tag}"
    f = work / stem / "augmentations.json"
    if not f.exists():
        return None
    spans = S.load_pictures(work, stem, system) or []
    imgs = {}
    for s in json.loads(f.read_text(encoding="utf-8")):
        if s.get("augment") and s.get("image_path"):
            p = Path(s["image_path"])
            p = p if p.is_absolute() else _ROOT / p
            imgs.setdefault(s["event_label"], p if p.exists() else None)
    return [(lab, float(a), float(b), None if system == "audio_caption" else imgs.get(lab)) for lab, a, b in spans]


# ------------------------------------------------------------------------------------ answers
def cmd_answers(a):
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    gold = S.load_gold([GOLD])
    todo = {}
    for tag in a.tags:
        for system in ("proposed", "blind_a2i"):
            for stem in sorted(gold)[: a.repeat or None]:
                for lab, x, y, img in (pictures(tag, system, stem) or []):
                    if img is not None:
                        todo[_key(img)] = img
    if a.bench_fresh:                      # the round-2 pictures, for B4 against Adam's blind answers
        bench = _ROOT / "data" / "work" / "picture_bench_fresh"
        from benchmark.gold.picture_bench import picture_paths
        import benchmark.gold.picture_bench as PB
        PB.BENCH = bench
        items = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
        for arm in ("today", "N", "N0"):
            for i, p in picture_paths(arm, items).items():
                p = Path(p)
                if p.exists():
                    todo[_key(p)] = p
    out_f = CACHE.with_name(CACHE.stem + ("_repeat" if a.repeat else "") + ".json")
    cache = json.loads(out_f.read_text(encoding="utf-8")) if out_f.exists() else {}
    proc = AutoProcessor.from_pretrained(JUDGE)
    mdl = AutoModelForImageTextToText.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map="auto").eval()
    torch.manual_seed(1)
    kw = {"do_sample": True, "temperature": 1.0} if a.repeat else {"do_sample": False}
    for n, (k, p) in enumerate(sorted(todo.items()), 1):
        if k in cache:
            continue
        img = Image.open(p).convert("RGB").resize((512, 512))
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": PROMPT}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True,
                                       return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**inp, max_new_tokens=24, **kw)
        ans = proc.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        cache[k] = {"answer": ans, "path": str(p)}
        if n % 20 == 0:
            out_f.write_text(json.dumps(cache, indent=1), encoding="utf-8")
        print(f"  {n}/{len(todo)} {ans[:60]}", flush=True)
    out_f.write_text(json.dumps(cache, indent=1), encoding="utf-8")
    print(f"{len(cache)} answers -> {out_f}")


# -------------------------------------------------------------------------------------- score
def _cls(answer: str, g, syn, kids) -> str:
    if not answer or re.sub(r"[^a-z]", "", answer.lower()) in ("canttell", "cannottell"):
        return "cant"
    from src.labels import canonical
    return classify(answer, sheet_for(g["label"], canonical(g["label"]), syn, kids))


def score_clip(snds, pics, answers, syn, kids, vague_covers=0.0):
    """-> (cost, counts) for one clip, per the frozen rule (docs/prereg_per_picture_judge.md)"""
    need = [g for g in snds if g["needed"] and g["importance"] >= 2]
    covered = {id(g): 0.0 for g in need}           # 1.0 = a correct picture; vague_covers for a vague one
    touched = set()                                 # needed sounds that already have a picture
    counts = {k: 0 for k in ("good", "vague", "wrong", "cant", "unclassified", "on_screen", "obvious",
                             "duplicate", "dont_care", "late", "missed")}
    cost = 0.0
    for lab, x, y, ans in sorted(pics, key=lambda p: p[1]):
        cands = [g for g in snds if S.in_window(x, g["start"], S.EARLY, S.LATE)]
        if not cands:
            # matches no gold sound in time: unnecessary; a correct picture of a sound heard at another time
            # is logged as late, with its cost unchanged
            classes = [_cls(ans, g, syn, kids) for g in snds] or ["unclassified"]
            if any(c in ("correct", "narrower") for c in classes):
                counts["late"] += 1
            counts["cant" if "cant" in classes else "wrong"] += 1
            cost += 2
            continue
        rank, g = min(((BEST.index(_cls(ans, g, syn, kids)), i, g) for i, g in enumerate(cands)),
                      key=lambda t: (t[0], t[1]))[0::2]
        cls = BEST[rank]
        if cls in ("correct", "narrower", "vague"):
            if g["needed"] and g["importance"] >= 2:
                if id(g) in touched and (cls == "vague" or covered[id(g)] >= 1.0):
                    counts["duplicate"] += 1        # a second picture of the same sound costs nothing
                elif cls == "vague":
                    covered[id(g)] = max(covered[id(g)], vague_covers)
                    counts["vague"] += 1
                else:
                    if covered[id(g)] > 0 or id(g) in touched:
                        counts["vague"] -= 1        # a correct picture replaces an earlier vague one
                    covered[id(g)] = 1.0
                    counts["good"] += 1
                touched.add(id(g))
            elif g["needed"]:
                counts["dont_care"] += 1            # importance-1 needed sound: costs 0, covers nothing
            else:
                counts["on_screen" if g.get("visible") else "obvious"] += 1
                cost += 2
        else:
            counts[cls if cls in ("cant", "unclassified") else "wrong"] += 1
            cost += 2
    missed = sum(1.0 - covered[id(g)] for g in need)
    # in the counts a needed sound is "missed" only when it got no picture at all; a vague one is counted as
    # vague (the cost still charges it as uncovered under the primary rule)
    counts["missed"] = sum(1 for g in need if id(g) not in touched)
    return cost + 4 * missed, counts


def cmd_score(a):
    gold = S.load_gold([GOLD])
    subset = set(S.subsets_of(gold)[a.subset]) if a.subset != "all" else set(gold)
    syn = json.loads(SYN.read_text(encoding="utf-8"))
    syn = {k: v for k, v in syn.items() if not k.startswith("_")}
    kids = _children()
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    rows = {}
    for system in list(SYSTEMS) + ["silence"]:
        for stem in sorted(subset):
            if system == "silence":
                pics = []
            else:
                p = pictures(a.tag, system, stem)
                if p is None:
                    continue
                pics = []
                for lab, x, y, img in p:
                    if system == "audio_caption":
                        ans = lab                       # the tag text is the answer (mechanical)
                    elif img is None:
                        ans = ""
                    else:
                        ans = cache.get(_key(img), {}).get("answer", "")
                    pics.append((lab, x, y, ans))
            for v in (0.0, 0.5, 1.0):
                c, n = score_clip(gold[stem], pics, cache, syn, kids, vague_covers=v)
                rows.setdefault((system, v), {})[stem] = (c, n)
    clips = sorted(set.intersection(*[set(rows[(s, 0.0)]) for s in SYSTEMS]))
    print(f"PP-1 on {a.tag}, subset {a.subset}: {len(clips)} clips with all three arms\n")
    keys = ("good", "missed", "wrong", "on_screen", "obvious", "vague", "cant", "unclassified", "duplicate",
            "dont_care", "late")
    table = {}
    locked = not a.unlock
    if locked:
        print("ranking LOCKED: no per-arm number is shown until B4 (agreement with Adam's blind ratings)")
        print("passes -- docs/prereg_per_picture_judge.md. Trust checks only.")
    else:
        print(f"{'arm':14s} " + " ".join(f"{k[:9]:>9s}" for k in keys) + "   cost/clip")
    for system in list(SYSTEMS) + ["silence"]:
        tot = {k: sum(rows[(system, 0.0)][c][1][k] for c in clips) for k in keys}
        cost = np.mean([rows[(system, 0.0)][c][0] for c in clips])
        table[system] = {"counts": tot, "cost_per_clip": float(cost)}
        if not locked:
            print(f"{system:14s} " + " ".join(f"{tot[k]:9d}" for k in keys) + f"   {cost:8.2f}")
    if not locked:
        print("\npaired clip bootstrap of cost (2000, seed 0), ours minus other; lower is better for ours")
    for v in ((0.0, 0.5, 1.0) if not locked else ()):
        line = []
        for other in ("blind_a2i", "audio_caption", "silence"):
            d = np.array([rows[("proposed", v)][c][0] - rows[(other, v)][c][0] for c in clips])
            rng = np.random.default_rng(0)
            bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
            lo, hi = np.percentile(bs, [2.5, 97.5])
            line.append(f"vs {other[:7]} {d.mean():+.2f} [{lo:+.2f}, {hi:+.2f}]{'*' if lo > 0 or hi < 0 else ' '}")
        print(f"   vague counts {v:.1f}:  " + "   ".join(line))
    # trust checks B1 and B2 against the annotator's own viewer cost, per arm
    print("\ntrust checks")
    from benchmark.gold.judge_trust import spearman, boot
    for system in ("proposed", "blind_a2i"):
        ann, pp, wrongclip = [], [], []
        for c in clips:
            pics = S.load_pictures(_ROOT / "data" / "work" / f"protocol_{system}_{a.tag}", c, system)
            r = S.score_clip(gold[c], pics)
            ann.append(S.COST_MISS * r["miss"] + S.COST_FA * (r["visible"] + r["cross"] + r["phantom"]))
            pp.append(rows[(system, 0.0)][c][0])
            wrongclip.append((r["visible"] + r["cross"] + r["phantom"]) > 0)
        rho = spearman(ann, pp)
        lo, hi = boot(lambda ps: spearman([x for x, _ in ps], [y for _, y in ps]), list(zip(ann, pp)))
        print(f"   B1 {system:10s} rho(PP-1 cost, annotator cost) {rho:+.3f} [{lo:+.3f}, {hi:+.3f}]  "
              f"{'PASS' if rho >= 0.4 and lo > 0 else 'FAIL'} (same direction: both are costs)")
        cl = [p for p, w in zip(pp, wrongclip) if not w]
        wr = [p for p, w in zip(pp, wrongclip) if w]
        if cl and wr:
            ps = [(0, v) for v in cl] + [(1, v) for v in wr]
            gap = float(np.mean(wr) - np.mean(cl))

            def g(sample):
                c0 = [v for k, v in sample if k == 0]
                c1 = [v for k, v in sample if k == 1]
                if not c0 or not c1:
                    raise ValueError
                return float(np.mean(c1) - np.mean(c0))
            glo, ghi = boot(g, ps)
            print(f"   B2 {system:10s} wrong-picture clips cost more: {gap:+.2f} [{glo:+.2f}, {ghi:+.2f}]  "
                  f"{'PASS' if gap > 0 and glo > 0 else 'FAIL'}")
    if not locked:
        out = _ROOT / "benchmark" / f"judge_per_picture_{a.tag}_{a.subset}.json"
        out.write_text(json.dumps({"clips": clips, "table": table}, indent=1), encoding="utf-8")
        print("->", out)


def cmd_b4(a):
    """B4: the judge's answers on the round-2 pictures vs Adam's blind answers, both classified with the
    committed round-2 sheet; the checker's frozen bars. Reported overall and per generator."""
    from benchmark.gold.score_answers import _kappa
    import benchmark.gold.picture_bench as PB
    bench = _ROOT / "data" / "work" / "picture_bench_fresh"
    PB.BENCH = bench
    items = json.loads((bench / "specs.json").read_text(encoding="utf-8"))
    sheet = json.loads((_ROOT / "benchmark/gold/pictures/answer_sheet_picfresh_v32.json").read_text(encoding="utf-8"))
    scored = json.loads((_ROOT / "benchmark/gold/pictures/adam_answers_round2_scored.json").read_text(encoding="utf-8"))
    arms = json.loads((_ROOT / "benchmark/gold/pictures/rate_pictures2_ARM_KEY.json").read_text(encoding="utf-8"))
    cache = json.loads(CACHE.read_text(encoding="utf-8"))
    good = {"correct", "narrower"}
    paths = {arm: PB.picture_paths(arm, items) for arm in ("today", "N", "N0")}
    seen, rows = set(), []
    for code in sorted(scored):
        k = (arms[code]["arm"], arms[code]["i"])
        if k in seen:
            continue                                 # repeats excluded
        seen.add(k)
        p = Path(paths[k[0]][k[1]])
        ans = cache.get(_key(p), {}).get("answer") if p.exists() else None
        if ans is None:
            continue
        low = re.sub(r"[^a-z]", "", ans.lower())
        jc = "cant" if low in ("canttell", "cannottell") else classify(ans, sheet[str(k[1])])
        rows.append((k[0], scored[code]["class"] in good, jc in good))
    for name, sel in [("all", None), ("FLUX (today)", "today"), ("Qwen-Image (N, N0)", ("N", "N0"))]:
        rs = [r for r in rows if sel is None or r[0] == sel or (isinstance(sel, tuple) and r[0] in sel)]
        adam = np.array([r[1] for r in rs]); judge = np.array([r[2] for r in rs])
        catch = float((~judge[~adam]).mean()) if (~adam).any() else float("nan")
        falsrej = float((~judge[adam]).mean()) if adam.any() else float("nan")
        kap = _kappa(list(adam), list(judge))
        ok = catch >= 0.70 and falsrej <= 0.15 and kap >= 0.5
        print(f"B4 {name:20s} n={len(rs):3d}  catches {catch:.0%} of Adam's not-right (bar 70%), rejects "
              f"{falsrej:.0%} of his right (bar 15%), kappa {kap:.2f} (bar 0.5) -> {'PASS' if ok else 'FAIL'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["answers", "score", "b4"])
    ap.add_argument("--tags", nargs="+", default=["v4b4", "dev_monocap_v31"])
    ap.add_argument("--tag", default="dev_monocap_v31")
    ap.add_argument("--subset", default="dev")
    ap.add_argument("--bench-fresh", action="store_true")
    ap.add_argument("--repeat", type=int, default=0)
    ap.add_argument("--unlock", action="store_true", help="show the ranking; only after B4 has passed")
    a = ap.parse_args()
    {"answers": cmd_answers, "score": cmd_score, "b4": cmd_b4}[a.cmd](a)


if __name__ == "__main__":
    main()
