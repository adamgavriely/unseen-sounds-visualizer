"""Cache fill 1 Oct (docs/prereg_round13_detector_push.md, "Cache fill 1 Oct (not a round; evaluation gap)"): the listener
answers the shipped arm SHIP8+MD3 asks for but finds no cached item for (stage-4 LISTENER_STATS a_missing_list = band runs,
P2 lookup; b_missing_list = PANNs-vetoed FlexSED spans, PV lookup). The live path (src/listener_prep.py) asks every run, so
these are an evaluation gap, not a rule change. Same steps as the Round 14 K2 supplement (benchmark/gold/listener_k2.py):
  a-items  built exactly as listener_k2.pool (P2; run bounds asserted within 0.02 s of the asked run)
  b-items  built exactly as listener_variants.build's PV branch (the vetoed span, asked on its containing P2 run of the base
           cache, else run_from_flex)
  scored   listener_variants.score (Qwen3-Omni V1-V4) and listener_afnext.run (AF Next V4 + yes/no), code imported unchanged
  merged   the scored items are APPENDED to the base caches (<split>_listener_v.json, <split>_listener_afn.json; backups kept)
  replay   every listed item is found by the stage-4 lookup (listener_from_vcache) with a TIER answer and an AF answer
  reset    stage4.json entries + arm folders of the affected clips are backed up and removed, so stage 4/5 recompute them
  post     pictures before/after per affected clip (gold-scored classes); clips whose drawn pictures changed lose their
           DEPICT / GROUP cache entries (backup) so the existing depict.py / group.py steps re-ask them
No new prompt, model or rule. DEV / DEV2 only; TEST is never read. Run from ~/MscProj_r13 (dev) or ~/MscProj_tg (dev2):

    python benchmark/gold/listener_fill.py --split dev pool|score|merge|replay|reset|post|missing
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import dev_candidates_check as C
from benchmark.gold import dev_listener as L
from benchmark.gold import listener_variants as LV
from benchmark.gold import score_per_sound as S
from src.labels import canonical

GOLD = _ROOT / "benchmark" / "gold"
ARM = "SHIP8+MD3"
SYSTEMS = ("proposed", "blind_a2i")
BAK = C.WORK / "cachefill_bak_1oct"
TOL = 0.02
CFG = {"dev": {"base": GOLD / "dev_listener.json", "v": GOLD / "dev_listener_v.json", "afn": GOLD / "dev_listener_afn.json",
               "wav": C.WORK / "devcand" / "wav16", "r13": C.WORK / "r13"},
       "dev2": {"base": GOLD / "dev2_listener.json", "v": GOLD / "dev2_listener_v.json", "afn": GOLD / "dev2_listener_afn.json",
                "wav": C.WORK / "r13dev2" / "wav16", "r13": C.WORK / "r13dev2"}}


def fill_q(split):
    return GOLD / f"{split}_listener_fill.json"


def fill_a(split):
    return GOLD / f"{split}_listener_fill_afn.json"


def missing(split, stage4=None):
    """the arm's listed missing items, both systems, deduplicated: [(kind, clip, label, start, end)]"""
    s4 = json.loads(Path(stage4 or CFG[split]["r13"] / "stage4.json").read_text(encoding="utf-8"))
    out = []
    for k, v in sorted(s4.get("listener", {}).items()):
        arm, sysn, clip = k.split("|")
        if arm != ARM:
            continue
        for kind in ("a", "b"):
            for x in v.get(f"{kind}_missing_list", []):
                t = (kind, clip, x[0], float(x[1]), float(x[2]))
                if t not in out:
                    out.append(t)
    return out


def _key(x):
    return (x["clip"], x.get("pool"), x["family"], x.get("label"), round(x["start"], 2), round(x["end"], 2))


# ============================================================================= pool (CPU)
def pool(split):
    import soundfile as sf
    S.load_gold = LV._no_gold
    c = CFG[split]
    todo = missing(split)
    p2 = [x for x in json.loads(c["base"].read_text(encoding="utf-8"))["items"] if x["pool"] == "P2"]
    rng = random.Random(0)
    O = LV.Onto()
    out, flex = [], {}
    for kind, st, lab, a, b in todo:
        dur = float(sf.info(str(c["wav"] / f"{st}.wav")).duration)
        fam = canonical(lab)
        if kind == "a":                                   # listener_k2.pool
            run = LV.run_from_flex(st, lab, a, b, dur)
            assert run is not None, (st, lab, a, b)
            assert abs(run["start"] - a) <= TOL and abs(run["end"] - b) <= TOL, (st, lab, a, b, run)
            it = {"clip": st, "pool": "P2", "family": fam, "label": lab, "start": run["start"], "end": run["end"],
                  "cut_start": run["cut_start"], "cut_end": run["cut_end"], "run_len": run["run_len"], "peak": run["peak"],
                  "depictable": L.depictable(lab), "run_start": run["start"], "run_end": run["end"],
                  "variants": ["V1", "V2", "V3", "V4"], "source": "cache fill 1 Oct (a_missing of SHIP8+MD3)"}
            it["null_family"] = LV.gold_free_null(st, rng)
        else:                                             # listener_variants.build, PV branch
            run = LV.contain(p2, st, fam, a, b)
            it = {"clip": st, "pool": "PV", "family": fam, "label": lab, "start": a, "end": b,
                  "depictable": L.depictable(lab), "variants": ["V1", "V2", "V3", "V4"],
                  "source": "cache fill 1 Oct (b_missing of SHIP8+MD3)"}
            if run:
                it.update({"run_start": run["start"], "run_end": run["end"], "cut_start": run["cut_start"],
                           "cut_end": run["cut_end"], "run_len": run["run_len"], "peak": run["peak"], "run_from_cache": True,
                           "cached_score": run.get("score")})
            else:
                r = LV.run_from_flex(st, lab, a, b, dur)
                assert r is not None, (st, lab, a, b)
                it.update({"run_start": r["start"], "run_end": r["end"], "cut_start": r["cut_start"], "cut_end": r["cut_end"],
                           "run_len": r["run_len"], "peak": r["peak"], "run_from_cache": False, "cached_score": None})
            it["null_family"] = (run or {}).get("null_family") or LV.gold_free_null(st, rng)
        for tag, f in (("x", it["family"]), ("null", it["null_family"])):
            ds, src = LV.distractors(O, st, f, it["start"])
            it[f"v1_{tag}_options"], it[f"v1_{tag}_source"] = ds, src
        if st not in flex:
            flex[st] = C.load_fr(LV.FLEX_DIR / f"{st}.npz")
        a0 = max(0.0, it["cut_start"] - 1.0); b0 = min(dur, it["cut_end"] + 1.0)
        it["run_audio"] = [a0, max(b0, a0 + 1.0)]
        for tag, f in (("x", it["family"]), ("null", it["null_family"])):
            it[f"v2_{tag}_ctrl"], it[f"v2_{tag}_flex_col"] = LV.control(flex[st], f, it, dur)
        it["v3_cut"] = [max(0.0, it["run_start"] - 3.0), min(dur, it["run_end"] + 3.0)]
        out.append(it)
    have = {_key(x) for x in json.loads(c["v"].read_text(encoding="utf-8"))["items"]}
    dup = [_key(x) for x in out if _key(x) in have]
    assert not dup, f"already in the base cache: {dup}"
    meta = {"step": "cache fill 1 Oct", "split": split, "source": f"{c['r13'] / 'stage4.json'} listener[{ARM}|*] missing lists",
            "model": LV.MODEL, "V1": LV.V1_Q, "V3": LV.V3_Q, "V4": LV.V4_Q,
            "null": "listener_variants.gold_free_null, random.Random(0)", "n": len(out)}
    C.dump(fill_q(split), {"_meta": meta, "items": out})
    print(f"[fill pool] {split}: {len(out)} items (P2 {sum(x['pool'] == 'P2' for x in out)}, PV "
          f"{sum(x['pool'] == 'PV' for x in out)}) -> {fill_q(split)}", flush=True)
    for x in out:
        print(f"   {x['clip']} {x['pool']} {x['label']} {x['start']:.2f}-{x['end']:.2f} run {x['run_start']:.2f}-"
              f"{x['run_end']:.2f} cut {x['cut_start']:.2f}-{x['cut_end']:.2f} peak {x['peak']:.3f} null {x['null_family']}",
              flush=True)


# ============================================================================= score (GPU, one model at a time)
def score(split):
    import gc
    import torch
    from benchmark.gold import listener_afnext as LA
    S.load_gold = LV._no_gold
    LV.SPLITS = {split: {"cache": None, "out": fill_q(split), "wav": CFG[split]["wav"]}}
    LV.score()
    gc.collect(); torch.cuda.empty_cache()
    M = LA.load_model()
    if fill_a(split).exists():
        d = json.loads(fill_a(split).read_text(encoding="utf-8"))
        items, meta = d["items"], d["_meta"]
    else:
        items, meta = LA.base_items(split), LA.make_meta(M, split)
    LA.run(M, items, split, fill_a(split), meta)
    LA.summary(items, split)


# ============================================================================= merge (CPU): append to the base caches
def merge(split):
    c = CFG[split]
    for src, dst in ((fill_q(split), c["v"]), (fill_a(split), c["afn"])):
        new = json.loads(src.read_text(encoding="utf-8"))["items"]
        assert new and all("accept" in x for x in new), f"{src}: not fully scored"
        d = json.loads(dst.read_text(encoding="utf-8"))
        have = {_key(x) for x in d["items"]}
        add = [x for x in new if _key(x) not in have]
        if not add:
            print(f"[fill merge] {dst.name}: all {len(new)} already appended", flush=True)
            continue
        assert len(add) == len(new), f"{dst}: partly appended before"
        bak = BAK / split / dst.name
        bak.parent.mkdir(parents=True, exist_ok=True)
        if not bak.exists():
            shutil.copy2(dst, bak)
        n0 = len(d["items"])
        d["items"] = d["items"] + add
        d.setdefault("_meta", {})["cache_fill_1oct"] = (f"{len(add)} items appended from {src.name} (docs/prereg_round13_"
                                                        "detector_push.md, Cache fill 1 Oct)")
        C.dump(dst, d)
        assert len(json.loads(dst.read_text(encoding="utf-8"))["items"]) == n0 + len(add)
        print(f"[fill merge] {dst}: {n0} + {len(add)} items (backup {bak})", flush=True)


# ============================================================================= replay (CPU): the stage-4 lookup finds them
def replay(split):
    import config
    from src import stage4_audio_event_detection as S4
    c = CFG[split]
    config.LISTENER_AFCACHE = str(c["afn"])
    looks, bad, rows = {}, [], []
    for kind, st, lab, a, b in missing(split):
        if st not in looks:
            looks[st] = S4.listener_from_vcache(str(c["v"]), st)
        it = looks[st](lab, a, b, contain=(kind == "b"))
        acc = (it or {}).get("accept") or {}
        ok = it is not None and "TIER" in acc and not acc.get("AF_missing", True)
        rows.append([kind, st, lab, a, b, ok, {k: acc.get(k) for k in ("V4", "AF_V4", "TIER")}])
        if not ok:
            bad.append((kind, st, lab, a, b))
    for r in rows:
        print("   [replay]", *r, flush=True)
    assert not bad, f"not found by the lookup: {bad}"
    print(f"[fill replay] {split}: {len(rows)} / {len(rows)} found (TIER + AF answer)", flush=True)


# ============================================================================= reset (CPU): recompute these clips
def clips(split):
    return sorted({t[1] for t in missing(split)})


def reset(split):
    c = CFG[split]
    s4p = c["r13"] / "stage4.json"
    cl = clips(split)
    BAK.joinpath(split).mkdir(parents=True, exist_ok=True)
    lst = BAK / split / "clips.json"
    if not lst.exists():
        lst.write_text(json.dumps(cl), encoding="utf-8")
    bak = BAK / split / "stage4.json"
    if not bak.exists():
        shutil.copy2(s4p, bak)
    s4 = json.loads(s4p.read_text(encoding="utf-8"))
    n = 0
    for sysn in SYSTEMS:
        k = f"{ARM}|{sysn}"
        for st in cl:
            if st in s4["arms"].get(k, {}):
                del s4["arms"][k][st]; n += 1
            for sec in ("listener", "f7", "mirror"):
                s4.get(sec, {}).pop(f"{k}|{st}", None)
            d = c["r13"] / f"{ARM}_{sysn}" / st
            if d.exists():
                dst = BAK / split / f"{ARM}_{sysn}" / st
                dst.parent.mkdir(parents=True, exist_ok=True)
                assert not dst.exists(), dst
                shutil.move(str(d), str(dst))
    C.dump(s4p, s4)
    print(f"[fill reset] {split}: {len(cl)} clips {cl}; {n} stage-4 entries removed; folders -> {BAK / split}", flush=True)


# ============================================================================= post (CPU): picture diff + DEPICT/GROUP reset
def _gold(split):
    if split == "dev":
        gold, _st = C.dev_stems()
        return gold
    from benchmark.gold import tagger_prep as T
    DCC2, _R2, stems = T.configure("dev2")
    keep = set(stems)
    d = json.loads(T.TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [x for x in d.get("clips", []) if isinstance(x, dict) and Path(str(x.get("clip", ""))).stem in keep]
    tmp = BAK / split / "dev2_gold_only.json"
    DCC2.dump(tmp, d)
    return T._REAL_LOAD_GOLD([tmp])


def _drawn(d):
    f = d / "augmentations.json"
    if not f.exists():
        return None
    return sorted((s["event_label"], round(float(s["start"]), 2), round(float(s["end"]), 2))
                  for s in json.loads(f.read_text(encoding="utf-8")) if s.get("augment") and s.get("image_path"))


def post(split, apply=True):
    from benchmark.gold import round13_dev as R
    c = CFG[split]
    cl = json.loads((BAK / split / "clips.json").read_text(encoding="utf-8"))
    gold = _gold(split)
    sysn = "proposed"
    new_root, old_root = c["r13"] / f"{ARM}_{sysn}", BAK / split / f"{ARM}_{sysn}"
    changed, rep = [], {}
    with R.flags({k: R.arm_cfg(ARM)[k] for k in R.DISPLAY_KEYS}):
        for st in cl:
            dn, do = _drawn(new_root / st), _drawn(old_root / st)
            assert dn is not None, f"{st}: no new stage-5 output"
            po = S.load_pictures(old_root, st, sysn) or []
            pn = S.load_pictures(new_root, st, sysn) or []
            so = S.score_clip(gold[st], po) if st in gold else None
            sn = S.score_clip(gold[st], pn) if st in gold else None
            cls = lambda s: {k: s[k] for k in ("hit", "visible", "cross", "phantom")} if s else None
            rep[st] = {"drawn_changed": dn != do, "pictures_old": po, "pictures_new": pn, "old": cls(so), "new": cls(sn)}
            if dn != do:
                changed.append(st)
            print(f"[fill post] {split} {st}: drawn {'CHANGED' if dn != do else 'same'}; pictures {po} -> {pn}; "
                  f"{cls(so)} -> {cls(sn)}", flush=True)
    out = BAK / split / "post.json"
    out.write_text(json.dumps({"changed": changed, "clips": rep}, indent=1, default=float), encoding="utf-8")
    if apply and changed:
        for name in ("depict_answers.json", "group_answers.json"):
            p = C.WORK / name
            d = json.loads(p.read_text(encoding="utf-8"))
            b = BAK / split / name
            if not b.exists():
                shutil.copy2(p, b)
            rm = [st for st in changed if st in d]
            for st in rm:
                d.pop(st)
            p.write_text(json.dumps(d, indent=1), encoding="utf-8")
            print(f"[fill post] {name}: removed {rm} (re-asked next by the existing step)", flush=True)
    print(f"[fill post] {split}: drawn pictures changed on {changed}", flush=True)


def show_missing(split):
    m = missing(split)
    print(f"[fill missing] {split} {ARM}: a {sum(t[0] == 'a' for t in m)} b {sum(t[0] == 'b' for t in m)} {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=("dev", "dev2"), required=True)
    ap.add_argument("step", choices=("pool", "score", "merge", "replay", "reset", "post", "diff", "missing"))
    a = ap.parse_args()
    if a.step == "diff":
        post(a.split, apply=False)
    else:
        {"pool": pool, "score": score, "merge": merge, "replay": replay, "reset": reset, "post": post,
         "missing": show_missing}[a.step](a.split)


if __name__ == "__main__":
    main()
