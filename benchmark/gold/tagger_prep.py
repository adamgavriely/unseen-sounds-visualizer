"""Confirmation set 1, REVISED (docs/prereg_round13_detector_push.md): features for Adam's 16 tagger-set clips, split by
sha256(stem) % 2 into DEV2 (benchmark/gold/dev2_stems.txt, 6 clips) and TEST2 (test2_stems.txt, 10 clips, SEALED).

Everything the DEV/TEST harness has, built the same way, one folder per split:
  render   stages 1-6 of the pipeline with config.use_scored() (placeholder pictures: the per-sound score never looks at a
           picture) -> data/work/protocol_{proposed,blind_a2i}_<split>_v33 (media, audio.wav, scene, segments, gate
           votes, onset_trace, augmentations): the "scored render" B0 of the split
  flexsed  FlexSED 215 families (benchmark/gold/flexsed_run.py --clip-dir, unchanged) -> data/work/flexsed_cache (shared,
           as DEV/TEST); flexx: the 147 extra queries (flexsed_extra.run, unchanged but for the stem list) ->
           data/work/flexsed_extra_<split>
  wav16 / beats / panns / dasm   as r13_test_prep / dev_candidates_check -> data/work/r13<split>/wav16, j2_<split>_beats,
           r13<split>/panns, dasm_<split>
  qwen     gold-free listener caches, one Qwen3-Omni load: yes/no superset (test_listener.superset: P1/P2/P3/PV) ->
           benchmark/gold/<split>_listener.json; amendment-A variants (listener_variants build + score) ->
           <split>_listener_v.json
  afn      Audio Flamingo Next V4 + yes/no (listener_afnext.run) -> <split>_listener_afn.json
  stage4 / stage5 / gates   B0r, B1 (self-veto 0.1218, PANNs off) and C1 = TO1+F7F8 through round13_dev (pipeline code),
           caches mapped to the split's files; gates D0 (stage 4 == render trace), D5 (B0r == render), completeness,
           listener-cache coverage. Stops before scoring.
  score    DEV2 ONLY: score_per_sound on the DEV2 clips of tagger_AG.json (the file is filtered to DEV2 stems on parse);
           refuses split test2.
score_per_sound.load_gold raises in every other step. New code only: round13_dev, dev_listener, test_listener,
listener_variants, listener_afnext, flexsed_run, flexsed_extra, dev_candidates_check are imported, never edited.

    python benchmark/gold/tagger_prep.py links                     # CPU: split clip folders (symlinks), stem checks
    python benchmark/gold/tagger_prep.py --split dev2 render       # GPU
    python benchmark/gold/tagger_prep.py --split dev2 stage4 --arms B0r
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S

_REAL_LOAD_GOLD = S.load_gold


def _no_gold(*a, **k):
    raise RuntimeError("tagger_prep: gold must not be read in this step")


S.load_gold = _no_gold

GOLDD = _ROOT / "benchmark" / "gold"
WORK = _ROOT / "data" / "work"
CLIPS = _ROOT / "data" / "input" / "tagger_set"
TAGGER_GOLD = GOLDD / "annotations" / "tagger_AG.json"
SPLITS = ("dev2", "test2")
C1 = "TO1+F7F8"
B1_FLAGS = {"BEATS_SELF_VETO": 0.1218, "PANNS_VETO": 0.0}
SYSTEMS = ("proposed", "blind_a2i")


def parity(stem):
    return int(hashlib.sha256(stem.encode()).hexdigest(), 16) % 2


def stems_of(split):
    st = sorted(x.strip() for x in (GOLDD / f"{split}_stems.txt").read_text(encoding="utf-8").split() if x.strip())
    # batch 1 by sha256 parity, batch 2 by the tag-balanced split; membership = benchmark/gold/tagger_split.json
    sp = json.loads((GOLDD / "tagger_split.json").read_text(encoding="utf-8"))["batches"]
    want = sorted(x for b in sp.values() for x in b[split])
    assert st and st == want, (split, st, want)
    have = {p.stem for p in CLIPS.glob("tg_d*.mp4")}
    assert set(st) <= have, sorted(set(st) - have)
    return st


def clip_dir(split):
    return _ROOT / "data" / "input" / f"tagger_{split}"


def tag(split):
    return f"{split}_v33"


def out(split):
    return WORK / f"r13{split}"


def lcache(split, kind=""):
    return GOLDD / f"{split}_listener{kind}.json"


# ============================================================================= redirects (the DEV harness -> this split)
_ORIG: dict = {}


def configure(split):
    from benchmark.gold import dev_candidates_check as DCC
    from benchmark.gold import round13_dev as R
    stems = stems_of(split)
    o = out(split)
    DCC.dev_stems = lambda: (None, list(stems))
    DCC.TAG = tag(split)
    DCC.BEATS_DIR = WORK / f"j2_{split}_beats"
    DCC.WAV16 = o / "wav16"
    DCC.DC = o / "dc"                                   # empty on purpose: no devcand stage-4 ref, no memo copied
    DCC.STAGE4, DCC.MEMO = DCC.DC / "stage4.json", DCC.DC / "ask_memo.json"
    DCC.DASM_DIR = WORK / f"dasm_{split}"
    R.R13 = o
    R.STAGE4, R.MEMO, R.PANNS_DIR = o / "stage4.json", o / "ask_memo.json", o / "panns"
    R.ARMS.setdefault("B1", dict(B1_FLAGS))
    assert R.ARMS["B1"] == B1_FLAGS
    # C1's DEV files -> this split's files (by file / folder name); anything still pointing at DEV/TEST is refused
    cmap = {"dev_listener.json": lcache(split), "dev_listener_v.json": lcache(split, "_v"),
            "dev_listener_afn.json": lcache(split, "_afn"), "dasm_cache": DCC.DASM_DIR,
            "flexsed_extra_dev": WORK / f"flexsed_extra_{split}"}
    for arm in ("B0r", "B1", C1):
        _ORIG.setdefault(arm, dict(R.ARMS[arm]))       # map from the DEV originals, whatever split came before
        if True:
            new = {}
            for k, v in _ORIG[arm].items():
                if isinstance(v, str) and Path(v).name in cmap:
                    v = str(cmap[Path(v).name])
                if isinstance(v, str) and any(b in v.replace("\\", "/") for b in
                                              ("dev_listener", "/devcand", "flexsed_extra_dev", "test_listener", "r13test")):
                    raise SystemExit(f"{arm}: flag {k} = {v} still points at a DEV/TEST file")
                new[k] = v
            R.ARMS[arm] = new
    return DCC, R, stems


# ============================================================================= CPU: clip folders
def links():
    for split in SPLITS:
        st = stems_of(split)
        d = clip_dir(split)
        d.mkdir(parents=True, exist_ok=True)
        for s in st:
            p = d / f"{s}.mp4"
            if not p.exists():
                p.symlink_to(CLIPS / f"{s}.mp4")
        extra = sorted(p.stem for p in d.glob("*.mp4") if p.stem not in set(st))
        assert not extra, extra
        print(f"[links] {split}: {len(st)} clips in {d}", flush=True)
    a, b = set(stems_of("dev2")), set(stems_of("test2"))
    assert not a & b and a | b == {p.stem for p in CLIPS.glob("tg_d*.mp4")}
    print(f"[links] DEV2 {sorted(a)}\n[links] TEST2 {sorted(b)}", flush=True)


# ============================================================================= GPU: FlexSED, render, caches
def flexsed(split):
    """benchmark/gold/flexsed_run.py, unchanged, on the split's clip folder (same code/settings as the cache)"""
    subprocess.run([sys.executable, str(GOLDD / "flexsed_run.py"), "--clip-dir", str(clip_dir(split)),
                    "--out", str(WORK / "flexsed_cache")], check=True)


def flexx(split):
    """flexsed_extra.run, unchanged except the stem list (its DEV/TEST count assert is the only line replaced)"""
    from benchmark.gold import flexsed_extra as FX
    src = inspect.getsource(FX.run)
    line = '    assert len(stems) == {"dev": 49, "test": 60}[which], len(stems)\n'
    assert src.count(line) == 1, "flexsed_extra.run changed"
    src = src.replace(line, f"    assert len(stems) == {len(stems_of(split))}, len(stems)\n")
    FX.STEMS[split] = GOLDD / f"{split}_stems.txt"
    ns = FX.__dict__
    exec(compile(src, FX.__file__, "exec"), ns)
    ns["run"](split, 24)


def render(split):
    import config
    changed = config.use_scored()
    config.DEVICE = "cuda"
    config.GEN_BACKEND = "placeholder"                     # the metric never looks at a picture
    config.TRANSCRIBE = True
    print("[render] use_scored:", {k: v[1] for k, v in changed.items()}, flush=True)
    for k in [p.stem for p in clip_dir(split).glob("*.mp4")]:
        assert (WORK / "flexsed_cache" / f"{k}.npz").exists(), f"no FlexSED cache for {k}: run flexsed first"
    sys.argv = ["run_protocol", "--phase", "render", "--clip-dir", str(clip_dir(split)), "--tag", tag(split),
                "--systems", *SYSTEMS]
    from benchmark.run_protocol import main
    main()


def wav16(split):
    DCC, R, stems = configure(split)
    DCC.WAV16.mkdir(parents=True, exist_ok=True)
    for st in stems:
        w = DCC.WAV16 / f"{st}.wav"
        if not w.exists():
            video = json.loads((DCC.scored_dir("proposed") / st / "media.json").read_text(encoding="utf-8"))["video_path"]
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", video, "-ac", "1", "-ar", "16000", str(w)], check=True)
    print(f"[wav16] {split} {sum((DCC.WAV16 / f'{s}.wav').exists() for s in stems)} / {len(stems)}", flush=True)


def beats(split):
    """exactly j2_dev_check.beats(): the shipped infer_beats on the render's audio.wav"""
    DCC, R, stems = configure(split)
    from src.stage4_audio_event_detection.beats_infer import infer_beats
    DCC.BEATS_DIR.mkdir(parents=True, exist_ok=True)
    for st in stems:
        dst = DCC.BEATS_DIR / f"{st}.npz"
        wav = DCC.scored_dir("proposed") / st / "audio.wav"
        if dst.exists() or not wav.exists():
            continue
        fw, t, labs = infer_beats(wav, "cuda")
        np.savez_compressed(dst, fw=fw.astype(np.float32), times=np.asarray(t, np.float64), labels=np.array(labs))
        print(f"[beats] {st} {fw.shape}", flush=True)


def panns(split):
    DCC, R, stems = configure(split)
    R.panns()


def dasm(split):
    """dev_candidates_check.dasm() (round 6's scorer and queries), redirected; run in its own process (it takes this
    project off sys.path)"""
    DCC, R, stems = configure(split)
    DCC.dasm()


# ============================================================================= listener caches (gold-free)
def lpool(split):
    DCC, R, stems = configure(split)
    from benchmark.gold import test_listener as TL
    from benchmark.gold import dev_listener as L
    from benchmark.listener_round import MODEL, QUESTION
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))["arms"]
    p1 = {}
    for st in stems:
        rows, keys = [], set()
        for sysn in SYSTEMS:
            for r in s4[f"B0r|{sysn}"][st]:
                k = (r["label"], round(r["start"], 3), round(r["end"], 3))
                if k not in keys:
                    keys.add(k); rows.append(r)
        p1[st] = rows
    vet = {st: TL._vetoed(DCC.scored_dir("proposed") / st / "onset_trace.json") for st in stems}
    items, miss = TL.superset(stems, p1, DCC.BEATS_DIR, DCC.FLEX_DIR, DCC.WAV16, vet)
    p = lcache(split)
    if p.exists() and any("score" in x for x in json.loads(p.read_text(encoding="utf-8"))["items"]):
        print(f"[lpool] {p} already scored; kept", flush=True)
        return
    meta = {"split": f"{split} (tagger set, {len(stems)} clips, benchmark/gold/{split}_stems.txt); gold-free",
            "model": MODEL, "question": QUESTION, "construction": "test_listener.superset (P1 B0r both systems, P2/P3 no "
            "coverage filter, PV from the render's trace), unchanged", "audio": f"{DCC.WAV16}/<clip>.wav",
            "pv_not_contained_in_P2": miss, "clips": len(stems)}
    DCC.dump(p, {"_meta": meta, "items": items})
    TL.OUT = p
    TL.sizes(items)
    print(f"[lpool] {split}: PV not inside a P2 run: {len(miss)}", flush=True)


def _memo_qwen():
    """one Qwen3-Omni load for every scorer in this process (dev_listener.score and listener_variants.score each call
    from_pretrained; the second call gets the same objects)"""
    import transformers
    for name in ("Qwen3OmniMoeForConditionalGeneration", "Qwen3OmniMoeProcessor"):
        cls = getattr(transformers, name)
        orig, memo = cls.from_pretrained, {}

        def fp(*a, _orig=orig, _memo=memo, **k):
            key = (a[0] if a else k.get("pretrained_model_name_or_path"))
            if key not in _memo:
                _memo[key] = _orig(*a, **k)
            return _memo[key]
        cls.from_pretrained = fp


def qwen(splits):
    """yes/no scores (dev_listener.score) per split, then the amendment-A variants (listener_variants build + score)"""
    _memo_qwen()
    from benchmark.gold import dev_listener as L
    from benchmark.gold import test_listener as TL
    from benchmark.gold import listener_variants as LV
    for split in splits:
        DCC, R, stems = configure(split)                 # DCC.WAV16 -> this split's wav16
        L.OUT = TL.OUT = lcache(split)
        L.report = TL.report
        L.score()
    LV.SPLITS = {}
    for split in splits:
        LV.SPLITS[split] = {"cache": lcache(split), "out": lcache(split, "_v"), "wav": out(split) / "wav16"}
        if not lcache(split, "_v").exists():
            LV.build(split)
    LV.score()


def afn(splits):
    from benchmark.gold import listener_afnext as AF
    from benchmark.gold import listener_variants as LV
    S.load_gold = _no_gold
    LV.SPLITS = {s: {"cache": lcache(s), "out": lcache(s, "_v"), "wav": out(s) / "wav16"} for s in splits}
    M = AF.load_model()
    for split in splits:
        outp = lcache(split, "_afn")
        if outp.exists():
            d = json.loads(outp.read_text(encoding="utf-8"))
            items, meta = d["items"], d["_meta"]
        else:
            items, meta = AF.base_items(split), AF.make_meta(M, split)
        AF.run(M, items, split, outp, meta)
        AF.summary(items, split)


# ============================================================================= stage 4 / 5 / gates (no gold)
def gates(split, arms):
    DCC, R, stems = configure(split)
    o = out(split)
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    g = {"split": split, "complete": {}, "d0": {}, "d5": {}, "listener": {}}
    for a in arms:
        for sysn in SYSTEMS:
            g["complete"][f"{a}|{sysn}"] = {"stage4": len(s4["arms"].get(f"{a}|{sysn}", {})),
                                            "stage5": sum((o / f"{a}_{sysn}" / st / "augmentations.json").exists() for st in stems),
                                            "of": len(stems)}
    d0 = s4["d0"]
    g["d0"] = {"pass": sum(v["pass"] for v in d0.values()), "of": len(d0), "fails": [k for k, v in d0.items() if not v["pass"]]}
    for sysn in SYSTEMS:
        if not DCC.complete(o / f"B0r_{sysn}", stems):
            g["d5"][sysn] = "incomplete"; continue
        bad = [st for st in stems
               if DCC.pics_sig(S.load_pictures(o / f"B0r_{sysn}", st, sysn) or [])
               != DCC.pics_sig(S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [])
               or DCC.spec_sig(o / f"B0r_{sysn}", st) != DCC.spec_sig(DCC.scored_dir(sysn), st)]
        g["d5"][sysn] = {"pass": len(stems) - len(bad), "of": len(stems), "differ": bad}
    for k, v in s4.get("listener", {}).items():
        a = k.rsplit("|", 1)[0]
        agg = g["listener"].setdefault(a, {})
        for f, x in v.items():
            if isinstance(x, (int, float)):
                agg[f] = agg.get(f, 0) + x
            elif isinstance(x, list):
                agg[f] = agg.get(f, 0) + len(x)
    live = {k: sum(1 for rr in v.values() for r in rr if r.get("refine") == "live") for k, v in s4["arms"].items()}
    g["live_refinements"] = live
    DCC.dump(o / "gates.json", g)
    print(f"[gates {split}] complete {g['complete']}", flush=True)
    print(f"[gates {split}] D0 {g['d0']['pass']}/{g['d0']['of']} fails {g['d0']['fails']}; D5 {g['d5']}", flush=True)
    print(f"[gates {split}] listener {g['listener']}; live {live}", flush=True)


# ============================================================================= DEV2 only: score
def score_dev2(arms):
    DCC, R, stems = configure("dev2")
    keep = set(stems)
    d = json.loads(TAGGER_GOLD.read_text(encoding="utf-8"))
    d["clips"] = [c for c in d.get("clips", []) if isinstance(c, dict) and Path(str(c.get("clip", ""))).stem in keep]
    tmp = out("dev2") / "dev2_gold_only.json"
    DCC.dump(tmp, d)
    del d
    gold = _REAL_LOAD_GOLD([tmp])
    assert set(gold) <= keep, sorted(set(gold) - keep)
    missing = sorted(keep - set(gold))
    o = out("dev2")
    s4 = json.loads(R.STAGE4.read_text(encoding="utf-8"))
    names = ["B0"] + list(arms)
    res = {"split": "dev2", "clips": stems, "gold_missing": missing, "arms": {a: R.ARMS[a] for a in arms}, "rows": {},
           "heard": {}, "delta_vs_B0r": {}, "delta_vs_B1": {}, "needed_changes": {}}
    for sysn in SYSTEMS:
        P = {"B0": {st: S.load_pictures(DCC.scored_dir(sysn), st, sysn) or [] for st in stems}}
        for a in arms:
            with R.flags({k: R.arm_cfg(a)[k] for k in R.DISPLAY_KEYS}):
                P[a] = {st: S.load_pictures(o / f"{a}_{sysn}", st, sysn) or [] for st in stems}
        sc = [st for st in stems if st in gold]
        rows = {n: [S.score_clip(gold[st], P[n][st]) for st in sc] for n in names}
        res["rows"][sysn] = {n: DCC.metrics(r) for n, r in rows.items()}
        res["heard"][sysn] = {a: sum(S.score_clip(gold[st], DCC.heard_pics(s4["arms"][f"{a}|{sysn}"][st], 0.35))["hit"]
                                     for st in sc) for a in arms}
        cost = {n: [DCC.clip_cost(r) for r in rr] for n, rr in rows.items()}
        res["delta_vs_B0r"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B0r"])) for n in names if n != "B0r"}
        if "B1" in names:
            res["delta_vs_B1"][sysn] = {n: DCC.boot(np.subtract(cost[n], cost["B1"])) for n in names if n != "B1"}
        ch = {}
        for a in arms:
            if a == "B0r":
                continue
            gained, lost = [], []
            for st in sc:
                for (gg, a0), (_g2, a1) in zip(DCC.needed_hit(gold[st], P["B0r"][st]), DCC.needed_hit(gold[st], P[a][st])):
                    if a1 and not a0:
                        gained.append([st, gg["label"], gg["start"]])
                    if a0 and not a1:
                        lost.append([st, gg["label"], gg["start"]])
            ch[a] = {"gained": gained, "lost": lost}
        res["needed_changes"][sysn] = ch
        for n in names:
            x = res["rows"][sysn][n]; dd = res["delta_vs_B0r"][sysn].get(n)
            print(f"DEV2 {sysn:9s} {n:9s} heard {res['heard'][sysn].get(n)} hits {x['hits']}/{x['hits'] + x['misses']} "
                  f"wrong {x['wrong']} ({x['visible']}/{x['cross']}/{x['phantom']}) cost {x['viewer_cost']:.3f}"
                  + (f"  d vs B0r {dd[0]:+.3f} [{dd[1]:+.3f}, {dd[2]:+.3f}] p {dd[3]:.3f}" if dd else ""), flush=True)
    DCC.dump(GOLDD / "dev2_score.json", res)
    tmp.unlink()
    print(f"-> {GOLDD / 'dev2_score.json'}", flush=True)


def check():
    for split in SPLITS:
        DCC, R, stems = configure(split)
        ex = {"flexsed": sum((DCC.FLEX_DIR / f"{s}.npz").exists() for s in stems),
              "flexsed_extra": sum((WORK / f"flexsed_extra_{split}" / f"{s}.npz").exists() for s in stems),
              "beats": sum((DCC.BEATS_DIR / f"{s}.npz").exists() for s in stems),
              "wav16": sum((DCC.WAV16 / f"{s}.wav").exists() for s in stems),
              "panns": sum((R.PANNS_DIR / f"{s}.npz").exists() for s in stems),
              "dasm": sum((DCC.DASM_DIR / f"{s}.npz").exists() for s in stems)}
        for sysn in SYSTEMS:
            for fn in ("audio.wav", "onset_trace.json", "media.json", "scene.json", "segments.json", "augmentations.json",
                       "gate_votes.json"):
                ex[f"{sysn}/{fn}"] = sum((DCC.scored_dir(sysn) / s / fn).exists() for s in stems)
        for kind in ("", "_v", "_afn"):
            p = lcache(split, kind)
            if p.exists():
                its = json.loads(p.read_text(encoding="utf-8"))["items"]
                done = sum(("score" in x) or ("accept" in x) for x in its)
                ex[p.name] = f"{done}/{len(its)} scored"
            else:
                ex[p.name] = "missing"
        print(f"[check {split}] {len(stems)} clips: {json.dumps(ex)}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=("links", "check", "flexsed", "flexx", "render", "wav16", "beats", "panns", "dasm",
                                     "lpool", "qwen", "afn", "stage4", "stage5", "gates", "score"))
    ap.add_argument("--split", choices=SPLITS)
    ap.add_argument("--arms", nargs="+", default=["B0r"])
    a = ap.parse_args()
    if a.step in ("links", "check"):
        return {"links": links, "check": check}[a.step]()
    if a.step == "qwen":
        return qwen([a.split] if a.split else list(SPLITS))
    if a.step == "afn":
        return afn([a.split] if a.split else list(SPLITS))
    if not a.split:
        raise SystemExit("--split dev2|test2")
    if a.step == "score":
        if a.split != "dev2":
            raise SystemExit("TEST2 is sealed: never scored here")
        return score_dev2(a.arms)
    if a.step in ("stage4", "stage5"):
        DCC, R, stems = configure(a.split)
        for arm in a.arms:
            if arm not in R.ARMS:
                raise SystemExit(f"unknown arm {arm}")
            cfg = R.arm_cfg(arm)
            need = [p for p in (cfg.get("LISTENER_CACHE"), cfg.get("LISTENER_VCACHE"), cfg.get("LISTENER_AFCACHE"))
                    if cfg.get("LISTENER_RESCUE") or cfg.get("LISTENER_CONFIRMED_MIRROR") if p]
            miss = [p for p in need if not Path(p).exists()]
            if cfg.get("LISTENER_DASM_VOTE"):            # a missing DASM file silently turns the F8 vote off: refuse
                miss += [str(Path(cfg["LISTENER_DASM_DIR"]) / f"{s}.npz") for s in stems
                         if not (Path(cfg["LISTENER_DASM_DIR"]) / f"{s}.npz").exists()]
            if miss:
                raise SystemExit(f"{arm}: inputs missing, not run: {miss[:5]} ({len(miss)})")
        return (R.stage4 if a.step == "stage4" else R.stage5)(a.arms)
    if a.step == "gates":
        return gates(a.split, a.arms)
    {"flexsed": flexsed, "flexx": flexx, "render": render, "wav16": wav16, "beats": beats, "panns": panns, "dasm": dasm,
     "lpool": lpool}[a.step](a.split)


if __name__ == "__main__":
    main()
