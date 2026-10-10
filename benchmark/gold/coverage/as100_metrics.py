"""AS100 step 3: metrics on the 100 clips (env sota, PYTHONPATH=~/as100_eval/pylib for sed_scores_eval + sed_eval).
    python as100_metrics.py  -> ~/as100_eval/metrics.json + printed tables
(a) PSDS1 of each PretrainedSED model, their own protocol (ex_audioset_strong.py: median filter 9 frames, classes in
    gold of these clips AND in the 447 training classes, clips without such gold dropped, dtc=gtc=0.7, alpha_st=1).
(b) FAIR class set C = 447 strong classes that our stage 4 can name (BEATs 527 + FlexSED vocab, matched by MID)
    minus the Music and Speech ontology subtrees and the repo's never-drawn rule (speech labels, is_music).
    Every system restricted to C. sed_eval segment-based (1 s) and event-based (onset 0.2 s, offset 0.2 s or 20 %).
    SOTA events: median filter 9 frames (as PretrainedSED decoding), threshold, contiguous frames -> event.
(c) onset rule of our benchmark: one-to-one, same class (exact MID), detection onset in [gold onset -0.5, +1.0] s.
"""
import json, sys
from pathlib import Path
import numpy as np
import scipy.ndimage
import soundfile as sf

H = Path.home(); OUT = H / "as100_eval"
sys.path.insert(0, str(H / "PretrainedSED"))
MODELS = ["ATST-F", "BEATs", "fpasst", "M2D", "ASIT"]
GRID = [round(x, 2) for x in np.arange(0.05, 0.951, 0.05)]

clips = [l.strip() for l in open(OUT / "clips.txt") if l.strip()]
gold_n = json.load(open(OUT / "gold.json"))
ours_n = json.load(open(OUT / "ours.json"))
wav = json.load(open(OUT / "wav_paths.json"))
dur = {st: sf.info(wav[st]).duration for st in clips}

# ---- names <-> MIDs -------------------------------------------------------------------------------------------
tsv = dict(l.rstrip("\n").split("\t")[:2] for l in open(H / "open_data/as_strong/mid_to_display_name.tsv", encoding="utf-8"))
name2mid_tsv = {v: k for k, v in tsv.items()}
old = json.load(open(H / "MscProj_tg/src/audioset_mid_names.json"))          # names our BEATs/FlexSED use
name2mid_old = {v: k for k, v in old.items()}
onto = {d["id"]: d for d in json.load(open(H / "MscProj_tg/src/audioset_ontology.json"))}
for d in onto.values():
    name2mid_old.setdefault(d["name"], d["id"])


ALIAS = {"Footsteps": "Walk, footsteps"}          # our family name for the walk/footsteps class


def mid_of(name):
    name = ALIAS.get(name, name)
    return name2mid_tsv.get(name) or name2mid_old.get(name)


def subtree(mid):
    out, st = set(), [mid]
    while st:
        m = st.pop()
        if m not in out:
            out.add(m); st += onto.get(m, {}).get("child_ids", [])
    return out


from data_util import audioset_classes as AC
SOTA_NAMES = list(AC.as_strong_train_classes)
SOTA_MIDS = [name2mid_tsv[n] for n in SOTA_NAMES]                       # fails if a class has no MID
voc = json.load(open(OUT / "our_vocab.json"))
OURS_VOCAB = {mid_of(n) for n in voc["beats"] + voc["flexsed"]} - {None}
sys.path.insert(0, str(H / "MscProj_tg"))
from src.labels import SPEECH_LABELS, is_music                            # read-only import (never-drawn rule)
SPEECH_MUSIC = subtree("/m/04rlf") | subtree("/m/09x0r")                 # Music, Speech subtrees
for m in SOTA_MIDS:
    n = tsv[m]
    if n in SPEECH_LABELS or is_music(n) or (old.get(m) and is_music(old[m])):
        SPEECH_MUSIC.add(m)
C = sorted(m for m in set(SOTA_MIDS) & OURS_VOCAB if m not in SPEECH_MUSIC)
CS = set(C)

gold = {st: [(s, e, name2mid_tsv[l]) for s, e, l in gold_n[st]] for st in clips}
ours = {st: [(s, e, mid_of(l)) for s, e, l, c in ours_n[st]] for st in clips}
unmapped = sorted({l for st in clips for s, e, l, c in ours_n[st] if mid_of(l) is None})

# ---- SOTA frame probabilities --------------------------------------------------------------------------------
probs = {}
for m in MODELS:
    p = OUT / f"sota_{m}.npz"
    if p.exists():
        z = np.load(p)
        assert [str(x) for x in z["classes"]] == SOTA_NAMES
        probs[m] = {st: scipy.ndimage.median_filter(z[st], (9, 1)) for st in clips}   # postprocessed, as theirs
FR = 0.04


def decode(pp, th, keep):
    """events (onset, offset, mid) from median-filtered probs, classes in keep."""
    ev = {}
    idx = [i for i, m in enumerate(SOTA_MIDS) if m in keep]
    for st in clips:
        x = pp[st][:, idx] >= th
        out = []
        for j, ci in enumerate(idx):
            col = np.concatenate([[0], x[:, j].astype(int), [0]])
            d = np.diff(col)
            for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
                out.append((a * FR, min(b * FR, dur[st]), SOTA_MIDS[ci]))
        ev[st] = out
    return ev


# ---- (a) PSDS1 -----------------------------------------------------------------------------------------------
def psds1(m, keep=None):
    from sed_scores_eval import intersection_based
    from sed_scores_eval.base_modules.scores import create_score_dataframe
    gt_classes = {e[2] for st in clips for e in gold_n[st]}
    cls = [n for n in SOTA_NAMES if n in gt_classes and (keep is None or name2mid_tsv[n] in keep)]
    gt = {st: [(s, e, l) for s, e, l in gold_n[st] if l in cls] for st in clips}
    gt = {st: v for st, v in gt.items() if v}
    ci = [SOTA_NAMES.index(n) for n in cls]
    sc = {st: create_score_dataframe(probs[m][st][:, ci], np.arange(len(probs[m][st]) + 1) * FR, cls) for st in gt}
    du = {st: max(dur[st], len(probs[m][st]) * FR) for st in gt}
    r = intersection_based.psds(sc, gt, du, dtc_threshold=0.7, gtc_threshold=0.7, cttc_threshold=None,
                                alpha_ct=0, alpha_st=1, num_jobs=1)
    return {"psds1": float(r[0]), "psds1_macro": float(np.mean(list(r[1].values()))), "classes": len(cls), "clips": len(gt)}


# ---- (b) sed_eval --------------------------------------------------------------------------------------------
import sed_eval, dcase_util


def mdc(evs, st):
    return dcase_util.containers.MetaDataContainer(
        [{"filename": st, "event_label": l, "onset": float(s), "offset": float(e)} for s, e, l in evs])


def sed_scores(sysev):
    seg = sed_eval.sound_event.SegmentBasedMetrics(event_label_list=C, time_resolution=1.0)
    evm = sed_eval.sound_event.EventBasedMetrics(event_label_list=C, t_collar=0.2, percentage_of_length=0.2,
                                                 evaluate_onset=True, evaluate_offset=True)
    evo = sed_eval.sound_event.EventBasedMetrics(event_label_list=C, t_collar=0.2, evaluate_onset=True,
                                                 evaluate_offset=False)       # extra: onset-only
    for st in clips:
        ref = mdc([g for g in gold[st] if g[2] in CS], st)
        est = mdc([e for e in sysev[st] if e[2] in CS], st)
        seg.evaluate(reference_event_list=ref, estimated_event_list=est, evaluated_length_seconds=dur[st])
        evm.evaluate(reference_event_list=ref, estimated_event_list=est)
        evo.evaluate(reference_event_list=ref, estimated_event_list=est)
    res = {}
    for name, met in (("seg", seg), ("evt", evm), ("evt_on", evo)):
        cw = met.class_wise
        tp = sum(cw[c]["Ntp"] for c in C); fp = sum(cw[c]["Nfp"] for c in C); fn = sum(cw[c]["Nfn"] for c in C)
        f1c = [2 * cw[c]["Ntp"] / (2 * cw[c]["Ntp"] + cw[c]["Nfp"] + cw[c]["Nfn"]) for c in C if cw[c]["Nref"] > 0]
        res[name] = {"P": tp / (tp + fp) if tp + fp else 0.0, "R": tp / (tp + fn) if tp + fn else 0.0,
                     "F1_micro": 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0,
                     "F1_macro": float(np.mean(f1c)), "classes_in_gold": len(f1c),
                     "tp": int(tp), "fp": int(fp), "fn": int(fn)}
    return res


# ---- (c) onset rule ------------------------------------------------------------------------------------------
def onset_rule(sysev):
    hits = gold_total = wrong = 0
    for st in clips:
        g = sorted([x for x in gold[st] if x[2] in CS])
        d = [x for x in sysev[st] if x[2] in CS]
        used = set()
        gold_total += len(g)
        for gs, ge, gl in g:
            best = None
            for k, (ds, de, dl) in enumerate(d):
                if k in used or dl != gl or not (gs - 0.5 <= ds <= gs + 1.0):
                    continue
                if best is None or abs(ds - gs) < abs(d[best][0] - gs):
                    best = k
            if best is not None:
                used.add(best); hits += 1
        wrong += len(d) - len(used)
    return {"hits": hits, "gold": gold_total, "recall": hits / gold_total, "wrong": wrong,
            "detections": hits + wrong, "precision": hits / (hits + wrong) if hits + wrong else 0.0}


def main():
    res = {"clips": len(clips), "C_size": len(C), "C": [tsv[m] for m in C], "ours_unmapped_labels": unmapped,
           "gold_events_C": sum(1 for st in clips for g in gold[st] if g[2] in CS),
           "gold_events_all": sum(len(v) for v in gold.values()),
           "gold_classes_C_present": len({g[2] for st in clips for g in gold[st] if g[2] in CS}),
           "ours_events_C": sum(1 for st in clips for e in ours[st] if e[2] in CS),
           "ours_events_all": sum(len(v) for v in ours.values()), "rows": {}, "psds": {}}
    res["rows"]["ours (stage 4, D')"] = {**sed_scores(ours), "onset": onset_rule(ours)}
    for m in probs:
        res["psds"][m] = {"all_classes": psds1(m), "C": psds1(m, CS)}
        ev = decode(probs[m], 0.5, CS)
        res["rows"][f"{m} @0.5"] = {**sed_scores(ev), "onset": onset_rule(ev)}
        sweep = {}
        for th in GRID:
            ev = decode(probs[m], th, CS)
            sweep[th] = {**sed_scores(ev), "onset": onset_rule(ev)}
        bs = max(GRID, key=lambda t: sweep[t]["seg"]["F1_micro"])
        be = max(GRID, key=lambda t: sweep[t]["evt"]["F1_micro"])
        of1 = lambda o: 2 * o["hits"] / (o["gold"] + o["detections"])
        bo = max(GRID, key=lambda t: of1(sweep[t]["onset"]))
        bn = max(GRID, key=lambda t: sweep[t]["evt_on"]["F1_micro"])
        res["rows"][f"{m} best-th (optimistic)"] = {"seg": {**sweep[bs]["seg"], "th": bs}, "evt": {**sweep[be]["evt"], "th": be},
                                                    "evt_on": {**sweep[bn]["evt_on"], "th": bn},
                                                    "onset": {**sweep[bo]["onset"], "th": bo}}
        res.setdefault("sweep", {})[m] = {str(t): {"segF1": sweep[t]["seg"]["F1_micro"], "evtF1": sweep[t]["evt"]["F1_micro"],
                                                   "onset_hits": sweep[t]["onset"]["hits"], "onset_wrong": sweep[t]["onset"]["wrong"]} for t in GRID}
        print(m, "done", flush=True)
    json.dump(res, open(OUT / "metrics.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("clips", "C_size", "gold_events_C", "gold_events_all", "gold_classes_C_present",
                                          "ours_events_C", "ours_events_all", "ours_unmapped_labels")}))
    for m, v in res["psds"].items():
        print("PSDS1", m, v)
    print(f"{'system':28s} | seg P R F1mi F1ma th | evt P R F1mi F1ma th | onset hits/gold wrong P th | evt-onset-only P R F1mi F1ma th")
    for k, r in res["rows"].items():
        s, e, o, n = r["seg"], r["evt"], r["onset"], r["evt_on"]
        print(f"{k:28s} | {s['P']:.2f} {s['R']:.2f} {s['F1_micro']:.3f} {s['F1_macro']:.3f} {s.get('th', '')} | "
              f"{e['P']:.2f} {e['R']:.2f} {e['F1_micro']:.3f} {e['F1_macro']:.3f} {e.get('th', '')} | "
              f"{o['hits']}/{o['gold']} {o['wrong']} {o['precision']:.2f} {o.get('th', '')} | "
              f"{n['P']:.2f} {n['R']:.2f} {n['F1_micro']:.3f} {n['F1_macro']:.3f} {n.get('th', '')}")


if __name__ == "__main__":
    main()
