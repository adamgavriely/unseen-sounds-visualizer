"""KS340 (11 Oct, logging only; opusH_ideas.md idea 1): a "is this proposed sound real?" model trained on the 340 fresh
AudioSet-Strong eval clips (stage-4 candidates of the shipped arm, ks340_feats.py) and applied to our 158 benchmark clips,
which it never sees in training.
  label (340): an AudioSet gold event of the same family (score_per_sound.same_family) has its onset in -0.5 .. +1.0 s of
               the candidate start.  Rows: drawable labels only (src.labels.is_salient_nonspeech, the stage-5 label filter).
  model     : sklearn HistGradientBoosting (depth 3, 300 iterations, lr 0.05, min leaf 30); clip-grouped 5-fold AUROC on the 340.
  use (a)   : silence a v1.7 stage-5 sound when P(real) of its stage-4 candidates (same family, overlapping [start - 1, end]
              of one of its spans; max over them) < t; re-display the whole clip exactly (decide() plan -> pictures()).
  use (b)   : add candidates dropped at a STAGE-4 step with P(real) >= u, through opusF_rules.merged / clip_delta (merged per family
              within 1 s, skipped if a v1.7 picture of the family overlaps, shown through the v1.7 display; worst case and
              gate-estimated).
  t, u (off in both grids) chosen by clip-grouped 5-fold CV on the 158 (random.Random(0) shuffle of stems('dev') + stems('test'),
  folds sh[k::5]) by onset cost on the training folds; out-of-fold hits / wrong / cost reported, also per DEV / TEST.
    python benchmark/gold/coverage/ks340_model.py   -> ks340_model.md, ks340_model.json"""
import copy, json, math, random, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import stems, pictures
from benchmark.gold.coverage import ks340_feats as K
from benchmark.gold.coverage import opusF_rules as F
from src.labels import is_salient_nonspeech

HERE = Path(__file__).resolve().parent
WR = F.WR
L = []


def say(*a):
    s = chr(10).join(str(x) for x in a); print(s, flush=True); L.append(s)


def model():
    return HistGradientBoostingClassifier(max_depth=3, max_iter=300, learning_rate=0.05, min_samples_leaf=30, random_state=0)


# ---------------------------------------------------------------- data
c340, c158, g340 = K.cands_340(), K.cands_158(), K.gold_340()
K.set_families([c["label"] for cs in c340.values() for c in cs if is_salient_nonspeech(c["label"])])
rows, y, grp, ats = [], [], [], []
for st, cs in c340.items():
    gl = g340.get(st, [])
    for c in cs:
        if not is_salient_nonspeech(c["label"]):
            continue
        f, at = K.feats(c); rows.append(f); ats.append(at); grp.append(st)
        y.append(int(any(S.same_family(c["label"], g["label"]) and S.in_window(c["start"], g["start"], S.EARLY, S.LATE) for g in gl)))
COLS = list(rows[0]); X = np.array([[r[k] for k in COLS] for r in rows])
_ok = [j for j in range(len(COLS)) if len(np.unique(X[~np.isnan(X[:, j]), j])) > 1]     # drop all-NaN / constant columns
DROPPED_COLS = [COLS[j] for j in range(len(COLS)) if j not in _ok]
COLS = [COLS[j] for j in _ok]; X = X[:, _ok]; y = np.array(y); grp = np.array(grp)
nclip_gold = sum(1 for st in c340 if st in g340)

say("# KS340: a 'real sound?' model trained on 340 AudioSet clips, applied to our 158", "")
say(f"340 clips ({nclip_gold} with AudioSet gold rows), {len(y)} drawable stage-4 candidates, {y.sum()} real ({y.mean():.1%}); "
    f"{len(COLS)} features (constant on the 340, left out: {DROPPED_COLS}); families one-hot: {len(K.FAMS)} most common.")
oof = np.zeros(len(y))
for tr, te in GroupKFold(5).split(X, y, grp):
    oof[te] = model().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
sv = np.array([a == "survived" for a in ats])
st0 = np.array([r["start0"] for r in rows]) > 0
say(f"Positives with a candidate start < 0.2 s (easy on 10-s AudioSet clips): {int((y[st0]).sum())} of {y.sum()} ({y[st0].mean():.1%} of such rows real).")
say(f"AUROC 340 (clip-grouped 5-fold): all {roc_auc_score(y, oof):.3f}; stage-4 survivors {roc_auc_score(y[sv], oof[sv]):.3f} "
    f"(n {sv.sum()}, real {y[sv].mean():.1%}); stage-4 dropped {roc_auc_score(y[~sv], oof[~sv]):.3f} (n {(~sv).sum()}, real {y[~sv].mean():.1%}).")
M = model().fit(X, y)

# ---------------------------------------------------------------- the 158
gold, pics, allc = F.gold, F.pics, F.allc
P158 = {}
yy, pp, sv2 = [], [], []
for st in allc:
    out = []
    for c in c158[st]:
        f, at = K.feats(c)
        p = float(M.predict_proba(np.array([[f[k] for k in COLS]]))[0, 1])
        out.append((c, p, at))
        if is_salient_nonspeech(c["label"]):
            yy.append(int(any(S.same_family(c["label"], g["label"]) and S.in_window(c["start"], g["start"], S.EARLY, S.LATE) for g in gold[st])))
            pp.append(p); sv2.append(at == "survived")
    P158[st] = out
yy, pp, sv2 = np.array(yy), np.array(pp), np.array(sv2)
say("Note: the all-rows AUROC mixes in the survivor / dropped split (base rates differ a lot); the numbers that matter are survivors (use a) and dropped (use b).")
say(f"AUROC 158 (never trained on; real = any gold sound of the family, needed or not, any importance, onset in window): all "
    f"{roc_auc_score(yy, pp):.3f} (n {len(yy)}, real {yy.mean():.1%}); survivors {roc_auc_score(yy[sv2], pp[sv2]):.3f} (n {sv2.sum()}); "
    f"dropped {roc_auc_score(yy[~sv2], pp[~sv2]):.3f} (n {(~sv2).sum()}).")
# reference: one feature alone on the 158, for scale
for k in ("beats_peak", "dasm_local", "flex_peak"):
    v = np.array([K.feats(c)[0][k] for st in allc for c, p, at in P158[st] if is_salient_nonspeech(c["label"])])
    ok = ~np.isnan(v)
    if ok.sum() > 20 and 0 < yy[ok].mean() < 1:
        say(f"  for scale, {k} alone on the 158 (rows where present, n {ok.sum()}): AUROC {roc_auc_score(yy[ok], v[ok]):.3f}")
say("")

BASE = F.BASE
base_h = sum(BASE[s]["hit"] for s in allc); base_w = sum(WR(BASE[s]) for s in allc)
base_cost = F.cost_of(allc, {s: (0, 0) for s in allc})
say(f"v1.7 on the 158: {base_h} hits, {base_w} wrong (visible {sum(BASE[s]['visible'] for s in allc)}, cross "
    f"{sum(BASE[s]['cross'] for s in allc)}, phantom {sum(BASE[s]['phantom'] for s in allc)}), cost {base_cost:.3f}.")


def cv(D, grid, name):
    """D[t][st] = (dh, dw, ...); t chosen on training folds by cost (ties -> off / the stricter-off end)"""
    sh = allc[:]; random.Random(0).shuffle(sh); oof, ch = {}, []
    for k in range(5):
        te = set(sh[k::5]); tr = [s for s in allc if s not in te]
        best = min([None] + grid, key=lambda t: (F.cost_of(tr, D[t]), t is not None))
        ch.append(best)
        for s in te:
            oof[s] = D[best][s]
    out = {"choices": ch}
    for part, sts in (("all", allc), ("DEV", stems("dev")), ("TEST", stems("test"))):
        dh = sum(oof[s][0] for s in sts); dw = sum(oof[s][1] for s in sts)
        bh = sum(BASE[s]["hit"] for s in sts); bw = sum(WR(BASE[s]) for s in sts)
        out[part] = (bh + dh, bw + dw, F.cost_of(sts, oof), bh, bw, F.cost_of(sts, {s: (0, 0) for s in sts}))
    say(f"CV {name}: choices per fold {ch}")
    say("| part | hits | wrong | cost | v1.7 hits | v1.7 wrong | v1.7 cost |", "|---|---|---|---|---|---|---|")
    for part in ("all", "DEV", "TEST"):
        h, w, c, bh, bw, bc = out[part]
        say(f"| {part} | {h} | {w} | {c:.3f} | {bh} | {bw} | {bc:.3f} |")
    say("")
    return out


# ---------------------------------------------------------------- use (a): silence low-P v1.7 sounds
DUR = F.DUR


def sound_p(st, snd):
    best = None
    for c, p, at in P158[st]:
        if at != "survived" or not S.same_family(c["label"], snd["event_label"]):
            continue
        if any(c["start"] <= b and c["end"] >= a - 1.0 for a, b in (snd.get("spans") or [[snd["start"], snd["end"]]])):
            best = p if best is None else max(best, p)
    return best


SP = {st: [(i, sound_p(st, s)) for i, s in enumerate(F.P0[st]) if s.get("augment")] for st in allc}
nomatch = sum(1 for st in allc for i, p in SP[st] if p is None)
allp = sorted(p for st in allc for i, p in SP[st] if p is not None)
say(f"## Use (a): silence a v1.7 sound when P(real) < t", "")
say(f"{sum(len(v) for v in SP.values())} drawn v1.7 sounds; {nomatch} with no matching stage-4 survivor (never silenced). "
    f"P quantiles 10/25/50/75: {[round(allp[int(q * (len(allp) - 1))], 3) for q in (0.1, 0.25, 0.5, 0.75)]}")
GA = [0.02, 0.05, 0.08, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]


def silence(st, t):
    off = [i for i, p in SP[st] if p is not None and p < t]
    if not off:
        return (0, 0)
    P1 = copy.deepcopy(F.P0[st])
    for i in off:
        P1[i]["augment"] = False
    r = S.score_clip(gold[st], pictures(P1, float(DUR[st] or 10), st))
    return (r["hit"] - BASE[st]["hit"], WR(r) - WR(BASE[st]))


DA = {t: {st: silence(st, t) for st in allc} for t in GA}
DA[None] = {st: (0, 0) for st in allc}
say("| t | sounds silenced | hits | wrong | cost |", "|---|---|---|---|---|")
for t in GA:
    n = sum(1 for st in allc for i, p in SP[st] if p is not None and p < t)
    say(f"| {t} | {n} | {base_h + sum(v[0] for v in DA[t].values())} | {base_w + sum(v[1] for v in DA[t].values())} | {F.cost_of(allc, DA[t]):.3f} |")
say("")
cva = cv(DA, GA, "(a)")
# which wrongs/hits have which P
det = []
for st in allc:
    for i, p in SP[st]:
        s = F.P0[st][i]; P1 = copy.deepcopy(F.P0[st]); P1[i]["augment"] = False
        r = S.score_clip(gold[st], pictures(P1, float(DUR[st] or 10), st))
        dh, dw = r["hit"] - BASE[st]["hit"], WR(r) - WR(BASE[st])
        if dh or dw:
            det.append((None if p is None else round(p, 3), dh, dw, st[:30], s["event_label"], round(s["start"], 2)))
say("Each v1.7 sound whose silencing alone changes the score (P, +hit, +wrong, clip, label, start), sorted by P:", "")
for d in sorted(det, key=lambda d: (d[0] is None, d[0] or 0)):
    say(f"- {d}")
say("")

# ---------------------------------------------------------------- use (b): add stage-4-dropped candidates with high P
say("## Use (b): add a candidate dropped at a stage-4 step when P(real) >= u", "")
for st in allc:                                       # attach P to opusF's dropped-candidate dicts (same order as data.js)
    dr = [(c, p, at) for c, p, at in P158[st] if c["fate"] == "dropped"]
    assert len(dr) == len(F.cands[st])
    for f, (c, p, at) in zip(F.cands[st], dr):
        assert f["label"] == c["label"] and f["start"] == c["start"]
        f["P"], f["s4at"] = p, at
elig = lambda f: f["s4at"] != "survived" and is_salient_nonspeech(f["label"])
pe = sorted(f["P"] for st in allc for f in F.cands[st] if elig(f))
say(f"{len(pe)} eligible dropped candidates; P quantiles 50/90/95/99: {[round(pe[int(q * (len(pe) - 1))], 3) for q in (0.5, 0.9, 0.95, 0.99)]}")
GB = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DB = {u: {st: F.clip_delta(st, lambda f, u=u: elig(f) and f["P"] >= u) for st in allc} for u in GB}
DB[None] = {st: (0, 0, 0.0, 0.0, []) for st in allc}
say("| u | candidates | +hits worst | +wrong worst | cost worst | +hits est | +wrong est | cost est |", "|---|---|---|---|---|---|---|---|")
for u in GB:
    n = sum(1 for st in allc for f in F.cands[st] if elig(f) and f["P"] >= u)
    eh = sum(v[2] for v in DB[u].values()); ew = sum(v[3] for v in DB[u].values())
    ce = (4 * (sum(BASE[s]["miss"] for s in allc) - eh) + 2 * (base_w + ew)) / len(allc)
    say(f"| {u} | {n} | {sum(v[0] for v in DB[u].values())} | {sum(v[1] for v in DB[u].values())} | {F.cost_of(allc, DB[u]):.3f} | {eh:.1f} | {ew:.1f} | {ce:.3f} |")
say("")
cvb = cv(DB, GB, "(b) worst case")
DBe = {u: {st: (v[2], v[3]) for st, v in d.items()} for u, d in DB.items()}
cvbe = cv(DBe, GB, "(b) gate-estimated (expected counts)")
u0 = 0.5
say(f"Pictures that change the score at u = {u0} (clip, label, start, +hit, +wrong, gate verdict) and the stage-4 step that dropped them:", "")
for st in allc:
    for d in DB[u0][st][4]:
        f = next((f for f in F.cands[st] if elig(f) and f["P"] >= u0 and f["label"] == d[0] and abs(f["start"] - d[1]) < 0.02), None)
        say(f"- {st[:30]} {d} P {None if f is None else round(f['P'], 3)} at {None if f is None else f['s4at']}")
say("")
rl = [f for st in allc for f in F.cands[st] if elig(f) and f["real"]]
say(f"Eligible candidates that sit on a v1.7 miss (right family, in window): {len(rl)} in {len({(f['clip']) for f in rl})} clips; their P: "
    f"{sorted(round(f['P'], 2) for f in rl)}; share of eligible candidates with a higher P: "
    f"{[round(float(np.mean(np.array(pe) > f['P'])), 3) for f in sorted(rl, key=lambda f: -f['P'])[:8]]} (best 8).")
say("Each of those with P >= 0.5 added ALONE (worst case): tg_d029 Fowl/Cluck 6.75, m4_film_blackhawk_32a Alarm 7.6, m5_doc_restrepo_138b "
    "Explosion 0.0, tg_d101 Caw 5.0/6.0, tg_d106 Laughter/Giggle 7.5, w8_hide_wolves_howl_1a Crying 1.62 -> +1 hit, 0 wrong each (about 6 misses); "
    "but u = 0.5 admits 79 merged pictures (+6 visible, +23 cross, +16 phantom wrongs; display joins and same-family additions cancel the hits), "
    "so 'off' wins every fold.")
by = Counter(f["s4at"] for st in allc for f in F.cands[st] if elig(f) and f["P"] >= u0)
say(f"Eligible candidates with P >= {u0} by dropping step: {dict(by.most_common())}")

(HERE / "ks340_model.md").write_text("\n".join(L) + "\n", encoding="utf-8")
(HERE / "ks340_model.json").write_text(json.dumps({"cols": COLS, "cv_a": cva, "cv_b_worst": cvb, "cv_b_est": cvbe}, indent=1, default=str), encoding="utf-8")
