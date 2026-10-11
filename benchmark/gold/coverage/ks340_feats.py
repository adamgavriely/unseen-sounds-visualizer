"""KS340 (11 Oct, logging only): one feature row per stage-4 candidate, read from its decision trail, with the SAME code
for the 340 fresh AudioSet-Strong eval clips (raw trail.json of the shipped arm, turned into candidates by the Decision
Inspector's own build_cands / chain, inspector_trail_export.py) and for our 158 benchmark clips (docs/decision_trail/data.js,
built by that exporter from the same trail format). Only stage-4 records feed the features (stage 5/6 never do).
Inputs (a local copy in KS340_DIR; the cluster is only read):
    ssh ... 'cd ~/MscProj_tg/data/work && tar czf - r13freshf0*/SHIP8+MD3+WW5+SL_proposed/*/trail.json' | tar xzf -
    ssh ... 'cat ~/open_data/as_strong/audioset_eval_strong.tsv' > eval_strong.tsv ; mid_to_display_name.tsv > mid.tsv
    python benchmark/gold/coverage/ks340_feats.py   -> prints a format / feature survey"""
import glob, json, math, os, re, sys
from collections import Counter
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold import inspector_trail_export as E
from src.labels import canonical

KS = Path(os.environ.get("KS340_DIR") or r"C:\Users\adamg\AppData\Local\Temp\claude\P--MscProj\a0d0af39-f4d2-4c0f-a22f-83b13bdcdb43\scratchpad\ks340")
STAGE = {s[0]: s[1] for s in E.STEPS}
S4 = {k for k, v in STAGE.items() if v == 4} - {"never_heard"}
KEEP = ("step", "res", "value", "bar", "note", "asks")
NUM = re.compile(r"(-?\d+\.\d+|-?\d+)")
NAN = float("nan")


# ---------------------------------------------------------------- candidates (data.js format) for the 340
def cands_340():
    out = {}
    for f in sorted(glob.glob(str(KS / "r13freshf0*" / "SHIP8+MD3+WW5+SL_proposed" / "*" / "trail.json"))):
        st = Path(f).parent.name
        trail = json.loads(Path(f).read_text(encoding="utf-8"))
        cs = []
        for n in E.build_cands(trail):
            if n["hidden"]:
                continue
            recs = E.chain(n)
            cs.append({"label": n["label"], "start": float(n["start"]), "end": float(n["end"]), "origin": n["origin"],
                       "trail": [{x: r[x] for x in KEEP if x in r} for _s, r in recs]})
        out[st] = cs
    return out


def cands_158():
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    d = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    return {c["clip"]: c["cands"] for c in d["clips"]}


def gold_340():
    names = dict(l.rstrip("\n").split("\t") for l in (KS / "mid.tsv").read_text(encoding="utf-8").splitlines() if "\t" in l)
    g = {}
    for l in (KS / "eval_strong.tsv").read_text(encoding="utf-8").splitlines()[1:]:
        sid, a, b, mid = l.split("\t")
        g.setdefault(sid, []).append({"label": names.get(mid, mid), "start": float(a), "end": float(b)})
    return g


# ---------------------------------------------------------------- features
def _after(txt, key):
    if not txt or key not in txt:
        return NAN
    m = NUM.search(txt[txt.index(key) + len(key):])
    return float(m.group(1)) if m else NAN


ORIGINS = ["beats", "flexsed", "flexsed band", "dasm"]
DROPS = ["survived", "beats_extract", "flexsed_extract", "mirror_veto", "masked_weak", "dasm_clip_veto", "dasm_local_veto",
         "k4a_inventory", "continuation_veto", "flexsed_cross_veto", "panns_clip_veto", "band_rescue", "dasm_rescue",
         "finelap_veto", "dasm_vote", "rescue_once", "scene_margin", "other"]
FAMS = []          # filled by set_families()


def set_families(rows_labels, k=25):
    c = Counter(canonical(l) for l in rows_labels)
    FAMS[:] = [f for f, _ in c.most_common(k)]


def s4_fate(recs):
    alive, at = True, None
    for r in recs:
        if r["res"] == "drop":
            alive, at = False, r["step"]
        elif r["res"] == "rescue":
            alive, at = True, None
    return at


def feats(c):
    recs = [r for r in c["trail"] if r["step"] in S4]
    f = {"len": c["end"] - c["start"], "start0": float(c["start"] < 0.2)}
    for o in ORIGINS:
        f["o_" + o] = float(c["origin"] == o)
    for k in ("beats_peak", "flex_peak", "band_peak", "dasm_run", "mir_top", "mir_own", "mw_conf", "mw_sm", "dasm_clip", "dasm_local",
              "dasm_vote", "finelap", "flex_clip", "panns_clip", "q_list", "af_list", "q_logit", "q_v1", "q_v2", "scene"):
        f[k] = NAN
    f.update(twin=0.0, refine=0.0, pull=0.0, cont_exempt=0.0, n_asks=0.0, n_steps=float(len(recs)))
    for r in recs:
        st, v, note = r["step"], r.get("value") or "", r.get("note") or ""
        if st == "beats_extract":
            f["beats_peak"] = _after(v, "peak")
        elif st == "flexsed_extract":
            f["flex_peak"] = _after(v, "peak")
        elif st == "band_rescue":
            f["band_peak"] = _after(v, "run peak")
        elif st == "dasm_rescue":
            f["dasm_run"] = _after(v, "run peak")
        elif st == "twin_union":
            f["twin"] = float("absorbed" in note)
        elif st == "mirror_veto" and "top here" in v:
            f["mir_top"] = _after(v, ") ") if ") " in v else NAN
            f["mir_own"] = float(v.split()[-1]) if "own family" in v else NAN
        elif st == "masked_weak":
            f["mw_conf"], f["mw_sm"] = _after(v, "BEATs conf"), _after(v, "in span")
        elif st == "dasm_clip_veto" and "n/a" not in v:
            f["dasm_clip"] = float(v.split()[-1]) if v else NAN
        elif st == "dasm_local_veto" and "n/a" not in v:
            f["dasm_local"] = _after(v, "+- 0.5 s")
        elif st == "dasm_vote":
            f["dasm_vote"] = _after(v, "DASM max")
        elif st == "finelap_veto":
            f["finelap"] = _after(v, "FineLAP max")
        elif st == "flexsed_cross_veto" and "no query" not in v:
            f["flex_clip"] = float(v.split()[-1]) if v else NAN
        elif st == "panns_clip_veto":
            f["panns_clip"] = float(v.split()[-1]) if v else NAN
        elif st == "onset_refine":
            f["refine"] = 1.0
        elif st == "band_twin_pull":
            f["pull"] = 1.0
        elif st == "continuation_veto":
            f["cont_exempt"] = float("exempt" in note)
        elif st == "scene_margin":
            f["scene"] = float(r["res"] != "drop")
        for a in r.get("asks") or []:
            who, vote, ans = a.get("who", ""), a.get("vote", ""), a.get("a", "")
            if vote == "missing" or "not asked" in ans:
                continue
            f["n_asks"] += 1
            yes = float(vote.startswith("names") or vote.startswith("accept"))
            if "open list" in who:
                k = "q_list" if who.startswith("Qwen3-Omni") else "af_list" if who.startswith("Audio Flamingo") else None
                if k:
                    f[k] = yes if math.isnan(f[k]) else max(f[k], yes)
            elif who.startswith("Qwen3-Omni") and "V1" in who:
                f["q_v1"] = yes if math.isnan(f["q_v1"]) else max(f["q_v1"], yes)
            elif who.startswith("Qwen3-Omni") and "logit" in ans:
                x = _after(ans, "=")
                k = "q_v2" if "V2" in who else "q_logit"
                f[k] = x if math.isnan(f[k]) else max(f[k], x)
    at = s4_fate(recs)
    at = "survived" if at is None else at if at in DROPS else "other"
    for d in DROPS:
        f["at_" + d] = float(at == d)
    fam = canonical(c["label"])
    for x in FAMS:
        f["fam_" + x] = float(fam == x)
    f["fam_other"] = float(fam not in FAMS)
    f["generic"] = float(not S._specific(c["label"]))
    return f, at


def survey():
    a, b = cands_340(), cands_158()
    for name, D in (("340", a), ("158", b)):
        n = sum(len(v) for v in D.values())
        who, steps, ats = Counter(), Counter(), Counter()
        for cs in D.values():
            for c in cs:
                ats[feats(c)[1]] += 1
                for r in c["trail"]:
                    if r["step"] in S4:
                        steps[r["step"]] += 1
                        for q in r.get("asks") or []:
                            who[(r["step"], q["who"].split(" on ")[0][:40], "asked" if q.get("vote") != "missing" else "missing")] += 1
        print(f"== {name}: {len(D)} clips, {n} candidates")
        print("  s4 fate:", dict(ats.most_common()))
        print("  steps:", dict(steps.most_common()))
        for k, v in sorted(who.items()):
            print("   ", k, v)
    k1 = set().union(*[set(r) for cs in a.values() for c in cs for r in c["trail"]])
    k2 = set().union(*[set(r) for cs in b.values() for c in cs for r in c["trail"]])
    print("record keys 340:", sorted(k1), " 158:", sorted(k2))


if __name__ == "__main__":
    survey()
