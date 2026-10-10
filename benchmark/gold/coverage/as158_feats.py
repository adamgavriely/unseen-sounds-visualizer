"""BEATs-strong / PretrainedSED on the 158 benchmark clips (PREREG_beats_strong.md): features for the CPU analysis.
Cluster CPU, from ~/as158 (needs ~/wt_slice for score_per_sound.same_family):
    python as158_feats.py   -> as158_feats.json
Reads sota_<model>.npz (frames [T, 447], 40 ms, classes = AudioSet MIDs) and as158_inputs.json (v1.7 pictures with their
class hit / wrong / neutral, gold sounds with needed / importance / hit by v1.7, ban list)."""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "wt_slice"))
from benchmark.gold import score_per_sound as S

HERE = Path(__file__).resolve().parent
MODELS = ("BEATs", "ATST-F", "fpasst", "M2D", "ASIT")
DT = 0.04
TS = (0.2, 0.3, 0.4, 0.5, 0.6)
NOT_DRAWN = ("Speech", "Music", "Singing", "Male speech", "Female speech", "Child speech", "Conversation", "Narration")


def main():
    names = dict(l.rstrip("\n").split("\t") for l in open(Path.home() / "open_data/as_strong/mid_to_display_name.tsv"))
    I = json.loads((HERE / "as158_inputs.json").read_text())
    P = {m: np.load(HERE / f"sota_{m}.npz") for m in MODELS}
    cls = [names.get(str(c), str(c)) for c in P["BEATs"]["classes"]]
    fam_cache = {}

    def cols(label):
        if label not in fam_cache:
            fam_cache[label] = [i for i, c in enumerate(cls) if S.same_family(c, label)]
        return fam_cache[label]

    def fmax(m, st, label, a, b):
        c = cols(label)
        if not c:
            return None
        y = P[m][st]; i0, i1 = max(0, int(a / DT)), max(int(a / DT) + 1, int(np.ceil(b / DT)))
        seg = y[i0:i1, :][:, c]
        return float(seg.max()) if seg.size else 0.0

    out = {"pics": [], "gold": [], "runs": {}, "n_classes_with_family": {}}
    for st, rows in I["pics"].items():
        for lab, a, b, k in rows:
            f = {"clip": st, "label": lab, "start": a, "end": b, "cls": k, "has_family": bool(cols(lab))}
            for m in MODELS:
                f[f"{m}_in"] = fmax(m, st, lab, a, b); f[f"{m}_on"] = fmax(m, st, lab, a - 0.25, a + 0.75)
            out["pics"].append(f)
    for st, rows in I["gold"].items():
        for lab, a, b, need, imp, hit in rows:
            f = {"clip": st, "label": lab, "start": a, "end": b, "needed": need, "importance": imp, "hit": hit, "has_family": bool(cols(lab))}
            for m in MODELS:
                f[f"{m}_on"] = fmax(m, st, lab, a - 0.5, a + 1.0)
            out["gold"].append(f)
    ban = set(I["ban"])
    drawable = [i for i, c in enumerate(cls) if not any(c.startswith(x) for x in NOT_DRAWN) and c not in ban]
    for st in I["pics"]:
        y = P["BEATs"][st]; per_t = {}
        for t in TS:
            runs = []
            for i in drawable:
                on = y[:, i] >= t
                if not on.any():
                    continue
                k, n = 0, len(on)
                while k < n:
                    if not on[k]:
                        k += 1; continue
                    j = k
                    while j < n and (on[j] or (j + 12 < n and on[j:j + 13].any())):    # gaps < 0.5 s bridged
                        j += 1
                    if (j - k) * DT >= 0.3:
                        runs.append([cls[i], round(k * DT, 2), round(j * DT, 2), round(float(y[k:j, i].max()), 3)])
                    k = j
            per_t[str(t)] = runs
        out["runs"][st] = per_t
    (HERE / "as158_feats.json").write_text(json.dumps(out))
    g = [x for x in out["gold"] if x["needed"] and x["importance"] >= 2]
    miss = [x for x in g if not x["hit"]]
    print("needed", len(g), "missed", len(miss), "missed with a family class", sum(x["has_family"] for x in miss),
          "| pictures", len(out["pics"]), "with a family class", sum(x["has_family"] for x in out["pics"]))


if __name__ == "__main__":
    main()
