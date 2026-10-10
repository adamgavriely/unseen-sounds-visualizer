"""BEATs-strong as a second ear for candidates the pipeline already found and dropped (11 Oct, after PREREG_beats_strong.md
B failed): a dropped stage-4 candidate (docs/decision_trail/data.js) is rescued when a BEATs-strong run of its family at
bar t starts within 1 s of it. Counts per t: rescued candidates that start in the hit window of a needed sound v1.7
misses (recoverable) vs the rest (would be extra wrongs before the gate). Upper bound, all 158 clips.
    python benchmark/gold/coverage/as158_rescue.py -> as158_rescue.md"""
import json, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD, stems
from benchmark.gold.coverage.screen_reason import base_pics

HERE = Path(__file__).resolve().parent


def main():
    F = json.loads((HERE / "as158_feats.json").read_text()); gold = S.load_gold([GOLD]); pics = base_pics()
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = {c["clip"]: c["cands"] for c in json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))["clips"]}
    miss = {(g["clip"], g["label"], g["start"]) for g in F["gold"] if g["needed"] and g["importance"] >= 2 and not g["hit"]}
    L = ["# BEATs-strong rescue of dropped stage-4 candidates (upper bound, all 158 clips)", "",
         "| bar t | rescued candidates | recoverable misses | rescued with no needed sound (extra wrongs before the gate) | 4 x rec - 2 x extra |", "|---|---|---|---|---|"]
    for t in ("0.2", "0.3", "0.4", "0.5", "0.6"):
        n = extra = 0; rec = set()
        for st in stems("dev") + stems("test"):
            runs = F["runs"][st][t]
            for c in dj[st]:
                if c["fate"] != "dropped" or not any(S.same_family(r[0], c["label"]) and abs(r[1] - c["start"]) <= 1.0 for r in runs):
                    continue
                if any(S.same_family(c["label"], p[0]) and p[1] - 0.5 <= c["start"] <= p[2] for p in pics[st]):
                    continue
                n += 1
                m = [g for g in gold[st] if S.same_family(c["label"], g["label"]) and S.in_window(c["start"], g["start"], S.EARLY, S.LATE)]
                hit = [g for g in m if (st, g["label"], g["start"]) in miss]
                if hit:
                    rec.update((st, g["label"], g["start"]) for g in hit)
                elif not any(g["needed"] for g in m):
                    extra += 1
        L.append(f"| {t} | {n} | {len(rec)} | {extra} | {4 * len(rec) - 2 * extra} |")
    (HERE / "as158_rescue.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
