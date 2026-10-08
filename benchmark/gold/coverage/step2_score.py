"""Step 2 (PREREG_step2_gate_mirror.md): scorer v2 on every cell of step2_pics_dev.json (step2_gate_build.py), DEV only.

    python benchmark/gold/coverage/step2_score.py   -> step2_dev.json, step2_dev.md
"""
import json
import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.coverage import score_coverage as V

HERE = Path(__file__).resolve().parent
FROZEN = "SHIP8+MD3+WW5+SL|M"
NAMES = {"SHIP8+MD3+WW5+SL": "D'", "SHIP8+MD3+WW5+SL+TLS": "TL-span", "SHIP8+MD3+WW5+SL+TLO": "TL-onset",
         "SHIP8+MD3+WW5+SL+UNO3": "ATST-F untrained 0.3", "SHIP8+MD3+WW5+SL+UNO5": "ATST-F untrained 0.5",
         "SHIP8+MD3+WW5+SL+UNF3": "ATST-F fine-tuned 0.3", "SHIP8+MD3+WW5+SL+UNF5": "ATST-F fine-tuned 0.5"}


def name(cell):
    arm, rule = cell.split("|")
    n = NAMES.get(arm, arm)
    return n if rule == "M" else (rule if n == "D'" else f"{n}+{rule}")


def main():
    gold = S.load_gold([V.GOLD])
    dev = V.stems("dev")
    d = json.loads((HERE / os.environ.get("S2_PICS", "step2_pics_dev.json")).read_text(encoding="utf-8"))
    rows = {c: {st: V.score_clip_v2(gold[st], [tuple(x) for x in v["clips"][st]["pics_none"]]) for st in dev}
            for c, v in d["cells"].items()}
    res = {"cells": {}, "log": d.get("log", {})}
    for c, rr in rows.items():
        a = V.aggregate([rr[st] for st in dev])
        a["pass"] = a["hits"] >= 31 and a["wrong"] <= 16
        a["changed"] = [[st, rows[FROZEN][st]["hit"], rr[st]["hit"],
                         sum(rows[FROZEN][st][k] for k in ("visible", "cross", "phantom")), sum(rr[st][k] for k in ("visible", "cross", "phantom"))]
                        for st in dev if (rr[st]["hit"], rr[st]["visible"] + rr[st]["cross"] + rr[st]["phantom"]) !=
                        (rows[FROZEN][st]["hit"], rows[FROZEN][st]["visible"] + rows[FROZEN][st]["cross"] + rows[FROZEN][st]["phantom"])]
        res["cells"][c] = a
    ok = [c for c, a in res["cells"].items() if a["pass"]]
    key = lambda c: (round(res["cells"][c]["onset_cost"], 2), res["cells"][c]["cost_cov"])
    res["passing"] = ok
    res["selected"] = min(ok, key=key) if ok else None
    (HERE / (os.environ.get("S2_OUT", "step2_dev") + ".json")).write_text(json.dumps(res, indent=1), encoding="utf-8")
    L = ["| cell | hits | wrong | onset cost | cost_cov | hit cover | wrong s/clip | stale s/clip | pass (>= 31 hits, <= 16 wrong) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for c in sorted(res["cells"], key=lambda c: (c != FROZEN, res["cells"][c]["onset_cost"])):
        a = res["cells"][c]
        L.append(f"| {name(c)} | {a['hits']} | {a['wrong']} | {a['onset_cost']:.3f} | {a['cost_cov']:.3f} | {a['hit_cov']:.2f} | "
                 f"{a['wrong_s_per_clip']:.2f} | {a['stale_s_per_clip']:.2f} | {'yes' if a['pass'] else 'no'} |")
    L += ["", f"passing: {[name(c) for c in ok] or 'none'}; selected: {name(res['selected']) if res['selected'] else 'none'}", "",
          "Clips that change against D' (hits before -> after, wrong before -> after):"]
    for c in sorted(res["cells"]):
        if c != FROZEN and res["cells"][c]["changed"]:
            L.append(f"- {name(c)}: " + "; ".join(f"{st} {h0}->{h1} / {w0}->{w1}" for st, h0, h1, w0, w1 in res["cells"][c]["changed"]))
    (HERE / (os.environ.get("S2_OUT", "step2_dev") + ".md")).write_text("# Step 2, DEV (71 clips)\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
