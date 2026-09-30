"""Round 34 MOSS screen (docs/prereg_round13_detector_push.md, "Round 34 MOSS"): on merged-DEV P2/PV candidates (the clips with
gold), the shipped TIER rule (peak >= TIER_SPLIT: Qwen V4; below: Qwen V4 AND AF V4) against two rules that add
MOSS-Audio-8B-Thinking's V4 answer (M):
  (a) MOSS replaces Qwen:   high tier M;               low tier M AND A
  (b) MOSS as a third ear:  high tier Q OR (A AND M);  low tier (Q AND A) OR (M AND (Q OR A))
Counted per rule: needed-class (gold hit_needed) and other-class (other_gold + none) accepts ADDED and LOST vs shipped TIER.
GO iff needed added >= 2, other added <= 2 x needed added, needed lost 0. Sanity: MOSS / Qwen / AF yes-rates on the P1 items of
dev_listener_p1v4.json by the dev_listener.json gold field (hit_needed vs none). CPU, existing caches only.

    python benchmark/gold/moss_screen.py            # from ~/MscProj_tg on the cluster
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold import dev_listener as L
from benchmark.gold.ptc_screen import PARTS, key

G = _ROOT / "benchmark" / "gold"
H = Path.home()
MOSS = {"dev": G / "dev_listener_moss.json", "dev2": G / "dev2_listener_moss.json"}
P1 = {"moss": G / "dev_listener_p1_moss.json", "p1v4": H / "MscProj_r13" / "benchmark" / "gold" / "dev_listener_p1v4.json",
      "gold": H / "MscProj_r13" / "benchmark" / "gold" / "dev_listener.json"}
SPLIT = float(getattr(config, "TIER_SPLIT", 0.6))
KNOWN = ("nyc", "Air horn"), ("tg_d107", "Laughter"), ("carnival", "Whistle"), ("as_explosion", "Footsteps"), \
        ("as_explosion", "Gasp"), ("tg_d033", "Siren"), ("tg_d032", "Thunder")


def rules(q, a, m, peak):
    high = peak >= SPLIT
    tier = q if high else (q and a)
    ra = m if high else (m and a)
    rb = (q or (a and m)) if high else ((q and a) or (m and (q or a)))
    return {"TIER": tier, "A_moss_for_qwen": ra, "B_third_ear": rb}


def main():
    tot = {r: {"needed_added": 0, "other_added": 0, "needed_lost": 0, "other_lost": 0} for r in ("A_moss_for_qwen", "B_third_ear")}
    base = {"items": 0, "needed": 0, "tier_needed": 0, "tier_other": 0, "moss_needed": 0, "moss_other": 0, "moss_null": 0,
            "missing_moss": 0}
    lines, known = [], []
    for part, (vp, ap, _pp, gp, _dd) in PARTS.items():
        v = {key(x): x for x in json.loads(vp.read_text(encoding="utf-8"))["items"] if x.get("pool") in ("P2", "PV")}
        af = {key(x): x for x in json.loads(ap.read_text(encoding="utf-8"))["items"]}
        mo = {key(x): x for x in json.loads(MOSS[part].read_text(encoding="utf-8"))["items"]}
        gold = S.load_gold([gp])
        for k, x in v.items():
            if x["clip"] not in gold:
                continue
            peak = float(x.get("peak") or 0.0)
            q = bool(x["accept"].get("V4"))
            a = bool((af.get(k, {}).get("accept") or {}).get("V4"))
            if k not in mo:
                base["missing_moss"] += 1
            m = bool((mo.get(k, {}).get("accept") or {}).get("V4"))
            cls = L.gold_class(gold[x["clip"]], x["label"], x["start"], x["end"])
            needed = cls == "hit_needed"
            r = rules(q, a, m, peak)
            base["items"] += 1; base["needed"] += needed
            base["tier_needed" if needed else "tier_other"] += r["TIER"]
            base["moss_needed" if needed else "moss_other"] += m
            base["moss_null"] += bool((mo.get(k, {}).get("null_accept") or {}).get("V4"))
            desc = (f"[{part}] {x['clip']} {x['label']!r} {x['start']:.2f}-{x['end']:.2f} peak {peak:.2f} [{cls}] Q {int(q)} A {int(a)} "
                    f"M {int(m)} | MOSS: {mo.get(k, {}).get('moss_v4_text', '')[:70]!r}")
            for rn in tot:
                if r[rn] and not r["TIER"]:
                    tot[rn]["needed_added" if needed else "other_added"] += 1
                    lines.append(f"{rn} + {desc}")
                elif r["TIER"] and not r[rn]:
                    tot[rn]["needed_lost" if needed else "other_lost"] += 1
                    lines.append(f"{rn} - {desc}")
            if any(x["clip"].startswith(c) and x["label"] == lab for c, lab in KNOWN) and needed:
                known.append(desc)
    go = {rn: t["needed_added"] >= 2 and t["other_added"] <= 2 * t["needed_added"] and t["needed_lost"] == 0 for rn, t in tot.items()}
    for s in lines:
        print(s)
    print("known hit-side misses (needed items):")
    for s in known:
        print("   ", s)
    # sanity: P1 yes-rates by gold class (MOSS vs Qwen vs AF on the same p1v4 items)
    p1 = {}
    if all(p.exists() for p in P1.values()):
        gcls = {(x["clip"], x["label"], round(x["start"], 2), round(x["end"], 2)): x.get("gold")
                for x in json.loads(P1["gold"].read_text(encoding="utf-8"))["items"] if x["pool"] == "P1"}
        pm = {(x["clip"], x["label"], round(x["start"], 2), round(x["end"], 2)): x
              for x in json.loads(P1["moss"].read_text(encoding="utf-8"))["items"]}
        for x in json.loads(P1["p1v4"].read_text(encoding="utf-8"))["items"]:
            kk = (x["clip"], x["label"], round(x["start"], 2), round(x["end"], 2))
            c = gcls.get(kk)
            if c not in ("hit_needed", "none") or kk not in pm:
                continue
            e = p1.setdefault(c, {"n": 0, "moss": 0, "qwen": 0, "af": 0})
            e["n"] += 1
            e["moss"] += bool(pm[kk]["accept"].get("V4"))
            e["qwen"] += x["family"] in (x.get("qwen_fams") or [])
            e["af"] += x["family"] in (x.get("af_fams") or [])
    print("base:", json.dumps(base))
    print("P1 sanity (yes-rates by gold class, same cuts):", json.dumps(p1))
    for rn, t in tot.items():
        print(f"MOSS screen {rn}: {json.dumps(t)} {'GO' if go[rn] else 'STOP'}")
    (G / "moss_screen.json").write_text(json.dumps({"base": base, "rules": tot, "go": go, "p1_sanity": p1, "known_misses": known,
                                                    "changes": lines}, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
