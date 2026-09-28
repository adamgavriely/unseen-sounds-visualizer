"""feats.json (miss_feats.py) -> benchmark/gold/dev_miss_table.json + docs/dev_miss_table_2026-09-28.md"""
import json, sys
from pathlib import Path

ROOT = Path("P:/MscProj")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import miss_feats as M
from src.labels import canonical
C, S = M.C, M.S
f = json.loads((Path(__file__).resolve().parent / "feats.json").read_text(encoding="utf-8"))
GOLD, _stems = C.dev_stems()
DISP, AED = 0.35, 0.175

NAMES = {"a": "unheard", "b": "FlexSED 0.4-0.8, blocked by 0.5-s min span", "c": "FlexSED 0.4-0.8 span reachable",
         "d": "BEATs 0.175-0.35 (emitted, below display)", "e": "emitted, removed by gate / stage 5",
         "f": "emitted, onset outside window", "g": "emitted, wrong (near) family", "h": "other"}

NOTES = {
 ("ambient_citywalk_nyc_1689", 3.8): "FlexSED Air horn 0.71, span 3.76-4.40 s. Not an R1 band candidate: BEATs self-veto (Air horn clip peak 0.04 < 0.1218).",
 ("ambient_citywalk_nyc_1689", 13.7): "FlexSED Hammer span 14.20-15.04 s. Not an R1 band candidate: self-veto (BEATs Hammer clip peak 0.055).",
 ("ambient_citywalk_nyc_1689", 8.1): "No detector hears it.",
 ("ambient_citywalk_nyc_2627", 3.8): "No detector hears it.",
 ("ambient_nature_rainforest_2179", 6.5): "FlexSED 0.43 for 0.08 s only.",
 ("ambient_nature_rainforest_7629", 0.1): "BEATs Bird vocalization span at 0.89 s (0.46) in the window; the gate said visible ('macaws'). Gold says not visible.",
 ("ambient_nature_rainforest_7629", 0.1, "Cricket"): "BEATs Insect span at 0.14 s has conf 0.31 < 0.35. FlexSED Insect 0.93 (raw span 0-10 s at 0.8) was absorbed into it by the twin rule, which keeps the BEATs conf. A Cricket picture comes only at 4.50 s (counted as a cross wrong picture).",
 ("ambient_snow_walk_930", 8.1): "FlexSED Laughter 0.92, raw span 8.08-8.76 s at the 0.8 bar, dropped by the PANNs clip veto (PANNs Laughter clip peak 0.001 < 0.05). The shipped self-veto drops it too (BEATs Laughter clip peak 0.105 < 0.1218).",
 ("as_explosion_XJ8lc3I6", 0.0): "FlexSED Gunshot 0.55 for 0.24 s. BEATs Fusillade 0.21 for one 0.25-s frame, also below the min span.",
 ("as_explosion_XJ8lc3I6", 2.1): "FlexSED Footsteps span 2.08-2.72 s. Not an R1 band candidate: self-veto (BEATs Footsteps clip peak 0.016).",
 ("as_explosion_XJ8lc3I6", 2.8): "R1 band candidate (Explosion 2.84-3.52 s, 0.62).",
 ("as_explosion_XJ8lc3I6", 5.6): "R1 band candidate (Explosion 5.68-6.20 s, 0.56).",
 ("as_explosion_XJ8lc3I6", 6.7): "FlexSED Gasp 0.76 for 0.20 s. BEATs Gasp 0.41 (above the display bar) for one 0.25-s frame. Both are cut by the 0.5-s min span.",
 ("b3_carnival_parade", 6.1): "FlexSED Steam whistle 0.58, span 6.32-7.16 s. Not an R1 band candidate: self-veto (BEATs 0.003).",
 ("b3_golf_course", 6.5): "No detector hears it.",
 ("b3_golf_course", 24.4): "No detector hears it.",
 ("b3_pet_shop", 0.1): "BEATs Bird 0.86 over the whole clip; the gate said visible ('a small bird') on all 6 stretches.",
 ("bell_miami", 0.2): "BEATs Church bell 0.70; the gate said visible ('church bell').",
 ("birds_forest", 1.3): "BEATs Bird span 2.22-2.75 s, conf 0.22 < 0.35. The FlexSED 0.4 span starts at 0.52 s, 0.28 s before the window.",
 ("ly_ambulance_(siren)_-yPSgCn", 7.3): "The Vehicle span (BEATs 0.61) runs 0.0-14.75 s through the onset, and the gate called it visible. A new Car span starts at 8.25 s with conf 0.32 < 0.35.",
 ("ly_applause_62ZYD0u", 1.9): "The Crowd picture 0.0-7.75 s starts 1.9 s early: a FlexSED Crowd span 0.0-1.04 s (0.87) is merged with the Applause span (BEATs, 1.12 s). The same picture is a cross wrong picture at 0.0 s.",
 ("mv_storm_scene_house", 16.9): "FlexSED Siren 0.42 for 0.08 s.",
}


def rowkey(r):
    k = (r["clip"], r["onset"], r["label"])
    return k if k in NOTES else (r["clip"], r["onset"])


def buckets(r):
    on = r["onset"]
    inw = lambda t: S.in_window(t, on, S.EARLY, S.LATE)
    disp = lambda x: (x[3] >= DISP or x[4] == "flex") and M.sal(x[0])
    out = []
    if any(x[5] and disp(x) for x in r["s4_fam"]):
        out.append("e")
    cov = lambda a, b: (a < on - S.EARLY and b > on) or (on + S.LATE < a <= on + S.LATE + 1.0)
    if any(disp(x) and cov(x[1], x[2]) for x in r["s4_fam"]) or any(cov(p[1], p[2]) for p in r["pics_fam"]):
        out.append("f")
    if r["s4_sibling_inwin"] and any(x[2] >= DISP for x in r["s4_sibling_inwin"]):
        out.append("g")
    # stage-4 veto: a FlexSED 0.8 raw span near the onset with no flex row and no same-key tagger twin
    for lab, a, b in r["trace_flexraw_fam"]:
        if a <= on + S.LATE and b >= on - S.EARLY:
            tw = [x for x in r["s4_fam"] if canonical(x[0]) == canonical(lab) and x[1] - 1.0 <= b and a - 1.0 <= x[2]]
            if not tw:
                out.append("h-veto"); break
    if any(x[5] and x[4] == "tagger" and AED <= x[3] < DISP for x in r["s4_fam"]):
        out.append("d")
    if any(inw(x[1]) and x[3] < 0.8 for x in r["flex04_spans_fam"]):
        out.append("c")
    elif 0.4 <= r["flex_win"] < 0.8 or (r["flex_win"] >= 0.8 and "h-veto" not in out and r["flex_run04_bestlab"] < 0.5):
        out.append("b")
    if AED <= r["beats_win"] < DISP and "d" not in out:
        out.append("d-score")
    if r["beats_near"] < 0.3 and r["flex_near"] < 0.4:
        out.append("a")
    order = ["e", "f", "h-veto", "d", "c", "b", "g", "a"]
    prim = next((b for b in order if b in out), "h")
    return prim, [b for b in out if b != prim]


miss_rows = []
for r in f["misses"]:
    prim, also = buckets(r)
    cross = [f"{x[0]} {x[1]:.2f} s ({x[2]:.2f}{', shown' if x[2] >= DISP else ''})" for x in r["s4_other_inwin_aed"]]
    cross += [f"picture {p[0]} {p[1]:.2f} s" for p in r["pics_other_inwin"]]
    miss_rows.append({
        "clip": r["clip"], "gold_label": r["label"], "onset": r["onset"], "end": r["end"], "duration": r["dur"],
        "importance": r["importance"],
        "beats_best_in_window": r["beats_win"], "beats_best_label": r["beats_win_label"],
        "flexsed_best_in_window": r["flex_win"], "flexsed_best_label": r["flex_win_label"],
        "flexsed_run_ge04_s": r["flex_run04_bestlab"], "flexsed_run_ge04_span": r["flex_run04_bestlab_span"],
        "panns_best_in_window": r["panns_win"], "panns_best_label": r["panns_win_label"],
        "beats_near": r["beats_near"], "flexsed_near": r["flex_near"],
        "bucket": prim[0] if prim != "h-veto" else "h", "bucket_detail": "stage-4 second-detector veto" if prim == "h-veto" else NAMES[prim[0]],
        "also": also, "r1_band_candidate": bool(r["band_cands_fam"]),
        "cross_label_at_onset": cross, "note": NOTES.get(rowkey(r), ""),
        "stage4_family_spans": r["s4_fam"], "flexsed_04_spans": r["flex04_spans_fam"], "stage5_specs": r["specs_fam"],
        "gate_votes": r["gate_fam"]})

wrong_rows = []
for w in f["wrong"]:
    src = w["stage4_src"][0] if w["stage4_src"] else None
    det = None if src is None else ("FlexSED-only" if src[4] == "flex" else "BEATs")
    sub = ""
    if w["type"] == "cross":
        gold_same = [g for g in GOLD[w["clip"]] if S.same_family(g["label"], w["label"]) and g["start"] - 2.0 <= w["start"] <= g["end"]]
        if gold_same:
            sub = "same family as gold " + ", ".join(f"{g['label']} {g['start']:.1f} s ({'needed' if g['needed'] else 'visible' if g['visible'] else 'obvious'}, imp {g['importance']})" for g in gold_same) + ", off its onset"
    wrong_rows.append({"clip": w["clip"], "shown_label": w["label"], "start": w["start"], "end": w["end"], "type": w["type"],
                       "detector": det, "span": src[:4] if src else None, "score": src[3] if src else None,
                       "picture_conf": w["spec_conf"][0] if w["spec_conf"] else None,
                       "beats_at": w["beats_at"], "flexsed_at": w["flex_at"], "panns_at": w["panns_at"],
                       "gold_at_time": w["gold_at_time"],
                       "collided_with": [f"{g[0]} ({g[3]}, {g[1]:.1f}-{g[2]:.1f} s)" for g in w["gold_at_time"]] if w["type"] != "phantom" else [],
                       "subtype": sub})
# hand reading of the wrong pictures (checked against the rows above)
WNOTE = {("mv_detective_crime_scene", 3.92): "the ringing phone again (Alarm is Telephone's parent; BEATs Telephone bell ringing 0.58); the Telephone at 0.0 s is already hit",
         ("mv_detective_crime_scene", 9.08): "the ringing phone again (BEATs Telephone bell ringing 0.41)",
         ("b3_barbershop", 16.0): "second picture of the shaver that is already hit at 0.1 s",
         ("b3_favela_rio", 14.0): "Bird picture inside a long importance-1 Bird sound (don't care), off its onset",
         ("b3_crossing_bells", 0.22): "near label of the visible train (Steam)",
         ("london_protest_01", 0.25): "Vehicle on the visible air horn",
         ("ly_applause_62ZYD0u", 0.0): "the early Crowd picture that also causes the Crowd 1.9 s miss (f)",
         ("ambient_nature_rainforest_7629", 4.5): "the late Cricket picture of the Cricket 0.1 s miss (d)",
         ("as_explosion_XJ8lc3I6", 8.25): "starts 1.85 s before the visible machine gun (10.1 s)",
         ("b3_laundromat", 0.3): "BEATs Vehicle; FlexSED Train 0.71 at the same time (machine rumble heard as a vehicle)",
         ("b3_laundromat", 14.25): "same as above (FlexSED Train 0.73)",
         ("b3_laundromat", 23.25): "same as above (FlexSED Train 0.77)",
         ("mv_protest_scene_movie", 21.48): "FlexSED-only Train 0.87; BEATs 0.04 and PANNs 0.04 at that time",
         ("un_driving_motorcycle_DgdHSmwA", 13.97): "picture of an obvious sound (fireworks seen)"}
for w in wrong_rows:
    w["note"] = WNOTE.get((w["clip"], w["start"]), "")

cnt = {b: sum(1 for r in miss_rows if r["bucket"] == b) for b in "abcdefgh"}
wc = {"by_type": {t: sum(1 for w in wrong_rows if w["type"] == t) for t in ("visible", "cross", "phantom")},
      "by_detector": {d: sum(1 for w in wrong_rows if w["detector"] == d) for d in ("BEATs", "FlexSED-only")},
      "cross_same_family_off_onset": sum(1 for w in wrong_rows if w["subtype"]),
      "if_timed_right": "Cricket 4.5 and Crowd 0.0 -> hits; Alarm x2 and shaver 16.0 -> dup; favela Bird 14.0 -> dont-care (imp 1); Gunshot 8.25 -> visible (machine gun on screen)"}
assert sum(cnt.values()) == 22 and len(wrong_rows) == 24
res = {"what": "DEV (49 clips, 36 needed sounds) B0 = scored render dev_monocap_v31, ours (proposed, with gate): the 22 missed "
               "needed sounds with one cause bucket each, and the 24 wrong pictures",
       "source": {"pictures": "data/work/protocol_proposed_dev_monocap_v31 (cluster; = devcand/B0r_proposed on 49/49 clips)",
                  "stage4": "data/work/devcand/stage4.json arm B0r|proposed", "beats": "data/work/j2_dev_beats (cluster)",
                  "flexsed": "data/work/flexsed_cache", "panns": "benchmark/gold/panns_fw",
                  "hit_rule": "score_per_sound.needed_hit / score_clip, window [onset-0.5, onset+1.0] s",
                  "score_window": "best same-family frame score (score_per_sound.same_family, the hit rule) with frame time in [onset-0.5, onset+1.0]; BEATs times are 2-s window starts",
                  "near_window": "[onset-1.0, onset+2.0] s (bucket a)"},
       "bucket_rule": "first match wins: e, f, h(veto), d, c, b, g, a, else h; the other matches are in 'also'",
       "bucket_names": NAMES, "counts": cnt, "hits": len(f["hits"]), "misses": miss_rows, "wrong_counts": wc, "wrong": wrong_rows}
(ROOT / "benchmark" / "gold" / "dev_miss_table.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
print(cnt, wc)
for r in miss_rows:
    print(r["bucket"], r["also"], r["clip"][:30], r["gold_label"], r["onset"], "|", "; ".join(r["cross_label_at_onset"]))
for w in wrong_rows:
    print(w["type"], w["detector"], w["clip"][:30], w["shown_label"], w["start"], w["score"], w["subtype"], w["collided_with"])
exec(open(Path(__file__).resolve().parent / "build_md.py", encoding="utf-8").read())
