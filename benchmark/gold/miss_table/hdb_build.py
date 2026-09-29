"""classify each missed needed DEV sound of the best arm TO1+F7F8 (hdb_raw.json) and list its wrong pictures; write the outputs
(adapted from hd_build.py)"""
import json
from collections import OrderedDict, Counter
from pathlib import Path

ROOT = Path("P:/MscProj")
SCR = Path(__file__).resolve().parent
ARM = "TO1+F7F8"
OUT_JSON = ROOT / "benchmark/gold/dev_heard_dropped_best.json"
OUT_MD = ROOT / "docs/dev_heard_dropped_best_2026-09-29.md"
CAUSES = OrderedDict([
    ("unheard", "no detector reaches its heard bar in the window"),
    ("not drawable", "label outside the drawable (depictable) vocabulary"),
    ("below every bar", "heard, but under every bar (BEATs display 0.35, FlexSED 0.8) and the listener was not asked"),
    ("listener rejected", "asked; rule TIER said no (Qwen V4 no, or peak < 0.6 and Audio Flamingo V4 no)"),
    ("killed by F8 (DASM)", "rescued by the listener, then dropped: DASM below its bar 0.575 in the span +- 0.5 s"),
    ("killed by ONCE", "rescued, then dropped: not the first rescued span of its family"),
    ("killed by F7", "BEATs span dropped by the mirror veto b 0.7 and not confirmed by the listener"),
    ("killed by F4", "local-winner filter (off in this arm)"),
    ("PANNs veto", "FlexSED >= 0.8 span dropped by the PANNs clip veto, PV listener item said no"),
    ("gate visible", "a same-family span reached stage 5 and the gate called the source visible"),
    ("timing / merge", "a same-family picture exists but starts outside [onset - 0.5, onset + 1.0] s"),
])


def ovl(a0, a1, b0, b1):
    return min(a1, b1) - max(a0, b0) > -1e-9


def lis_txt(x):
    return (f"{x['pool']} {x['label']} {x['start']}-{x['end']} peak {x['peak']}: Qwen V4 {'yes' if x['qwen_v4'] else 'no'} "
            f"('{x['qwen_v4_text'][:40]}'), AF V4 {'yes' if x['af_v4'] else 'no'} ('{(x['af_v4_text'] or '')[:40]}'), "
            f"TIER {'yes' if x['tier'] else 'no'}" + (f" [{x['tier_fail']}]" if x["tier_fail"] else ""))


def classify(r):
    on = r["onset"]
    wa, wb = on - 0.5, on + 1.0
    if not r["heard_any"]:
        return "unheard", "no detector reaches its heard bar in the window (BEATs %.3f, FlexSED %.3f, PANNs %.3f, extra %.3f)" % (
            r["ev"]["beats"]["score"], r["ev"]["flex"]["score"], r["ev"]["panns"]["score"], r["ev"]["xq"]["score"]) + (
            "" if r["depictable"] else "; also outside the drawable vocabulary")
    if not r["depictable"]:
        xq = [x for x in r["runs"] if x["src"] == "xq"]
        return "not drawable", ("label outside the drawable vocabulary (no FlexSED-215 query); heard only by the extra query "
                                f"{xq[0]['query']} {xq[0]['peak']} for {xq[0]['len']} s (XQ off)" if xq else "label outside the drawable vocabulary")
    # rescued, then killed by a named filter
    for f, name in (("F8", "killed by F8 (DASM)"), ("ONCE", "killed by ONCE"), ("F4", "killed by F4")):
        for x in r["dropped"].get(f, []):
            if ovl(x[1], x[2], wa, wb):
                li = [l for l in r["lis"] if l["tier"] and abs(l["start"] - x[1]) < 0.05]
                why = f"DASM {x[3]} < bar 0.575" if f == "F8" else str(x[3])
                return name, (f"rescued {x[0]} {x[1]}-{x[2]} (" + (lis_txt(li[0]) if li else "listener yes") + f"); dropped by {f}: {why}")
    for x in r["f7"]:
        if not x[3]:
            return "killed by F7", f"BEATs span {x[0]} {x[1]}-{x[2]} mirror-vetoed, listener P1 not confirming ({x[4]})"
    # gate visible: a same-family stage-5 span with an in-window onset that the gate called visible
    inwin4 = [x for x in r["s4"] if x[6]]
    vis = [s for s in r["spec"] if "source visible" in s[5] and wa <= s[1] <= wb]
    if vis:
        s = vis[0]
        gv = [v for v in r["gate_votes"] if v[1] and ovl(v[1][0], v[1][1], wa, wb)]
        return "gate visible", (f"span {s[0]} {s[1]}-{s[2]} (conf {s[3]}) reached stage 5; gate: '{s[5]}'" +
                                ("; votes over the window: " + ", ".join(f"{v[1][0]:.1f}-{v[1][1]:.1f} s seen={v[2]} '{v[3]}'" for v in gv) if gv else ""))
    early = [p for p in r["pics"] if not (wa <= p[1] <= wb) and p[1] <= on <= p[2] + 1.0]
    if early:
        p = early[0]
        return "timing / merge", (f"picture {p[0]} {p[1]}-{p[2]} covers the onset but starts {round(on - p[1], 2)} s early "
                                  f"(stage-4 spans: " + "; ".join(f"{x[0]} {x[1]}-{x[2]} {x[4]} {x[3]}" for x in r["s4"]) + ")")
    # PANNs veto: FlexSED >= 0.8 in the window with no stage-4 span, PV item asked
    r215 = [x for x in r["runs"] if x["src"] == "215"]
    pv = [l for l in r["lis"] if l["pool"] == "PV"]
    if r215 and r215[0]["peak"] >= 0.8 and not inwin4 and not r["b_kept"]:
        return "PANNs veto", (f"FlexSED {r215[0]['query']} {r215[0]['peak']} >= 0.8, PANNs {r['ev']['panns']['score']}; " +
                              (lis_txt(pv[0]) if pv else "no PV item"))
    # listener asked (P2 run in the band) and TIER said no
    p2 = [l for l in r["lis"] if l["pool"] == "P2" and wa - 0.5 <= l["start"] <= wb]
    if p2 and not any(l["tier"] for l in p2):
        extra = ""
        if r["ev"]["beats"]["score"] >= 0.35:
            extra = f"; BEATs {r['ev']['beats']['label']} {r['ev']['beats']['score']} for one 0.25-s frame, cut by the 0.5-s min span"
        return "listener rejected", "; ".join(lis_txt(l) for l in sorted(p2, key=lambda z: -z["peak"])) + extra
    # below every bar
    bits = []
    weak = [x for x in inwin4 if x[3] < 0.35]
    if weak:
        w = max(weak, key=lambda x: x[3])
        bits.append(f"in-window BEATs span {w[0]} {w[1]}-{w[2]} conf {w[3]} < display 0.35")
    if r215 and r215[0]["peak"] >= 0.8:
        x = r215[0]
        bits.append(f"FlexSED {x['query']} {x['peak']} >= 0.8 but its run {x['start']}-{x['end']} starts outside the window")
    elif r215:
        x = r215[0]
        bits.append(f"FlexSED {x['query']} {x['peak']} < 0.8 (run {x['start']}-{x['end']}, {x['len']} s)")
        cov = any(ovl(s[1], s[2], x["start"], x["end"]) for s in r["s4"])
        if x["peak"] < 0.8:
            bits.append("listener not asked: " + ("the run is covered by a same-family stage-4 span" if cov and x["peak"] >= 0.5
                                                  else f"run peak {x['peak']} < LO 0.5"))
    if r["lis_p1"]:
        bits.append("P1 item(s) (not used by the rescue): " + "; ".join(
            f"{x['label']} {x['start']}-{x['end']} Qwen {'/'.join(x['yes']) or 'no'} yn {x['yn']}" for x in r["lis_p1"][:2]))
    xq = [x for x in r["runs"] if x["src"] == "xq" and x["peak"] >= 0.8]
    if xq:
        bits.append(f"extra query {xq[0]['query']} {xq[0]['peak']} for {xq[0]['len']} s (XQ off)")
    return "below every bar", "; ".join(bits)


# hand notes (checked against hdb_raw.json)
NOTES = {
    ("ly_ambulance_(siren)_-yPSgCn", 7.3): "The long Vehicle span 0.0-14.75 (BEATs 0.611) was gated visible (stretch 4.9-9.8 s seen=True, named 'nothing'), but it starts 7.3 s before the onset, so it could not hit anyway: with FIX_GATE (TO1+F7F8+FIX) it is drawn at 0.0 s as a cross wrong picture and 7.3 s is still missed. The FlexSED Ice cream truck span (0.94, from 0.0 s) was PANNs-vetoed and its PV item said no (Qwen V4 'Siren', AF 'siren'), also out of window. The only in-window candidate is Car 8.25 s (P1 item: Qwen V1 yes, yn 4.6; P1 is not a rescue pool).",
    ("as_explosion_XJ8lc3I6", 2.8): "TIER+ONCE+R13-1 (no F8) hits it; F8 then drops it in this arm. Its twin at 5.6 s passed F8 and is hit.",
    ("ambient_citywalk_nyc_1689", 13.7): "TIER+ONCE+R13-1 (no F8) hits it; F8 then drops it in this arm (DASM far below its bar).",
    ("as_explosion_XJ8lc3I6", 6.7): "Qwen V4 names the explosions, not the gasp; V1/V2/V12 said yes (the old best arm's V12 rescued it, F8 then killed it).",
    ("birds_forest", 1.3): "The weak BEATs Bird span 2.22-2.75 (0.221) overlaps the FlexSED run 0.52-6.36, so the band rescue skips the run; the P1 item on that BEATs span says yes (V1, yn 6.5) but P1 is not a rescue pool.",
    ("ly_applause_62ZYD0u", 1.9): "The FlexSED Crowd span 0.0-1.04 (0.867) and the Applause span 1.12-7.75 are merged into one picture that starts at 0.0 s; the same picture is a cross wrong picture.",
}
WNOTE = {
    ("b3_laundromat", 1.0): "NEW: F7 dropped the three Vehicle pictures (listener did not confirm); these BEATs Train spans were hidden inside the Vehicle picture in B0r (FlexSED Train 0.71)",
    ("b3_laundromat", 16.25): "NEW: as above (in B0r: 'same picture as Vehicle'); FlexSED Train 0.71",
    ("bell_miami", 8.0): "NEW: R13-1 (TWIN_MAX) lifts the BEATs Train horn span 0.27 -> 0.408 with its FlexSED Train twin (0.86); over the needed church bell",
    ("b3_golf_course", 18.84): "NEW: rescued FlexSED Bird run (0.785, TIER yes) inside a long needed Bird sound (0-28 s), off its onset",
    ("movie_blueplanet_115", 0.0): "NEW: rescued FlexSED Laughter (0.674) on visible ducks quacking",
    ("un_driving_motorcycle_DgdHSmwA", 13.52): "NEW: rescued FlexSED Gunshot (0.74) on the obvious fireworks",
    ("mv_protest_scene_movie", 4.75): "BEATs Shatter mirror-vetoed, kept by F7 (listener P1 yes); off the needed Glass onset",
    ("ly_applause_62ZYD0u", 0.0): "the merged Crowd picture of the Crowd 1.9 s miss",
    ("as_explosion_XJ8lc3I6", 8.25): "starts 1.85 s before the visible machine gun",
    ("mv_detective_crime_scene", 3.92): "the ringing phone again (Telephone 0.0 s already hit)",
    ("mv_detective_crime_scene", 9.08): "the ringing phone again",
    ("b3_barbershop", 16.0): "second picture of the shaver already hit at 0.1 s",
}


def main():
    R = json.loads((SCR / "hdb_raw.json").read_text(encoding="utf-8"))
    rows = []
    for r in R["needed"]:
        c, why = ("hit", "hit") if r["hit_best"] else classify(r)
        r215 = [x for x in r["runs"] if x["src"] == "215"]
        rows.append({
            "clip": r["clip"], "label": r["label"], "onset": r["onset"], "end": r["end"], "importance": r["importance"],
            "depictable": r["depictable"], "hit_B0r": r["hit_B0r"], "hit_best": r["hit_best"],
            "heard_by": [k for k, v in r["heard"].items() if v],
            "evidence_window": {k: [v["score"], v["label"]] for k, v in r["ev"].items()},
            "flexsed_run": r215[0] if r215 else None, "runs_all": r["runs"][:6],
            "listener_P2_PV": r["lis"], "listener_P1": r["lis_p1"],
            "rescued_added": r["a_added"], "pv_kept": r["b_kept"], "filters_dropped": r["dropped"],
            "mirror_dropped": r["mirror_dropped"], "f7_decisions": r["f7"],
            "stage4_family": r["s4"], "stage4_B0r_family": r["s4_B0r"], "stage5_family": r["spec"],
            "gate_votes": r["gate_votes"], "pictures": r["pics"], "pictures_B0r": r["pics_B0r"],
            "cause": c, "why": why, "note": NOTES.get((r["clip"], r["onset"]), "")})
    miss = [x for x in rows if x["cause"] != "hit"]
    assert len(miss) == 18, len(miss)
    counts = {k: sum(x["cause"] == k for x in miss) for k in CAUSES}
    W = R["wrong"]
    for w in W:
        w["note"] = WNOTE.get((w["clip"], w["start"]), "")
    wc = {"by_type": dict(Counter(w["type"] for w in W)), "by_origin": dict(Counter(w["origin"] for w in W)),
          "type_x_origin": {t: {o: sum(1 for w in W if w["type"] == t and w["origin"] == o) for o in ("BEATs", "FlexSED", "rescued")}
                            for t in ("visible", "cross", "phantom")},
          "new_vs_B0r": sum(1 for w in W if not w["in_B0r"]), "f7_kept": sum(1 for w in W if w["f7_kept"])}
    meta = {
        "what": f"DEV (49 clips, gold_AG, 36 needed sounds): where each needed sound is lost in the best DEV arm {ARM} "
                "(TIER + ONCE + R13-1 + F7 + F8; 18/36 hits, 24 wrong (6/12/6), cost 2.45 vs B0r 2.78), and its 24 wrong pictures",
        "date": "2026-09-29/30", "split": "DEV only; no TEST gold read",
        "sources": {
            "stage4": "data/work/r13/stage4_r14.json = cluster ~/MscProj/data/work/r13/stage4.json (read-only copy): arms['TO1+F7F8|proposed'] "
                      "(post-filter rows, 'rescued' flag), r14_dropped (F8 DASM value / ONCE), listener (a_added band rescues, b_kept "
                      "PANNs-vetoed kept), f7 (mirror-vetoed BEATs spans and the listener's keep decision), mirror",
            "stage5": "data/work/r13/TO1+F7F8_proposed (read-only copy of the cluster folder): augmentations.json, gate_votes.json, _stage5_log.json",
            "pictures": "score_per_sound.load_pictures on data/work/r13/TO1+F7F8_proposed: 18 hits / 24 wrong = round13_dev.json rows[proposed][TO1+F7F8]; "
                        "hits agree with needed_changes on 36/36",
            "listener": "dev_listener_v.json (Qwen3-Omni V1/V2/V3/V4/V12 accept, V4 text, run peak), dev_listener_afn.json (Audio Flamingo "
                        "Next V4 accept + text + yes/no); TIER recomputed as the pipeline does: Qwen V4 AND (peak >= 0.6 OR AF V4); pools P2 "
                        "(band runs) and PV (PANNs-vetoed spans) are the ones the rescue reads; P1 shown for reference",
            "detectors": "heard = same-family best frame score in [onset-0.5, onset+1.0] s: BEATs >= 0.175 (benchmark/gold/beats_fw), FlexSED-215 "
                         ">= 0.4 (data/work/flexsed_cache), PANNs >= 0.1 (benchmark/gold/panns_fw), extra queries >= 0.4 (data/work/flexsed_extra_dev)"},
        "hit_rule": "dev_candidates_check.needed_hit: same family, picture start in [onset-0.5, onset+1.0] s",
        "cause_rule": "first match: unheard; not drawable; rescued then dropped by F8/ONCE/F4; BEATs span dropped by F7; gate visible "
                      "(same-family stage-5 span with in-window start); timing/merge (a same-family picture covers the onset but starts "
                      "outside the window); PANNs veto (FlexSED >= 0.8, no stage-4 span); listener rejected (a P2 item in the window, TIER no); "
                      "else below every bar",
        "wrong_rule": "type = the change of score_clip's counts when the picture is added in start order (miss_feats.py); origin = the "
                      "nearest overlapping same-family stage-4 row: rescued (listener band/PV rescue), FlexSED (FlexSED-only span), BEATs (tagger span)",
        "causes": CAUSES, "scripts": "scratchpad hdb_collect.py + hdb_build.py (adapted from hd_collect.py / hd_build.py)"}
    out = {"meta": meta, "counts": counts, "misses": miss, "hits": [x for x in rows if x["cause"] == "hit"],
           "wrong_counts": wc, "wrong": W}
    OUT_JSON.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ markdown
    L = [f"# DEV: where each needed sound is lost — best arm TO1+F7F8 (2026-09-29)", "",
         "DEV only (49 clips, gold_AG, 36 needed sounds). Arm `TO1+F7F8` = TIER + ONCE + R13-1 + F7 (confirmed mirror veto) + "
         "F8 (DASM vote): 18/36 hits, 24 wrong (6 visible / 12 cross / 6 phantom), cost 2.45 (B0r 14/36, 24 wrong, 2.78). "
         "Built from the arm's real stage-4 / stage-5 outputs (copied read-only from the cluster). Diagnostic only, no selection.", "",
         "## The 18 misses: counts per cause", "", "| cause | meaning | n |", "|---|---|---|"]
    for k, v in CAUSES.items():
        L.append(f"| {k} | {v} | {counts[k]} |")
    L += ["", "## What it shows", "",
          "- Gained over B0r (4): Cricket 0.1 (R13-1), Laughter 8.1 (PV listener keeps it past the PANNs veto), Gunshot 0.0 and "
          "Explosion 5.6 (band rescue). No needed sound is lost.",
          "- F8 kills 2 sounds the rescue had won: Hammer 13.7 (DASM 0.281) and Explosion 2.8 (DASM 0.516, bar 0.575). "
          "F7, ONCE and the PANNs veto kill none of the 18; F4 is off. "
          "F8 also has a +1 side effect: filter_rescued runs F8 before ONCE, so dropping Explosion 2.84 makes Explosion 5.68 the "
          "first rescued Explosion and ONCE keeps it (in TIER+ONCE+1 and TO1+F7, ONCE drops 5.68 as 'not the first'). Net F8 effect on "
          "needed sounds: -2 +1.",
          "- The listener (TIER) says no to 4 heard sounds, all on the Qwen V4 leg: Vehicle/air horn nyc_1689 (V4 'Door closing', AF says "
          "'car horn'), Footsteps 2.1 and Gasp 6.7 (V4 names the explosions), Whistle 6.1 (V4 'Train ...'). AF agrees with the no "
          "except on the air horn.",
          "- 4 are below every bar and never asked: two FlexSED blips under LO 0.5 (Bird 0.435, Siren 0.42, 0.08 s each), birds_forest "
          "Bird (run covered by a weak BEATs span, so the band rescue skips it), and the ambulance Vehicle (in-window Car 0.317 < 0.35; the "
          "long Vehicle span was gated visible but starts 7.3 s early anyway).",
          "- 3 gate visible (macaws, pet-shop birds, church bell): gold says not visible. 1 timing/merge (Crowd picture starts 1.9 s early).",
          "- 3 unheard (Hammer 8.1, Whack x2) and 1 not drawable (Clang, heard only by an extra query).", "",
          "## The 18 misses", "",
          "| clip | sound | onset s | heard by | cause | evidence |", "|---|---|---|---|---|---|"]
    order = list(CAUSES)
    for x in sorted(miss, key=lambda x: (order.index(x["cause"]), x["clip"], x["onset"])):
        L.append(f"| {x['clip']} | {x['label']} | {x['onset']:.1f} | {', '.join(x['heard_by']) or '–'} | {x['cause']} | "
                 f"{x['why']}{(' — ' + x['note']) if x['note'] else ''} |")
    L += ["", "## Per video (needed sounds)", "", "| clip | needed | hit B0r | hit TO1+F7F8 | misses (cause) |", "|---|---|---|---|---|"]
    pc = OrderedDict()
    for x in rows:
        c = pc.setdefault(x["clip"], [0, 0, 0, []])
        c[0] += 1; c[1] += x["hit_B0r"]; c[2] += x["hit_best"]
        if x["cause"] != "hit":
            c[3].append(f"{x['label']}@{x['onset']} ({x['cause']})")
    for k, v in pc.items():
        L.append(f"| {k} | {v[0]} | {v[1]} | {v[2]} | {'; '.join(v[3]) or '–'} |")
    L += ["", "## The 24 wrong pictures", "",
          "Origin = the stage-4 span behind the picture: BEATs (tagger span), FlexSED (FlexSED-only span >= 0.8), rescued (listener rescue). "
          f"{wc['new_vs_B0r']} are new vs B0r; B0r's other 18 are shared. {wc['f7_kept']} is a BEATs span the mirror veto dropped and F7 kept.", "",
          "| type | BEATs | FlexSED | rescued | total |", "|---|---|---|---|---|"]
    for t in ("visible", "cross", "phantom"):
        d = wc["type_x_origin"][t]
        L.append(f"| {t} | {d['BEATs']} | {d['FlexSED']} | {d['rescued']} | {sum(d.values())} |")
    bo = wc["by_origin"]
    L.append(f"| total | {bo.get('BEATs', 0)} | {bo.get('FlexSED', 0)} | {bo.get('rescued', 0)} | {len(W)} |")
    L += ["", "| clip | picture | start-end s | type | origin | vs B0r | stage-4 span (conf) | BEATs / FlexSED / PANNs at start | gold at that time | note |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for w in sorted(W, key=lambda w: (("visible", "cross", "phantom").index(w["type"]), w["origin"], w["clip"], w["start"])):
        s = w["stage4_src"][0] if w["stage4_src"] else None
        L.append(f"| {w['clip']} | {w['label']} | {w['start']}-{w['end']} | {w['type']} | {w['origin']}{' (F7 kept)' if w['f7_kept'] else ''} | "
                 f"{'same' if w['in_B0r'] else 'new'} | {(s[0] + ' ' + str(s[1]) + '-' + str(s[2]) + ' (' + str(s[3]) + ')') if s else '–'} | "
                 f"{w['beats_at'][0]:.2f} / {w['flex_at'][0]:.2f} / {w['panns_at'][0]:.2f} | "
                 f"{'; '.join(g[0] + ' ' + g[3] + ' ' + str(g[1]) + '-' + str(g[2]) for g in w['gold_at_time'][:2]) or '–'} | {w['note'] or ''} |")
    L += ["", "## Data and rules", ""]
    for k, v in meta["sources"].items():
        L.append(f"- {k}: {v}")
    L += [f"- hit: {meta['hit_rule']}", f"- cause rule: {meta['cause_rule']}", f"- wrong pictures: {meta['wrong_rule']}",
          "- Nothing was run or written on the cluster (scp from it only). Machine-readable: `benchmark/gold/dev_heard_dropped_best.json`. "
          "Earlier trace for the round-14 filter arm: `docs/dev_heard_dropped_2026-09-29.md`."]
    OUT_MD.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(counts); print(wc)
    for x in miss:
        print(f"{x['cause']:20s} {x['clip'][:26]:26s} {x['label'][:18]:18s} {x['onset']:5.1f} | {x['why'][:140]}")


if __name__ == "__main__":
    main()
