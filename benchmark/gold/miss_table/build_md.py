# executed at the end of build_out.py (shares its names): writes docs/dev_miss_table_2026-09-28.md
def fmt(v, lab=None):
    return f"{v:.2f}" + (f" ({lab})" if lab and v >= 0.1 else "")


L = []
A = L.append
A("# DEV misses and wrong pictures of B0 (scored render, ours with gate) — 2026-09-28")
A("")
A("*Diagnostic, DEV only (49 clips, 36 needed sounds). No TEST data. B0 = `dev_monocap_v31` with the PANNs veto 0.05, the arm "
  "that the DEV check calls B0 (docs/dev_candidates_check_2026-09-28.md): 14 hits, 22 misses, 24 wrong pictures (6 visible / "
  "11 cross / 7 phantom). All three numbers are reproduced here with `score_per_sound`. Data: `benchmark/gold/dev_miss_table.json`.*")
A("")
A("## How it was made")
A("- **Pictures**: `data/work/protocol_proposed_dev_monocap_v31/<clip>/augmentations.json`, copied from the cluster. They equal "
  "`devcand/B0r_proposed` on 49 / 49 clips. Read with `score_per_sound.load_pictures`. Hit or miss per needed sound: "
  "`dev_candidates_check.needed_hit` (same hits as `score_clip`). The type of each wrong picture = the change in `score_clip`'s "
  "counts when that picture is added (pictures in start order, the scorer's own order).")
A("- **Stage 4**: arm `B0r|proposed` of `data/work/devcand/stage4.json`. These are the spans after the vetoes and onset refinement; "
  "origin *tagger* = BEATs, *flex* = FlexSED-only. The scored run's `onset_trace.json` gives the FlexSED 0.8 spans before the "
  "vetoes. `gate_votes.json` and each picture spec's `reason` give stage 5.")
A("- **Scores**: BEATs `data/work/j2_dev_beats` (2-s windows, 0.25-s hop; time = window start). FlexSED `data/work/flexsed_cache` "
  "(25 fps). PANNs `benchmark/gold/panns_fw` (100 fps). Each score = the best same-family frame (same canonical label, or "
  "ancestor / descendant below the top level: `score_per_sound.same_family`, the hit rule) with frame time in the hit window **[onset − 0.5, onset + 1.0] s**, as in "
  "the 09-27 table. *FlexSED run* = the longest stretch ≥ 0.4 of the best FlexSED label that touches the window. *Near* = "
  "[onset − 1, onset + 2] s.")
A("- Scripts: `benchmark/gold/miss_table/` (miss_feats.py -> build_out.py -> build_md.py); inputs partly copied read-only from the cluster.")
A("")
A("**Buckets.** The first match wins, in this order. Every other match is listed under *also*.")
A("1. **(e)** Stage 4 has a same-family span that could be shown (conf ≥ 0.35, or FlexSED-only) and it **starts in the window**, "
  "but there is no picture: the gate or stage 5 removed it.")
A("2. **(f)** A same-family span or picture that could be shown **covers the onset** (it starts earlier and is still on), or it "
  "starts up to 1 s after the window. A much later picture of the same sound does not count.")
A("3. **(h)** A FlexSED 0.8 span near the onset is in the trace's `flexsed_raw` step but not in its `veto` step, and it has no "
  "same-family BEATs twin: the stage-4 clip veto removed it.")
A("4. **(d)** A same-family BEATs span with 0.175 ≤ conf < 0.35 starts in the window (emitted, but below the display bar). "
  "*d-score* under *also* = BEATs is 0.175–0.35 in the window, but there is no span.")
A("5. **(c)** FlexSED at a 0.4 bar (`_extract_events`, 0.5-s min span) gives a same-family span that starts in the window, "
  "peak < 0.8.")
A("6. **(b)** FlexSED is 0.4–0.8 in the window, but gives no such span (its run is shorter than 0.5 s).")
A("7. **(g)** A span of a sibling label (same direct parent), conf ≥ 0.35, starts in the window.")
A("8. **(a)** BEATs < 0.3 and FlexSED < 0.4 near the onset. Anything else is **(h)**.")
A("")
A("## Counts")
A("")
A("| bucket | what | misses |")
A("|---|---|---|")
for b in "abcdefgh":
    A(f"| ({b}) | {NAMES[b] if b != 'h' else 'other (here: the stage-4 veto)'} | {cnt[b]} |")
A(f"| | **total** | **{sum(cnt.values())}** |")
A("")
A(f"Wrong pictures: {wc['by_type']['visible']} visible, {wc['by_type']['cross']} cross, {wc['by_type']['phantom']} phantom. "
  f"{wc['by_detector']['BEATs']} come from BEATs spans, {wc['by_detector']['FlexSED-only']} from FlexSED-only spans. "
  f"{wc['cross_same_family_off_onset']} of the 11 cross pictures show the right family of a real gold sound, but not at its onset.")
A("")
A("## Where the levers are")
A("1. **FlexSED 0.4–0.8 band: 10 of 22** (c 6, b 4). Of the 6 reachable spans, only the 2 explosions pass R1's own filter. The "
  "other 4 (air horn, hammer, footsteps, whistle) are removed by the BEATs self-veto, because BEATs does not hear them at all "
  "(clip peak ≤ 0.055). So a band rescue needs a check that does not lean on BEATs. The 4 (b) sounds are short: FlexSED runs of "
  "0.08–0.24 s, and two of them reach only 0.42–0.43.")
A("2. **The gate says visible: 3** (e: Bird ×2, Bell). The detectors heard them well (BEATs 0.46–0.86). The gate named a source "
  "on screen (macaws, a small bird, a church bell) where the annotator said it is not visible. No detector change helps these.")
A("3. **Heard at the onset, then lost in stage-4 bookkeeping: 5** (d 2, f 2, h 1). The twin rule keeps the BEATs conf 0.31 of "
  "an Insect span that FlexSED scores 0.93 (Cricket). The display bar is 0.35 but BEATs gives 0.22 (Bird). A long or merged span "
  "starts too early (Crowd, Vehicle). Both vetoes drop a FlexSED 0.92 Laughter. Two of these also make a cross wrong picture "
  "(Cricket at 4.5 s, Crowd at 0.0 s), so one fix there could turn 2 misses into hits and remove 2 wrong pictures.")
A("4. **Unheard: 4** (a: Hammer, Clang, golf Whack ×2): no detector reaches 0.3 / 0.4. **Wrong family: 0** (g). Only 3 misses "
  "have another family's span at the onset, and none of those became a picture, so relabelling does not help the misses.")
A("5. **Wrong pictures: 20 of 24 come from BEATs.** 6 visible sounds pass the gate (thunder ×2, church bell, fire alarm, a vehicle "
  "on a visible air horn, fireworks). 7 of 11 cross pictures show the right family at the wrong time (a ringing phone labelled "
  "Alarm ×2, a second shaver picture, Bird, Cricket, Crowd, Gunshot before a visible machine gun): this is a cost of the onset "
  "rule, not a label error. Right timing would stop 6 of the 7 costing as false alarms, but only 2 would turn into hits "
  "(Cricket, Crowd): Alarm x2 and the shaver would become duplicates, the Bird would be don't-care (importance 1), and the "
  "Gunshot would still be a visible false alarm (the machine gun is on screen). 3 of the 7 phantoms are one BEATs Vehicle label on washing-machine rumble in `b3_laundromat`.")
A("")
A("## 1. The 22 missed needed sounds")
A("")
A("Scores = the best same-family frame in the hit window (the label is in brackets when the score is ≥ 0.1). *Cross-label* = "
  "another family's stage-4 span (salient, conf ≥ 0.175) that starts in the window; *shown* = conf ≥ 0.35.")
A("")
A("| # | clip | gold label | onset s | dur s | imp | BEATs | FlexSED | FlexSED run ≥ 0.4 (s) | PANNs | bucket | also | cross-label at onset | note |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, r in enumerate(miss_rows, 1):
    cl = "; ".join(r["cross_label_at_onset"]) or "—"
    A(f"| {i} | `{r['clip']}` | {r['gold_label']} | {r['onset']:.1f} | {r['duration']:.1f} | {r['importance']} | "
      f"{fmt(r['beats_best_in_window'], r['beats_best_label'])} | {fmt(r['flexsed_best_in_window'], r['flexsed_best_label'])} | "
      f"{r['flexsed_run_ge04_s']:.2f} | {fmt(r['panns_best_in_window'], r['panns_best_label'])} | **({r['bucket']})** "
      f"{r['bucket_detail']} | {', '.join(r['also']) or '—'} | {cl} | {r['note']} |")
A("")
A("## 2. The 24 wrong pictures")
A("")
A("*Detector* = the origin of the stage-4 span that sets the picture's start: a BEATs span (its start may have been pulled "
  "earlier by a FlexSED twin) or a FlexSED-only span. *Score* = that span's conf (BEATs peak or FlexSED peak). BEATs / FlexSED / "
  "PANNs = the best score of the picture's family in [start − 0.5, start + 1.0] s. *Same family, off its onset* = a gold sound "
  "of the picture's family is already on, or starts up to 2 s after the picture.")
A("")
A("| # | clip | shown label | start–end s | type | detector | score | BEATs | FlexSED | PANNs | collided with (gold at that time) | note |")
A("|---|---|---|---|---|---|---|---|---|---|---|---|")
for i, w in enumerate(wrong_rows, 1):
    coll = "; ".join(w["collided_with"]) or "—"
    note = "; ".join(x for x in (w["subtype"], w["note"]) if x) or "—"
    A(f"| {i} | `{w['clip']}` | {w['shown_label']} | {w['start']:.2f}–{w['end']:.2f} | {w['type']} | {w['detector']} | "
      f"{w['score']:.2f} | {fmt(*w['beats_at'])} | {fmt(*w['flexsed_at'])} | {fmt(*w['panns_at'])} | {coll} | {note} |")
A("")
A("## TLDR")
A("- 22 misses: FlexSED band 10 (reachable 6, too short 4), gate said visible 3, stage-4 bookkeeping 5 (below the display bar 2, "
  "onset outside the window 2, veto 1), unheard 4, wrong family 0.")
A("- Only 2 of the 6 reachable band spans survive R1's BEATs self-veto (the explosions); BEATs is deaf to the other 4.")
A("- 24 wrong pictures: 20 BEATs, 4 FlexSED-only; 7 of the 11 cross pictures are the right family at the wrong time; fixing their timing gives only 2 hits.")
(ROOT / "docs" / "dev_miss_table_2026-09-28.md").write_text("\n".join(L) + "\n", encoding="utf-8")
print("md written")
