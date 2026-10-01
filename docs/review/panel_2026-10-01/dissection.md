# Error dissection for the 5-Fable panel (1 Oct 2026, 12:45 UTC)

## The task and the pipeline (short)
Video in → we show small pictures of NON-SPEECH sounds a deaf/hard-of-hearing (DHH) viewer cannot hear AND cannot see the source of
(off-screen or not obvious). Stages: (1) audio extract; (2) scene/video analysis; (3) speech transcript; (4) sound event detection:
BEATs tagger (frame scores, bar 0.175, min span 0.3 s) ∪ FlexSED (text-query SED, bar 0.8; 0.5–0.8 "band" rescued only via
listeners) → vetoes (FlexSED cross-detector veto 0.3, mirror veto 0.7, PANNs veto, DASM clip veto 0.084, masked-weak veto N2,
continuation veto CONT, FineLAP veto 0.329 on rescued spans, K4A-D) → listener rescue TIER (Qwen3-Omni yes/no + Audio Flamingo
Next; DASM vote 0.575) → onset refinement (CAM, BTP pull to FlexSED run); (5) visibility gate: Qwen3.8-27B VLM on frames around
each sound, majority of 3 questions (name / A-B / describe) per stretch, every stretch must be "seen" to silence; (6) picture
generation (Qwen-Image) + display: min dwell 1.5 s, repeat-merge gap 2.5 s, GROUP (Qwen3-Omni "same sound or new event" on
same-family pictures <= 8 s apart), display bar 0.35.

## Scoring
Merged DEV = 71 clips, 58 needed sounds (selection set). Held-out 415 AudioSet-Strong clips = second selection set (audio labels
only, no visibility). TEST (88 clips) is NOT to be used for any choice. Hit = same-family picture starting −0.5…+1.0 s from the
gold onset of a NEEDED sound. Wrong = visible (source on screen), cross (wrong family / wrong time during another sound), phantom
(nothing there). Cost = (4·miss + 2·wrong)/clips (w = 2; secondary w = 1 counts visible wrongs as 1).
**Current base (SHIP8+MD3): 29 hits / 29 misses / 18 wrong (6 visible / 10 cross / 2 phantom), cost 2.141.**

## The 30 misses of SHIP8 (MD3 has since recovered tg_d107 Laughter)
Bucket A = no detector hears it; B = heard but no stage-4 row; C = stage-4 row exists but dropped (gate "seen", below display
bar, merged away); D = drawn but outside the hit window. Columns: BEATs / FlexSED / DASM max in the hit window; Qwen-Omni (Q) and
Audio Flamingo (A) whole-clip lists name the family.

| bucket | clip | sound @onset | BEATs | FlexSED | DASM | Q A | what removed it |
|---|---|---|---|---|---|---|---|
| B | ambient_citywalk_nyc_1689 | Vehicle 3.8 | 0.1 | 0.708 | 0.718 | -Y | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| B | ambient_citywalk_nyc_1689 | Hammer 8.1 | 0.01 | 0.005 | 0.003 | YY | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| B | ambient_citywalk_nyc_1689 | Hammer 13.7 | 0.032 | 0.7 | 0.273 | YY | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| A | ambient_citywalk_nyc_2627 | Clang 3.8 | 0.015 | None | None | -- |  |
| B | ambient_nature_rainforest_2179 | Bird 6.5 | 0.033 | 0.435 | 0.495 | -Y | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| C | ambient_nature_rainforest_7629 | Bird 0.1 | 0.42 | 0.682 | 0.544 | YY | Bird vocalization, bird call, bird song@0.14 conf 0.26: below display threshold 0.35; Bird@0.14 conf 0.26: below display |
| B | as_explosion_XJ8lc3I6 | Footsteps 2.1 | 0.008 | 0.658 | 0.262 | YY | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| B | as_explosion_XJ8lc3I6 | Explosion 2.8 | 0.137 | 0.621 | 0.516 | YY | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| B | as_explosion_XJ8lc3I6 | Gasp 6.7 | 0.415 | 0.758 | 0.065 | -- | above stage-4 bar (BEATs), no row: filtered/vetoed (which veto not recorded) |
| B | b3_carnival_parade | Whistle 6.1 | 0.006 | 0.585 | 0.372 | -- | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| B | b3_favela_rio | Train 14.6 | 0.076 | 0.871 | 0.242 | Y- | above stage-4 bar (BEATs, FlexSED), no row: filtered/vetoed (which veto not recorded) |
| A | b3_golf_course | Whack, thwack 6.5 | 0.006 | None | None | -- |  |
| A | b3_golf_course | Whack, thwack 24.4 | 0.005 | None | None | -- |  |
| C | b3_pet_shop | Bird 0.1 | 0.839 | 0.682 | 0.718 | YY | Bird vocalization, bird call, bird song@0.22 conf 0.86: stage 5 augment=false (source visible on screen (birds) - stay s |
| C | bell_miami | Bell 0.2 | 0.699 | 0.978 | 0.92 | YY | Church bell@0.22 conf 0.70: stage 5 augment=false (source visible on screen (church bell) - stay silent); Bell@0.22 conf |
| D | birds_forest | Bird 1.3 | 0.211 | 0.708 | 0.599 | YY | Bird@2.22 conf 0.22: below display threshold 0.35; Crowing, cock-a-doodle-doo@6.25 conf 0.30: below display threshold 0. |
| C | ly_ambulance_(siren)_-yPSgCn | Vehicle 7.3 | 0.544 | 0.929 | 0.599 | YY | Emergency vehicle@0.00 conf 0.80: gate seen (named 'white car'); Ambulance (siren)@0.00 conf 0.75: gate seen (named 'whi |
| D | ly_applause_62ZYD0u | Crowd 1.9 | 0.005 | 0.886 | 0.474 | -- | Applause@1.12 conf 0.45: placed at 0.00, outside hit window |
| B | mv_storm_scene_house | Siren 16.9 | 0.006 | 0.42 | 0.022 | YY | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| B | tg_d029 | Bird 6.9 | 0.813 | 0.258 | 0.107 | -Y | above stage-4 bar (BEATs), no row: filtered/vetoed (which veto not recorded) |
| B | tg_d032 | Thunder 2.8 | 0.015 | 0.646 | 0.013 | -- | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| C | tg_d032 | Thunder 7.4 | 0.426 | 0.73 | 0.013 | -- | Thunder@8.00 conf 0.27: below display threshold 0.35 |
| B | tg_d033 | Siren 0.0 | 0.025 | 0.59 | 0.257 | -- | below stage-4 bars (BEATs < 0.175, FlexSED < 0.8) |
| B | tg_d095 | Dishes 16.6 | 0.373 | 0.311 | 0.193 | -- | above stage-4 bar (BEATs), no row: filtered/vetoed (which veto not recorded) |
| B | tg_d107 | Laughter 8.2 | 0.252 | 0.886 | 0.887 | YY | above stage-4 bar (BEATs, FlexSED), no row: filtered/vetoed (which veto not recorded) |
| C | tg_d120 | Cat 2.9 | 0.805 | 0.117 | 0.924 | YY | Domestic animals, pets@2.97 conf 0.64: no stage-5 spec holds it (merged/deduped away) |
| A | tg_d125 | Explosion 5.4 | 0.008 | 0.032 | 0.152 | -- |  |
| B | tg_d125 | Clapping 8.5 | 0.017 | 0.792 | 0.338 | -Y | above stage-4 bar (FlexSED), no row: filtered/vetoed (which veto not recorded) |
| C | tg_d133 | Fart 0.0 | 0.908 | 0.869 | 0.612 | YY | Fart@0.22 conf 0.94: stage 5 augment=false (a kind of Fart, whose source is visible - stay silent); Digestive@0.06 conf  |
| C | tg_d133 | Fart 5.6 | 0.863 | 0.86 | 0.578 | YY | Fart@6.30 conf 0.91: stage 5 augment=false (source visible on screen (boxer dog) - stay silent) |

Kill-flag analysis of the 6 strongly heard B misses (`docs/review/kill_flags_ship8.md`): Gasp — min span (0.25 s) + N2 + DASM
clip veto; Train — CONT continuation veto (reverting adds 66 rows); rooster tg_d029 — mirror veto + FlexSED cross veto (Round 52
running: conf >= 0.7 skips both); Dishes — mirror + N2 + K4A-D; Laughter — min span (fixed by MD3); Clapping — FlexSED bar +
min span + PANNs. C-bucket gate kills: pet_shop Bird, bell_miami Bell, tg_d133 Fart ×2 (source judged visible; Adam's ratings:
off-screen/needed), ambulance Vehicle (gate named "white van").

## The 18 wrong pictures (base)
| type | clip | picture (start) | gold sounds near it |
|---|---|---|---|
| visible | ambient_weather_storm_16200 | Thunder (0.06) | Thunder |
| visible | ambient_weather_storm_7200 | Thunder (0.06) | Thunder |
| cross | as_explosion_XJ8lc3I6 | Gunshot (8.25) | Walk, footsteps 2.1-11.3 |
| cross | b3_crossing_bells | Steam (0.22) | Train 0.0-17.0 (seen) |
| cross | b3_golf_course | Bird (18.84) | Bird 0.0-28.0, Walk, footsteps 17.1-19.3 (seen) |
| phantom | b3_laundromat | Train (1.0) | nothing |
| visible | london_protest_01 | Vehicle (0.25) | Air horn, truck horn |
| cross | ly_applause_62ZYD0u | Crowd (0.0) | Laughter 0.0-13.8 (seen) |
| cross | mv_protest_scene_movie | Glass (4.75) | Crowd 0.0-20.0, Baby cry, infant cry 3.2-5.1 (seen) |
| visible | un_driving_motorcycle_DgdHSmwA | Explosion (13.52) | Fireworks |
| phantom | un_hair_dryer_drying_WWu24rJs | Computer keyboard (11.25) | nothing |
| cross | tg_d022 | Dog (7.25) | Chopping (food) 7.0-8.3 (seen) |
| cross | tg_d088 | Explosion (10.75) | Rain 0.0-14.8 (seen), Thunder 11.1-14.7 (seen) |
| cross | tg_d088 | Thunder (13.25) | Rain 0.0-14.8 (seen), Thunder 11.1-14.7 (seen) |
| cross | tg_d107 | Screaming (6.52) | Bird vocalization, bird call, bird song 6.0-8.9 (seen) |
| visible | tg_d127 | Water (0.14) | Water |
| visible | tg_d128 | Laughter (3.08) | Laughter |
| cross | tg_d128 | Hammer (9.0) | Clang 5.6-10.0 (seen) |

Adam's blind re-rating (Round 46, 20 random SHIP8 wrongs): ~3/20 are real, wanted, unlisted off-screen sounds; 5/20 are real
sounds the gold times elsewhere; 9/20 not audible. Gold lists only salient sounds and splits a sound at pauses > 2 s.

## Grouping / the limit question (Adam, 12:35: "the limit is useful — what should we use after 3 s? DEV 3–8 gives the same")
GROUP asks Qwen3-Omni on the stretch between two same-family pictures (picture gap = next start − previous picture end, which
includes dwell/after-end stretching) "same continuing sound or a new event?" (both orders). DEV pairs: 2.07 new/new (Explosion),
2.75 same (shaver, was a cross wrong → merged = −1 wrong), 2.96 same (alarm, −1 wrong), 3.14 & 4.5 new/same (Glass, kept),
6.75 Gunshot. Any limit 3–8 s gives the same DEV score. DEV gold: NEEDED same-family re-onsets occur at real pauses 0.6, 1.4,
2.1, 2.4, 3.0, 3.2, 4.1, 4.5, 5.9 s. The annotation rule says a new row starts after a pause > 2 s. Round 51 (running): add a
pause question "does it stop completely > 2 s?" calibrated on the held-out 415 (true pause lengths), merge iff same/same and
no-long-pause. Omni says "same" for the same SOURCE even when the gold wants a new picture ("the sound came back").

## Closed rounds (pre-registered, failed on DEV or the held-out set) — do not re-propose without a new angle
Detectors: EAT, Dasheng, DASM as main detector, PANNs, SAM-Audio separation, FlexSED extra queries (Round 14 D: Whack/Clang still
unheard), BEATs self-veto, lower FlexSED bar (R1 +3 hits +7 wrong), short-sound path R13-5, BANDLIST (band run + Omni list + DASM:
0.69 precise held-out, DEV 0 hits +6 cross), BOX/BOX-2 arm, kill-flag reverts (each restore costs 30+ rows except MD3).
Listeners: Qwen3-Omni yes/no/MC/paired-cut/localisation/open-list rescue (~3 wrong per hit), EXPECT (Omni whole-clip list →
FineLAP onset → DASM: DEV +3 hits/+5 wrong, extras mostly real sounds with visible maker), AGREE-EARS, NAMED-VETO, PRIOR415.
Gate/visibility: Gemma-4-31B gate, Qwen2.5-VL, OWLv2, SAM 3 grounding, BOX-2 crop, SYNC/SYNC-2 (Synchformer AV sync), CF
counterfactual, HUMAN/HUMAN-2 ("is the source acting at the onset", majority of stretches), HUMAN-BOX (named source → box → crop
"making the sound now?": crops say no on visible birds/waves), AVNAME, MAKER-VIS, CONTRAST, onset motion OM. All move along one
trade-off line: silence more seen sounds ⇄ lose needed ones (the look-alike rule: "a bird is on screen but a different one sings").
Wrong-type: RELABEL-GATE (listener relabel then gate), DETACHED-ADD, RPT-S/DBR repeat rules, ONSET_RELOC, RETRIGGER.

## Constraints
General methods only (no per-clip rules); choices on DEV or held-out 415 only; pre-register before numbers; models available
offline on the cluster (H200/A100): Qwen3-Omni-30B, Audio Flamingo Next, Qwen3.8-27B VLM, Gemma-4-31B, BEATs, FlexSED, DASM, EAT,
CED, SSLAM, AST, CLAP, FineLAP, OWLv2, SigLIP, CLIP, Kimi-Audio, Step-Audio-2-mini, Qwen-Image. Home disk ~25 GB free (no big
downloads). Time: hours, not days.
