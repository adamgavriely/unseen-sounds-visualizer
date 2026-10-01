# Trail hook sites (inventory of every decision point in the shipped pipeline, D′)

Where `src/trail.py: decide(...)` calls go, what each step decides, and what must be logged. Written 2 Oct 2026 from main; line numbers drift, so re-check them at the freeze. Hooks are added only after the detector is frozen, and parity must reproduce the shipped scores exactly.


Repo state: `/home/claude/MscFinalProject` at `f36bf7d` (origin/main). No repo file was edited.

Note: the config file is `/home/claude/MscFinalProject/config.py` (repo root), not `src/config.py`. `use_shipped()` is
at config.py:502. Arm `SHIP8+MD3+WW5+SL` is defined at benchmark/gold/round13_dev.py:242 (built from BASE line 44, `_TO1`,
`_F7`, `_F8`, then SHIP..SHIP8, +MD3, +WW, +WW5, +SL).


| area | flag = shipped value |
|---|---|
| detector | AED_MODEL beats; AED_THRESHOLD 0.175; AED_HYSTERESIS 1.0 (low = 0.175); AED_MIN_DUR **0.3**; ONSET_CAM True; ONSET_MONOTONE True; UNION_START "min"; UNION_WEAK_TWIN "absorb"; MERGE_START "earliest"; MAX_SPAN None; LABEL_FILTER "depictable" |
| FlexSED / vetoes | FLEXSED_BAR 0.8 (low 0.8); FLEXSED_VETO 0.3; PANNS_VETO 0.05; TWIN_MAX True; MIRROR_VETO 0.7, MIRROR_OWN_MAX 0.4; LISTENER_CONFIRMED_MIRROR True; MASKED_WEAK_VETO True + MASKED_WEAK_AF True (NEED_MASK True); DASM_CLIP_VETO 0.08392; DASM_LOCAL_VETO 0.35, DASM_LOCAL_KEEP "both", DASM_LOCAL_SCENE = WORK_DIR/scene_videos.json, SCENE_FIT_LOGIT True; KEEP_NEEDS_V4 True; KEEP_NEEDS_V4_ALL "onto" + KEEP_NEEDS_V4_ALL_DASM_KEEP True; BAND_TWIN_PULL 0.5; CONTINUATION_VETO 0.5 |
| rescue | LISTENER_RESCUE True, LISTENER_RULE "TIER", TIER_SPLIT 0.6, LISTENER_LO 0.5, LISTENER_ONCE True; LISTENER_DASM_VOTE True (bar 0.575, pad 0.5); FINELAP_VETO 0.329; DASM_RESCUE + DASM_RESCUE_NEW_ONLY True; LISTENER_REQUIRE_CACHES True; caches from `set_listener_split(name)` (config.py:586) |
| stage 5 | VIDEO_BACKEND owlv2 (stage-2 concept list); VLM_MODEL Qwen/Qwen3.8-27B, thinking off, greedy; GATE_ENABLED, VLM_VISIBILITY, DEPICTION_REASONING True; VISIBILITY_STRETCH 5.0; VISIBILITY_RULE "majority"; KINSHIP_DIRECTED True; DISPLAY_THRESHOLD = AUGMENT_THRESHOLD 0.35; SPEECH_CONTEXT True (Whisper "base"); DISAMBIGUATE True; DEDUP_SIM 0.80 (report 0.45); PICTURE_V3 + PICTURE_SCENE + PICTURE_SCENE_GUARD2 + PICTURE_MAKER True |
| stage 6 | GEN Qwen/Qwen-Image-2512 1024²; PICTURE_FINAL True; PICTURE_VERIFY True, 5 tries; DEPICT_EVENT True (cache WORK_DIR/depict_answers.json - the DEPICT_CACHE line in use_shipped is inside a comment, so the default path is used); MIN_DWELL 1.5; MERGE_GAP 2.5; MAX_AFTER_END 1.0; GROUP_ASK True, GROUP_MAX_GAP 8.0, GROUP_CACHE WORK_DIR/group_answers.json; MAX_SLOTS 3; CONFIDENCE_FADE False; PICTURE_MIN_CONF None |

Execution order (src/pipeline.py): stage 4 `detect_events` (pipeline.py:83) -> events.json -> stage 5 `plan_augmentations`
(pipeline.py:91) -> trace "family"/"family_spans" -> augmentations.json -> `reason.decide_subjects` (pipeline.py:111) ->
gate_votes.json -> `generate_augmentations` (pipeline.py:117) -> augmentations.json again -> GROUP subprocess -> DEPICT
subprocess -> trace "display" -> onset_trace.json (pipeline.py:138) -> `composite_alongside` (MAX_SLOTS).

Inside stage 4 the order is: extract BEATs -> FlexSED extract -> twin union -> mirror veto (+F7) -> N2 masked-weak ->
DASM clip veto -> DASM local veto (+scene margin) -> K4A -> BTP -> CONT -> FlexSED cross veto -> PANNs veto (+listener b)
-> listener band rescue (a) [end of fuse_flexsed] -> DR2 DASM rescue -> trace "veto" -> onset refinement -> trace
"refine" -> filter_rescued (FLAP, F8, ONCE). Consequence: rescued spans (band (a), DR2) never face the DASM, K4A, BTP,
CONT, FlexSED or PANNs steps (the `rescued` skips inside those steps are no-ops in the pipeline order); a listener-(b)
kept span faced them all as an ordinary FlexSED-only span first.

File paths below are relative to `/home/claude/MscFinalProject`. S4 = `src/stage4_audio_event_detection/__init__.py`,
R = `src/stage5_cross_modal_analysis/reason.py`, S5 = `src/stage5_cross_modal_analysis/__init__.py`,
S6 = `src/stage6_visual_augmentation/__init__.py`.

"Recorded today" uses: **TRACE** = onset_trace.json row via trace(); **STDOUT** = print only (lost unless the job log is
kept); **MEM** = an in-memory structure never written (LISTENER_STATS, `_dropped`, MAKER_LOG); **NOTHING**.

---

## Stage 4 - detection, vetoes, rescues (all in S4 unless noted)

| # | id | where | what it does (plain) | exact condition (value vs config bar) | model / question / answer rule | outcome | recorded today | minimal log needed |
|---|---|---|---|---|---|---|---|---|
| 4.0 | `require_caches` | S4:290-317 (called from `_pipeline_listener` S4:324) | Stops the clip if any per-clip listener/DASM/FineLAP input is missing. | missing LISTENER_CACHE / VCACHE / AFCACHE items for clip, `dasm_<split>/<clip>.npz`, RELABEL_P1V4, DASM_P4_CACHE, `finelap_<split>/<clip>.npz` | - | stop (RuntimeError) | exception text | (fine as is) |
| 4.1 | `beats_extract` | S4:73-124, call S4:223 | Turns BEATs window scores into spans; a class must peak ≥ bar, span = contiguous stretch ≥ low, too-short spans dropped. | class peak ≥ AED_THRESHOLD 0.175; frames ≥ low 0.175 (=0.175×AED_HYSTERESIS 1.0); span length ≥ AED_MIN_DUR 0.3 s; conf = span peak | BEATs iter3+ AS2M, 2-s window / 0.25-s hop, stamp offset 0.5 | keep / drop (short or never ≥ 0.175) | TRACE `extract` (survivors only, note "BEATs") | dropped short spans: label, start, end, peak, length vs 0.3 |
| 4.2 | `flexsed_extract` | S4:807 | Same span cutter on FlexSED's 215 family queries. | peak ≥ FLEXSED_BAR 0.8; low 0.8; length ≥ 0.3 s | FlexSED (cached frames, 25 fps) | keep / drop | TRACE `flexsed_raw` (survivors) | as 4.1 |
| 4.3 | `twin_union` | S4:842-873 | A FlexSED span with a same-family BEATs span within 1 s is absorbed into it: the BEATs span takes the earlier start and (TWIN_MAX) the stronger bar-normalised confidence; un-twinned FlexSED spans become FlexSED-only ("fresh"). | same canonical family and `b.start-1 ≤ f.end` and `f.start-1 ≤ b.end`; start = min(b.start, f.start) (UNION_START "min"); conf = min(1, 0.35·max(b/0.35, f/0.8)); UNION_WEAK_TWIN "absorb" (a sub-0.35 BEATs twin also absorbs) | - | merge + move start earlier + raise conf; FlexSED span consumed | TRACE `union` (survivors after union); absorbed FlexSED span and old start/conf: NOTHING | per absorbed pair: FlexSED label/start/peak, BEATs label/start/conf before, start and conf after |
| 4.4 | `mirror_veto` | S4:905-908, `_mirror_veto` S4:533-559 | Drops a BEATs-only span when FlexSED at that moment clearly hears a different family and not this one. | only spans not FlexSED-only and not twinned; family has a FlexSED query; in [start,end) FlexSED top query is another family with score ≥ MIRROR_VETO 0.7 AND own-family max < MIRROR_OWN_MAX 0.4 (STRONG_BEATS_KEEP None) | FlexSED | drop (unless 4.5) | TRACE `mirror_veto` (dropped rows, note "R13-2 dropped (b 0.7)") + STDOUT count; top family, its score, own score: NOTHING | top family + score, own-family max, bars 0.7 / 0.4 |
| 4.5 | `mirror_keep_F7` | S4:911-921, `listener_p1_lookup` S4:332-363, `_v4_names` S4:1452-1465 | A mirror-vetoed span comes back when the listener accepts it on its P1 cut AND (K-V4) an open-inventory listener names its family. | P1 item matched by family, end within 0.02 s, start in [start-0.02, end]; accept = item.accept V4 if present, else V12 (P1 items only carry V1/V2/V12, so in practice **V12**); not in vcache -> dev_listener yes/no score > 3.0; not found -> "missing" = reject. Then K-V4: Qwen V4 family list (`qwen_fams` in RELABEL_P1V4) contains the family OR AF V4 accept on the P1 item. | Qwen3-Omni-30B-A3B. V1 MC: "Listen carefully. Which ONE of these is actually present in this recording? A) {} B) {} C) {} D) {} E) none of A-D. Answer with a single letter." (X at A and at D; accept p(X) > 0.5 and > 2× max other). V2: "Is the sound of {family} present in this recording? Answer yes or no." logit(yes)-logit(no) on run cut vs control cut; accept s_run > 3 and s_run − s_ctrl > 2 (no control window -> reject). V12 = V1 and V2. V4 (Qwen and AF Next): "List every distinct non-speech sound you hear in this recording, one per line, most prominent first." greedy 64 tok; a line matches family by name/synonym/child/parent word, else mpnet cosine > 0.6 | rescue (undo drop) | MEM `LISTENER_STATS["f7"]` = [label, start, end, ok, how]; only the final drop list reaches TRACE | per span: how (V12/yesno/missing), V1 p(X), V2 s_run/s_ctrl, Qwen V4 text/qwen_fams, AF V4 text/af_fams |
| 4.6 | `masked_weak_N2` | S4:927-978 | A weak BEATs-only span under speech or music is dropped unless a listener confirms it. | not FlexSED-only, not twinned, conf < 0.5, and BEATs "Speech" or "Music" ≥ 0.3 in the span (MASKED_WEAK_NEED_MASK True); kept if F7 rule (4.5 lookup, incl. K-V4) accepts, or AF V4 accepts on the P1 cut (`_af_p1_accepts`, S4:1572) | as 4.5 (V12 / yes-no>3, + K-V4) and AF Next V4 | drop / keep | TRACE `masked_weak_veto` (dropped, note "N2 dropped") + STDOUT count | speech/music max, conf, Qwen how/answer, AF V4 answer |
| 4.7 | `dasm_clip_veto` | S4:985-1005 | Drops a span whose family DASM never hears anywhere in the clip, unless a listener keeps it. | family clip-max of DASM `fw` < DASM_CLIP_VETO 0.08392 (family without a DASM query -> 1.0 -> pass); keep if listener_p1 accepts (V12 / yes-no>3, NO K-V4 here) or AF V4 accepts on P1 | DASM (frames in `dasm_<split>/<clip>.npz`) + listeners as 4.5 | drop / keep | STDOUT count only | DASM clip max vs 0.0839, Qwen how, AF V4 |
| 4.8 | `dasm_local_veto` (WEAK-WITNESS) | S4:1006-1036 | Drops a span DASM does not hear around it, unless both open-inventory listeners name it, or one does and the VLM finds it plausible in the scene. | DASM family max in [start−0.5, end+0.5] < DASM_LOCAL_VETO 0.35 (no DASM column or no frames -> pass); DASM_LOCAL_KEEP "both": keep iff Qwen V4 names it (`_v4_names_qwen` S4:1534, `qwen_fams` contains family) AND AF V4 accepts on P1 (`_af_p1_accepts`); if exactly one does -> 4.9 | DASM + Qwen3-Omni V4 + AF Next V4 (prompt as 4.5) | drop / keep | STDOUT count only | DASM max vs 0.35, Qwen V4 yes/no + text, AF V4 yes/no + text, scene verdict |
| 4.9 | `scene_margin` (SCENE-MARGIN, logit) | S4:1027, `_scene_margin` S4:1488-1531, `R._scene_fit` R:1386-1422, `_yes_no_margin` R:1364-1383 | When only one listener names the sound, the gate VLM is asked if the sound is plausible in this scene; yes keeps it. | per stretch (span cut into max(1, round(len/5)) pieces), 6 frames from a−1 to b+1; d = margin(Q) − margin(twin) where margin = max logit("yes") − max logit("no") at first answer token; stretch yes if d > 0, no if d < 0; verdict = yes > no; no frames/video -> None = not kept | Qwen/Qwen3.8-27B (thinking off). Q: "Could the sound of {label} plausibly be heard in this scene? Answer yes or no." twin: "Could the sound of {label} NOT plausibly be heard in this scene? Answer yes or no." (label = canonical family, first part, lower-case) | keep (rescue from 4.8) | `<WORK_DIR>/scene_videos.json.logit_answers.jsonl`: {key:[clip,family,start,end], label, verdict, answers:[{stretch, answer, d}], video} + STDOUT "scene margin key: verdict" | (already logged; link it to the span's drop record) |
| 4.10 | `k4a_inventory` (K4A-D) | S4:1037-1058, `_p1v4_lists` S4:1468 | Drops a displayable span that neither listener's open list names (or names a kind of), unless DASM hears it. | conf ≥ 0.35, both `qwen_fams` and `af_fams` present for its P1 item (else kept); drop if family ∉ names and no name is an ontology descendant of family ("onto"); keep if DASM ≥ LISTENER_DASM_BAR 0.575 in [start−0.5, end+0.5] (`_dasm_keeps` S4:1551) | Qwen3-Omni V4 + AF Next V4 lists (RELABEL_P1V4 cache) + DASM | drop / keep | STDOUT count only | qwen_fams, af_fams (and texts), DASM max vs 0.575 |
| 4.11 | `band_twin_pull` (BTP) | S4:1059-1077 | Moves a span's start earlier to a FlexSED run of its family that ends just before it. | FlexSED family run (frames ≥ 0.5, gaps ≤ 0.24 s merged) with 0 ≤ start − run_end ≤ 1.0 and 0 ≤ start − run_start ≤ 1.5; latest such run start wins | FlexSED | move start (earlier) | STDOUT count; visible only as start difference between TRACE `union` and `veto` rows | old start, new start, run [rs, re], run peak |
| 4.12 | `continuation_veto` (CONT) | S4:1132-1151 | Drops a span that is a later piece of a sound FlexSED already hears going on. | span start ≥ 1.5 s; a FlexSED family run (≥ 0.5, gaps ≤ 0.24 s) starts ≤ start − 1.5 and reaches ≥ start | FlexSED | drop | STDOUT count only | run start/end/peak that covers it |
| 4.13 | `flexsed_cross_veto` | S4:1152-1160 | Drops a span whose family FlexSED never hears in the whole clip. | FlexSED clip-max of the family < FLEXSED_VETO 0.3 (family without a FlexSED query -> pass; STRONG_BEATS_KEEP None) | FlexSED | drop | STDOUT count only | family clip max vs 0.3 |
| 4.14 | `panns_clip_veto` | S4:1183-1222 | Drops a FlexSED-only span whose family PANNs never hears in the clip, unless the listener keeps it (4.15). | span is FlexSED-only (`fresh`) and PANNs CNN14 family clip-max < PANNS_VETO 0.05 | PANNs CNN14 (computed live once per clip) | drop | STDOUT count; TRACE only implicitly (in `flexsed_raw`/`union`, absent from `veto`) | PANNs clip max vs 0.05 |
| 4.15 | `listener_keep_b` | S4:1198-1213, `listener_from_vcache` S4:1328-1394, `_tier` S4:1302-1307 | A PANNs-vetoed FlexSED span is kept (marked rescued) if the listeners accept its family. | PV item with same family, start and end within 0.02 s; TIER: item peak ≥ TIER_SPLIT 0.6 -> Qwen V4; peak < 0.6 -> Qwen V4 AND AF V4 (AF missing -> no); item missing -> drop | Qwen3-Omni V4 + AF Next V4 (prompt as 4.5) on the PV item's run cut | rescue | MEM LISTENER_STATS b_kept/b_missing_list; STDOUT summary counts; events.json `rescued: true` | PV peak, Qwen V4 (text, matched), AF V4 (text, matched), rule branch |
| 4.16 | `listener_band_rescue_a` | S4:1248-1253, `_listener_band` S4:1813-1862 | Adds a sound FlexSED heard below its bar if the listeners confirm it. | FlexSED family run (frames ≥ 0.4, gaps ≤ 0.24 s) with peak in [LISTENER_LO 0.5, 0.8), no same-family span overlapping it; P2 item matched (family, start, end ±0.02); accepted by TIER (as 4.15); span = run, or the listener's 1-s cut when run < 0.5 s; conf = run peak | Qwen3-Omni V4 + AF Next V4 | rescue (add) | MEM LISTENER_STATS a_added / a_missing_list; STDOUT counts; appears in TRACE `veto`; events.json `rescued: true` | run, peak, TIER branch, Qwen V4 text/match, AF V4 text/match; and the refused ones |
| 4.17 | `dasm_rescue_DR2` | S4:248-250, `dasm_rescue_events` S4:1430-1446 | Adds a sound only DASM found when both audio LLMs name it and nothing else found that family in the clip. | P4 item of clip with `qwen_v4` True AND AF `accept.V4` True; family not among current events' families (DASM_RESCUE_NEW_ONLY); conf = DASM peak; rescued, agree | DASM + Qwen3-Omni V4 + AF Next V4 (DASM_P4_CACHE) | rescue (add) | STDOUT list; TRACE `veto`; events.json `rescued`/`agree` | P4 run, DASM peak, both texts |
| 4.18 | `onset_refine` | S4:254-257, `_refine_onsets_cam` S4:156-192; `beats_infer.occlusion_onset` | Sharpens a BEATs span's start by silencing the window's opening; only later, never earlier (monotone). | skipped for FlexSED-only / rescued / DR spans; window = start + 0.5 − 2.0; 26 cuts of 0.08 s; no change if logit drop < 1.0 or evidence gone at first cut; onset = first cut removing 10 % of evidence, clamped to [start, end] | BEATs occlusion | move start (later) | TRACE `refine` (all events after) | old start, new start, evidence drop |
| 4.19 | `finelap_veto` (FLAP) | S4:266-267, `filter_rescued` S4:1679-1695 | Drops a rescued span FineLAP does not support. | rescued and family ∈ FineLAP labels; max FineLAP score over segments overlapping (fe > start, fs < end) < FINELAP_VETO 0.329 | FineLAP (`finelap_<split>/<clip>.npz`) | drop | NOTHING (`_dropped["FLAP"]` returned and discarded at S4:267; no print) | FineLAP max vs 0.329 |
| 4.20 | `dasm_vote_F8` | S4:1753-1779 | A rescued span is kept only if DASM also hears its family around it. | rescued; DASM family max in [start−0.5, end+0.5] ≥ LISTENER_DASM_BAR 0.575 (no DASM file -> kept, but 4.0 forbids that) | DASM | drop / keep | NOTHING (`_dropped["F8"]` discarded) | DASM max vs 0.575 |
| 4.21 | `rescue_once` (ONCE) | S4:1799-1809 | Keeps only the earliest rescued span per family per clip. | among surviving rescued spans sorted by start, a later one of the same family is dropped (ONCE_GAP None) | - | drop | NOTHING (`_dropped["ONCE"]` discarded) | which earlier span it lost to |
| 4.22 | `events_out` | pipeline.py:87 | Final stage-4 spans. | - | - | - | events.json (AudioEvent dicts incl. rescued/agree) | - |

The final TRACE row of stage 4 is `refine`; FLAP/F8/ONCE drops happen after it, so a span can be in `refine` and missing
from events.json with no trace of why.

## Stage 5 - planning and the visibility gate

| # | id | where | what it does | exact condition | model / question / answer rule | outcome | recorded today | minimal log needed |
|---|---|---|---|---|---|---|---|---|
| 5.1 | `label_filter` | S5:47; `labels.is_salient_nonspeech` src/labels.py:97-130 | Labels that are not drawable (speech, music, textures, environment branch, generic names, wind) are discarded. | LABEL_FILTER "depictable" rules (texture/wind on raw name; speech, music/singing, ENV branch, Silence, Sound effect, Human voice, Respiratory sounds, GENERIC, Source-ambiguous on the canonical name) | - | drop | NOTHING (inferable: in events.json, no spec) | label, which rule |
| 5.2 | `family_consolidate` | S5:53-61; `labels.consolidate_families` labels.py:570-619, `merge_by_label` labels.py:335-387 | Relabels each firing to its canonical family; firings of a family within 1 s join one burst; spec start/end = strongest burst, spans = all bursts; most confident child -> `detail`; PICTURE_V3 `source` via `choose_source`. Firings below the bar of a family that has a strong firing are thrown away. | strong set: conf ≥ 0.35; marginal set: 0.175 ≤ conf < 0.35 AND family has no strong firing (consolidated separately); burst join gap 1.0 s; spec `rescued` = all firings rescued | - | merge + relabel (family) + drop weak firings | TRACE `family` (spec label/start/end) + `family_spans` (note = detail); dropped sub-bar firings: NOTHING | raw firings -> spec mapping, discarded firings |
| 5.3 | `display_bar` | S5:73-95 | A family below the display bar is planned but not shown. | conf < min_confidence(label, 0.35) (NOISE_LIKE floors are all ≤ 0.35, so 0.35); augment also needs conf ≥ AUGMENT_THRESHOLD 0.35; stage-2 (OWLv2) "visible" only adds a note (deferred to the VLM when salient) | stage 2 OWLv2 concept list (scene.json) | drop (marginal) | augmentations.json `reason` "below display threshold (0.30 < 0.35)[; source visible on screen anyway]" or "stage 2 saw the source somewhere in the clip; VLM to confirm" | ok |
| 5.4 | `speech_rescue` | R:1586-1610, `_talked_about` R:1433-1457 | A marginal sound people talk about is shown; a shown one people talk about is marked (row priority). | speech within ±3 s (SPEECH_WINDOW) has ≥ 4 words; marginal only if conf ≥ 0.175 and reason has no "visible"; both orderings must pick "reacting" | Qwen3.8-27B text only: "A sound of {label} was heard in a video. Around that moment, someone said:\n\"{speech}\"\nAre they (a) {opt_a}, or (b) {opt_b}? Answer with the letter only." opts "reacting to that sound or talking about it" / "not referring to that sound", asked in both orders, yes iff both pick it | rescue / mark | augmentations.json reason "marginal (0.xx) but people are reacting to it -> augment" or "… \| people are reacting to it", `talked_about`; speech text and votes: STDOUT | speech text, votes per order |
| 5.5 | `visibility_gate` | R:1629-1702; `_sound_is_visible` R:554-627; `_ab` R:520-541 | For each burst, cut into ≤5-s stretches, the VLM decides if the source is seen; a sound is silenced only if every stretch is seen. | stretches: k = max(1, round(len/5)) per burst; 6 frames from a−1 to b+1; per stretch votes name/ab/desc; seen = (#True > #False) (VISIBILITY_RULE majority; an a/b split counts neither); sound silenced iff no stretch is unseen; partly seen -> shown throughout | Qwen3.8-27B on frames. **name**: VISIBLE_PROMPT "These frames are from the moment a sound of {label} was heard.\nName the thing in these frames that is making that sound. Answer with a short noun phrase of at most 5 words. If nothing that could make that sound is visible in these frames, answer exactly: nothing." -> "nothing/none/no/not…" = False; phrase sharing a word with label = True; else MAKES_SOUND_PROMPT "Sound heard: {label}\nThing visible in the video: {named}\nCould that thing be what is making that sound? Answer yes or no." (reply starts "y"). **ab**: "These frames are from the moment a sound of {label} was heard. Judge from the frames alone." (a) "you can SEE {label} happening on screen -- the source is in the frames and visibly making that sound" (b) "{label} is not visibly happening in these frames", "Answer with the letter only.", both orders: both pick visible = True, neither = False, else None. **desc**: "Describe what is happening in these frames in one sentence. Name what you see and what it is doing. No more than 25 words." -> True if it names the label (or names `named` when name=True), else MAKES_SOUND_PROMPT with the description | drop (silence) | gate_votes.json one row per stretch: label, start, end, confidence, stretch, seen, name, ab, desc, named; augmentations.json reason "source visible on screen (named) - stay silent"; desc text, MAKES_SOUND replies, raw a/b letters: STDOUT only (desc first 60 chars) | per stretch: named + makes-sound reply, a/b letters both orders, description + makes-sound reply |
| 5.6 | `family_rule` (kinship, directed) | R:1703-1737 | A sound is silenced when a related sound overlapping it in time was silenced as visible. | g silenced with "visible" in reason (not a "below display threshold" one); KINSHIP_DIRECTED: g.label == s.label or g.label is a descendant of s.label (visible specific silences general, never the reverse); `same_source(s, g)` and bursts overlap within 1 s | - | drop (silence) | augmentations.json reason "a kind of {g}, whose source is visible - stay silent" | which g, its overlapping stretch, its votes |
| 5.7 | `disambiguate` | R:807-848 (call R:1818) | Two live labels on one acoustic event (starts and ends within 0.6 s, not same family): the frames pick one. | |Δstart| ≤ 0.6 and |Δend| ≤ 0.6; 4 frames from a.start−0.4; winner only if both orders pick the same spec | Qwen3.8-27B: "A sound detector heard ONE sound at this moment and could not decide between two labels. Look at the frames from that moment.\nWhich is it? (a) {a}  (b) {b}  (c) cannot tell from the frames\nAnswer with the letter only." both orders | drop the loser / keep both | augmentations.json reason "same sound as X; the frames say it is that one"; replies STDOUT | both replies |
| 5.8 | `depiction_v31` (subject) | R:1828-1847; `_depict_v31` R:971-1021; `_scene_thing` R:936-968; `with_maker` R:1119-1159 | Decides WHAT to draw (never drops): RESOLVE qualifier, V3.1 phrase, guards fall back to the head word, maker rule may replace the subject. | RESOLVE answer refused if it does not end in the heard head word, adds > 2 words, names a forbidden kind or a person, or (GUARD2) an other-branch maker; phrase person guard / forbidden-kind guard / restate / head-word-lost -> head word | Qwen3.8-27B: RESOLVE_PROMPT, DEPICT_PROMPT_V31, MAKER_PROMPT (text in R:901-932, R:1064-1072) | relabel subject | augmentations.json `subject`, `source`, reason "… \| depiction (v3, source S): phrase"; guard refusals: STDOUT; maker changes: MEM MAKER_LOG | RESOLVE answer + refusal reason, first phrase + guard fired, maker old->new + how |
| 5.9 | `dedup` | R:660-777 (call R:1905) | Merges sounds that would be the same picture; a family keeps only its bursts outside a member. | ordered by conf; same_source + depiction SigLIP sim ≥ DEDUP_SIM 0.80 -> merge "same sound under two names"; else subtract member bursts (pieces < 1 s dropped); nothing left -> merge "fully explained by X"; identical subject -> merge; sim ≥ 0.80 and overlap within 1 s -> merge | SigLIP text encoder (google/siglip-base-patch16-224) | merge (loser off, survivor gets both spans) / trim spans | augmentations.json reason "same picture as X (why)", survivor `spans`; trims and near-duplicate sims: STDOUT | sim value, removed bursts |

Planned-but-inactive stage-5 steps (corroboration band, plausibility, F3/N1 scene fit, arbiter, I2, concealed action, box
check, FIX_GATE, non-v3 "no depiction reads as this sound") are listed at the end.

## Stage 6 - pictures and display

| # | id | where | what it does | exact condition | model / question / answer rule | outcome | recorded today | minimal log needed |
|---|---|---|---|---|---|---|---|---|
| 6.1 | `card_or_template` | S6:452-461 (`_final_picture`) | A sound with no drawable maker gets a word card; a known source uses a fixed template subject. | `CARDS.get(source or label)` / `TEMPLATES` (benchmark/gold/gen_screen.py) | - | relabel (card / template) | augmentations.json `image_prompt` "CARD:word", `backend` "card" | ok |
| 6.2 | `picture_verify` | S6:476-543; src/stage6_visual_augmentation/verify.py:312-321 | Each picture is checked; up to 5 draws, then a word card. | ok = MC pick == intended AND OCR finds no word (≥ 3 letters, conf ≥ 0.5, ≥ 2 % tall) | Qwen3.8-27B: "Look at this picture. What is the main thing shown in it? Choose exactly one option.\n{options}\nAnswer with the letter only." (shuffled intended + look-alikes + "something else"); EasyOCR; log-only "In at most eight words, what is the main thing shown in this picture?" | redraw / replace by word card | picture_verify.json (per try: seed, prompt, ok, picked, intended, options, text, saw, learned; final); augmentations.json "VCARD:word" | ok |
| 6.3 | `depict_event` (DEPICT-EVENT) | filter S6:670-672 -> depict.py:50-54; answers depict.py:57-94 | Drops a picture whose depicted event is visibly happening and whose gate-named maker could sound like it. | event = [E1, E2] == ["yes","no"] on 6 frames from start−1 to start+1; then for any gate `named` maker B (≠ "nothing") of that spec: L1 and L2 both "yes"; silenced = event and lookalike; key "label\|spec.start" | Qwen3.8-27B: E1 'Is "{event}" visibly happening in these frames? Answer yes or no.' E2 'Do these frames show no "{event}" happening? Answer yes or no.' L1 "Could the sound of {B} be mistaken for the sound of {A}? Answer yes or no." L2 "Could the sound of {A} be mistaken for the sound of {B}? Answer yes or no." (first word yes/no) | drop picture | depict_answers.json {clip: {"label\|start": {subject, makers, event, lookalike_yes, silenced, event_replies}}}; L1/L2 replies NOT stored; drop visible only as absence in TRACE `display` | L1/L2 replies per maker |
| 6.4 | `display_join` | S6:686-707 | Two bursts of the same label closer than the gap become one appearance. | spec augment and image_path; per label, bursts sorted; join if next start − real end of the chain ≤ gap = max(MERGE_GAP 2.5, min(MIN_DWELL 1.5, MAX_AFTER_END 1.0)) = 2.5 s | - | merge | TRACE `display` (final windows only) | joined bursts and gaps |
| 6.5 | `display_dwell_after` | S6:695, S6:709-715 | Every appearance lasts ≥ 1.5 s but at most 1.0 s past the sound's real end. | end = max(end, start + 1.5) then min(end, max(real_end, start) + 1.0), clipped to duration | - | move end | TRACE `display` | raw end vs shown end |
| 6.6 | `group_same_new` (GROUP) | S6:716-718 -> group.py:69-80; answers group.py:83-96 | Two close same-family pictures are merged when Omni hears the second as the same continuing sound. | consecutive same-family pictures (label equal, same canonical or ancestor/descendant), 0 < next.start − this.end ≤ GROUP_MAX_GAP 8.0; audio cut from this.end − 1.0 to next.start + 1.5; merge iff both answers start with "same"; no chaining | Qwen3-Omni-30B-A3B thinker, greedy 4 tokens: "You hear a {lab} sound near the start and again near the end of this recording. Is the {lab} near the end {o1}, or {o2}? Answer with exactly one word: {w1} or {w2}." o = "the same continuing sound as at the start (for example one alarm or engine with a short pause)" / "a new, separate event (for example a second bark, knock or shot)"; both orders | merge | group_answers.json {clip: {"label\|picture_start": [ans1, ans2]}}; result in TRACE `display` | ok (link key to pair) |
| 6.7 | `max_slots` | S6:725-772, limit at S6:743 (`_assign_rows`, at render) | More than 3 pictures at once: keep by (talked_about first, then confidence). | peak simultaneous pictures > MAX_SLOTS 3 | - | drop picture | NOTHING (not in TRACE `display`, which is computed before row assignment) | dropped picture + rank |

## Flags that exist but are OFF in the shipped config (one line each)

- ONSET_RELOC (I1 relocate onsets) off; RETRIGGER / RETRIGGER_RAW (breaks) off; PERC_RETURN (RET rows) off; TAG_ENS off.
- TWIN_SHORT off; IMPULSE_MIN_SPAN off; FLEXSED_CORROB off; BEATS_LOWBAND_CORROB off; BEATS_SELF_VETO 0; FLEXSED_FAMILY_BARS off; FLEXSED_EXTRA off.
- STRONG_BEATS_KEEP None (no strong-span exemption from mirror / FlexSED veto); PANNS_VETO_SKIP_ABOVE None.
- TIER_SPECIFIC, TIER_HIGH_OR, TIER_2OF3_DASM off; LISTENER_V4B_CACHE, LISTENER_KCACHE (Kimi) off; LISTENER_BEATS_TH (rescue c) off.
- F1 LISTENER_NEW_TYPE_ONCE, F4 LOCAL_WINNER, F5 SHADOW, F6 EDGE off; ONCE_GAP None; F8_BYPASS_BOTH, CONTEXT_F8_BYPASS, RESCUE_COVERED off; LISTENER_DASM_RANK off.
- LISTENER_ARBITER (stage-5 arbiter), LISTENER_SCENE_FIT (F3), SCENE_FIT_ALL (N1) off.
- MASKED_WEAK_PANNS / MISSING_KEEP / DASM_KEEP off; MASKED_WEAK_NEED_MASK stays True; FLEX_ONLY_CONFIRM off.
- CO_ONSET_ARB (R3), RELABEL_2L (R1) off; REPEAT_NEEDS_SILENCE (RPT-S), REPEAT_DASM_BRIDGE (DBR) off.
- MAX_SPAN None; AED_RELEASE None; ONSET_MONOTONE on (listed above).
- FIX_FAM / FIX_EARLY / FIX_CTRL / FIX_GATE off; ACTIVITY_GATE (I2) off; CONCEALED_ACTION None; GATE_BOX_CHECK off.
- PLAUSIBILITY_CHECK off; CORROBORATE_BELOW 0 (no corroboration band); USE_LOCALIZATION off.
- KIND_FROM_FRAMES / KIND_ALWAYS / DEPICT_V2 and the non-v3 "no depiction reads as this sound - dropped" branch: unreachable under PICTURE_V3.
- PICTURE_SENSE, PICTURE_LOOK_VLM, PICTURE_LOOKALIKE_VLM off; PICTURE_MIN_CONF None; CONFIDENCE_FADE off; GROUP_LIVE off (GROUP runs in a subprocess).

---

## File formats

### media.json (pipeline.py:56; `MediaInfo.to_dict`)
`{video_path, wav_path, duration, sample_rate}`. Real example (docs/inspector/data.json, clip ambient_nature_rainforest_7629):
```json
{"video_path": "/home/dsi/adamg/MscProj/data/input/benchmark/mixed/ambient_nature_rainforest_7629.mp4",
 "wav_path": "/home/dsi/adamg/MscProj/data/work/protocol_proposed_dev_monocap_v31/ambient_nature_rainforest_7629/audio.wav",
 "duration": 16.0, "sample_rate": 16000}
```

### events.json (pipeline.py:87; `AudioEvent.to_dict`, src/types.py:54-84)
List of `{label, start, end, confidence, source_on_screen, on_screen_prob, detail, spans, source, breaks, rescued, arbiter, agree}`.
No origin (BEATs / FlexSED / twin / band rescue / DR2) and no veto history. The inspector keeps only label/start/end/confidence:
`{"label": "Music", "start": 0.30000000000000004, "end": 15.75, "confidence": 0.7128391861915588}`.

### onset_trace.json (S4:32-47, pipeline.py:96-138)
Flat list of rows `{step, label, start, end, conf, note}` (rounded to 3 decimals; `conf` 0.0 for stage-5/6 rows). Steps written in
the shipped run, in order: `extract` (BEATs spans, note "BEATs"), `flexsed_raw` (FlexSED spans), `union` (after twin rule),
`mirror_veto` (DROPPED rows, note "R13-2 dropped (b 0.7)"), `masked_weak_veto` (DROPPED rows, note "N2 dropped"),
`veto` (survivors after all of fuse_flexsed + DR2; note "after the cross-detector and PANNs vetoes"), `refine` (after onset
refinement), `family` (stage-5 spec label/start/end, note "after consolidate_families and the plan"), `family_spans` (one
row per burst, note = spec.detail), `display` (shown windows after DEPICT/join/dwell/after/GROUP, note "as the panel will
show it"). No example file is in the repo (data/work is not checked in). Illustrative row:
`{"step": "mirror_veto", "label": "Vehicle", "start": 1.0, "end": 2.25, "conf": 0.224, "note": "R13-2 dropped (b 0.7)"}`.
What it lacks: the reason and values of every drop except two steps' bare names; no rows for DASM clip/local veto, K4A, BTP,
CONT, FlexSED veto, PANNs veto, listener keeps/rescues, FLAP/F8/ONCE, label filter, gate, dedup, MAX_SLOTS.

### augmentations.json (pipeline.py:101 and :120; `AugmentationSpec.to_dict`, src/types.py:87-116)
List of `{index, event_label, start, end, augment, confidence, reason, subject, image_prompt, placement, backend, image_path,
detail, talked_about, spans, source, breaks, rescued, arbiter}`. `reason` is free text accumulated with " | ". Reason texts that
mark a decision: "below display threshold (c < 0.35)[; source visible on screen anyway]", "stage 2 saw the source somewhere in
the clip; VLM to confirm", "salient non-speech sound, source not visible -> augment", "marginal (c) but people are reacting to it
-> augment", "… | people are reacting to it", "source visible on screen (<named>) - stay silent", "a kind of <X>, whose source
is visible - stay silent", "same sound as <X>; the frames say it is that one", "same picture as <X> (<why>)", "… | depiction
(v3, source <S>): <phrase>". Real example (inspector subset of fields, scored run v31):
```json
{"event_label": "Cricket", "start": 4.5, "end": 6.25, "augment": true,
 "reason": "salient non-speech sound, source not visible -> augment | depiction: Cricket chirping",
 "subject": "Cricket chirping", "source": null}
```

### gate_votes.json (pipeline.py:116; R:1666-1669)
List, one row per (sound, stretch): `{label, start, end, confidence, stretch:[a,b], seen, name, ab, desc, named}` (+ `fix_gate`,
`box` only with inactive flags). `name`/`desc` are bool, `ab` is true/false/null (split), `named` is the ≤5-word phrase or
"nothing". Not stored: description text, the MAKES_SOUND replies, raw a/b letters, the frame times. Real examples
(docs/inspector/data.json):
```json
{"label": "Siren", "start": 0.14, "end": 9.75, "confidence": 0.8204, "stretch": [0.14, 4.945],
 "seen": true, "name": true, "ab": false, "desc": true, "named": "white police car"}
{"label": "Cricket", "start": 4.5, "end": 6.25, "confidence": 0.3629, "stretch": [4.5, 6.25],
 "seen": false, "name": false, "ab": false, "desc": false, "named": "nothing"}
```

### depict_answers.json (WORK_DIR, all clips; depict.py:57-94)
`{clip: {"<label>|<spec.start:.2f>": {subject, makers:[gate named phrases], event, lookalike_yes, silenced, event_replies?:[r1, r2]}}}`.
Real (benchmark/gold/depict_answers_bench.json): `{"tg_d016": {"Cough|5.75": {"subject": "Person coughing", "makers": [], "event": false, "lookalike_yes": false, "silenced": false}}}`; one silenced entry there: `b3_crossing_bells` / `Steam|0.22`.

### group_answers.json (WORK_DIR, all clips; group.py:83-96, 141-143)
`{clip: {"<label>|<picture start:.2f>": ["<answer order same/new>", "<answer order new/same>"]}}`; `{}` for a clip with no pair.
Real (benchmark/gold/grp/group_answers_bench.json): `"b3_barbershop": {"Electric shaver, electric razor|0.14": ["same", "same"]}`,
`"as_explosion_XJ8lc3I6": {"Gunshot|0.00": ["new", "new"], "Explosion|5.68": ["new", "new"]}`.

### scene-margin answers (`<WORK_DIR>/scene_videos.json.logit_answers.jsonl`, S4:1528-1529)
One JSON per line: `{key:[clip, family, start, end], label, verdict: true|false|null, answers:[{stretch:[a,b], answer:"yes"|"no"|"", d}], video}`.
Real (text variant, benchmark/gold/scenemargin_diff.json): `{"key": ["b3_favela_rio", "Train", 14.72, 16.5], "label": "Train", "verdict": false, "answers": [{"stretch": [14.72, 16.5], "answer": "no."}], "video": ".../b3_favela_rio.mp4"}`; logit values for the same ask in benchmark/gold/logit_scene_recheck.json (`"d_twin": -10.875`).

### picture_verify.json (per clip; S6:619-623)
List of `{clip, index, label, source, subject, tries:[{try, seed, prompt, ok, picked, intended, options, text, saw, learned?}], final}`.

### Listener caches (benchmark/gold/<split>_listener*.json, each `{"_meta": {...}, "items": [...]}`)
- `<split>_listener.json` (yes/no, R13-3; pools P1/P2/P3): `{pool, label, start, end, conf, origin, run_len, clip, family, depictable, gold, null_family, cut_start, cut_end, score, null_score, p_yes, null_p_yes}`; score = logit(yes) − logit(no) for "Is the sound of {family} present in this recording? Answer yes or no." on [start−1, end+1].
- `<split>_listener_v.json` (amendment A, Qwen3-Omni; pools P1/P2/PV). Real P2 item (dev_listener_v.json, abridged):
```json
{"clip": "ambient_citywalk_nyc_1689", "pool": "P2", "family": "Air brake", "label": "Air brake", "start": 8.0, "end": 8.36,
 "cut_start": 7.68, "cut_end": 8.68, "run_len": 0.36, "peak": 0.558, "depictable": true, "null_family": "Pig",
 "run_start": 8.0, "run_end": 8.36, "variants": ["V1","V2","V3","V4"], "v1_x_options": ["ice cream truck, ice cream van","air horn, truck horn","rattle"],
 "run_audio": [6.68, 9.68], "v2_x_ctrl": [4.68, 7.68], "v3_cut": [5.0, 11.36],
 "accept": {"V1": true, "V2": false, "V12": false, "V3": false, "V4": false}, "null_accept": {...},
 "v1_x": {"p": {"X": 0.827, "none": 0.168, ...}}, "v2_x": {"s_run": 3.5, "s_ctrl": 3.0, "ctrl": [4.68, 7.68]},
 "v3_x": {"text": "4.0\nAssistant", "t_clip": 9.0}, "v4_text": "Mechanical hiss\n...",
 "v4_x": {"names": ["air brake","truck"], "matched_word": [], "matched_cos": [], "cos": [0.33, ...]}}
```
  P1 items carry only V1/V2/V12 (`"variants": ["V1","V2"]`, `"accept": {"V1": false, "V2": false, "V12": false}`); PV items carry the vetoed span start/end plus run_start/run_end/peak/cached_score.
- `<split>_listener_afn.json` (Audio Flamingo Next, same keys/pools): adds `afn_v4_text`, `accept:{V4, YN0}`, `afn_v4_x:{names, matched_word, matched_cos, cos}`, `afn_yn_x` (yes/no logit margin). Real: `"afn_v4_text": "air conditioner", "accept": {"V4": false, "YN0": true}, "afn_yn_x": 0.125`.
- `<split>_listener_p1v4.json` (K-V4/K4A/WW lists, pool P1): `{clip, pool, family, label, start, end, run_audio, qwen_v4_text, qwen_fams, af_v4_text, af_fams}`. Real: `{"family": "Vehicle", "start": 0.14, "end": 1.0, "qwen_v4_text": "Train\nustrian...", "qwen_fams": ["Train"], "af_v4_text": "train", "af_fams": ["Train"]}` - this is exactly the "Audio Flamingo said 'train'" evidence, but nothing links it to the drop it caused.
- `<split>_listener_p4.json` (DR2, pool P4): `{clip, pool, family, label, start, end, peak (DASM), run_audio, cut_start, cut_end, run_len, null_family, qwen_v4_text, qwen_v4_match, qwen_v4, afn_v4_text, accept:{V4, YN0}, afn_v4_x, afn_yn_x, ...}`.
- DASM frames `data/work/dasm_<split>/<clip>.npz`: `fw [T, Q]`, `times`, `labels`. FineLAP `data/work/finelap_<split>/<clip>.npz`: `labels, fs, fe, scores`.

Effective rules from these caches: F7/N2/DASM-clip listener = P1 V12 (or yes/no > 3 from `<split>_listener.json` if the
span is not in the variants cache); K-V4, K4A, DASM-local "both" = `qwen_fams` / `af_fams` of p1v4 plus AF `accept.V4` on P1;
rescue (a)/(b) = TIER on P2/PV items (peak ≥ 0.6: Qwen V4; else Qwen V4 and AF V4).

---

## What the inspector knows today (docs/inspector/data.json + build.py + benchmark/gold/inspector_data.py)

Built from the **scored** renders, not the shipped arm: tags `dev_monocap_v31` (DEV, 49 clips), `test_final_v33` (TEST, 60),
`sliceB_v32` (30); systems "ours" (`protocol_proposed_<tag>`) and "blind" (`protocol_blind_a2i_<tag>`, no gate).

- Top level: `sets` {DEV, TEST, sliceB: tag, clips, sounds, needed_2_3, visible, obvious_not_visible, categories, stats{ours, blind, silence}}, `clips` [139], `note`. build.py adds `sensitivity`, `build_note`, `window`, `baselines`, and per clip `derived`, and writes data.js.
- Per clip: `clip, split, tag, category, sounds, systems{ours, blind}` (+ `derived`).
- Per gold sound: `label, start, end, needed, importance, visible, obvious, heard{BEATs|PANNs|FlexSED: {top:[[label, score]×5], own, bar}}` (heard = detector peaks in [onset−0.5, onset+1.0], added by docs/inspector/add_heard.py).
- `systems.ours`: `sounds` [per gold sound {outcome: hit|miss|don't care|not needed (…), late?}], `pictures` [{label, start, end, class (hit / wrong: a different sound / wrong: no such sound / wrong: source visible or obvious / duplicate / don't care), sound?:[gold idx]}], `gate` (= gate_votes.json rows), `events` (events.json, 4 fields), `augmentations` (7 fields: event_label, start, end, augment, reason, subject, source), `media`. `systems.blind`: sounds + pictures only.
- `derived`: `ours_pics[j]` {gate (index of the gate stretch it came from), gate_approx, aug, aug_approx, kept: [[a, b, "before"|"join"|"dwell"]] parts shown while the sound was not playing}; `blind_pics[j]` {gate, gate_approx, in_ours, kept}; `miss.{ours|blind}[i]` {reason: taken by another sound | removed by the gate | timing | detected, not drawn | never detected, pic_start/pic_label/late/event_label/confidence, gate (near stretches when removed by gate), fam_pics, fam_events, why:{step, text}}; `near_gate[i]` (gate stretches of the family near the sound); `family[i]` (canonical family per gold sound).

Per **missed sound** it can say: whether any same-family event existed near the onset (with the final events.json confidence),
what the detectors heard at the onset (top-5 + own score vs bar), the augmentation `reason` text in plain words (display bar,
gate visible + VLM's named object, family rule, dedup), the gate stretch votes (name/ab/desc/named), and timing / matching to
another sound. It cannot say which stage-4 step removed a span (mirror, N2, DASM clip/local, K4A, CONT, FlexSED, PANNs, FLAP,
F8, ONCE), nor the listener answers, nor the label filter step explicitly (only "label filter" guessed via `drawable()` when no
augmentation exists), nor depiction/dedup similarity values, nor DEPICT/GROUP/MAX_SLOTS.

Per **wrong picture** it knows: label/start/end, scorer class, the gold sound it covered (if any), the gate stretch and votes it
came from (approximate match), the augmentation index (thumbnail) and the `kept` parts (dwell/join padding). It does not know
the picture's stage-4 origin (BEATs / FlexSED twin / rescue / DR2), which vetoes it passed and by what margin, which listener
accepted it, or the subject/verify history beyond `subject`.

docs/inspector2/README.md already specifies the target: a per-candidate `trail` of `{step, res, value, bar, asks:[{q, a, vote}], note}`
written by a `src/trail.py: decide(...)` call at every decision point into `trail.json`, with `res` ∈ pass/drop/move/merge/relabel/rescue/skip.
The ids in the tables above are meant to be those step ids.
