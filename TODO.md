# TODO

Living list of work we decided to do later. Add when something is postponed; move to "Done" (with date and
commit) when it is finished; delete only if Adam drops it. Deadline: thesis 3 Oct 2026.

## Running now (30 Sept)
- [ ] **1 Oct last-day push (detector thread, until ~13:00 UTC):** (1) Round 47 GROUP: GRP-A won DEV (28/58, 18, 2.197); TEST job 31599527 -> if it holds, ship as a src/ flag; if it merges the pet_parrot Dog barks, pre-register GRP-AV (audio + frames). (2) HUMAN-2 gate push (vision): why variant (b) lost 1 needed; refine with grounded source + crop + "is it making the sound now" or motion in the box; gate_gold set first, then DEV. (3) kill_flags: which stage-4 step removes 6 strongly heard misses -> relax one flag (held-out 415 first, then DEV). Rules: prereg first, DEV select, TEST once per new best.
- [ ] Round 14 (DEV): best arm **TO1+F7F8** = TIER listener rescue + ONCE + R13-1 + F7 swap + F8 DASM vote: 18 hits / 24 wrong, cost 2.45 vs B0r 2.78 (Δ −0.33 [−0.86, +0.08]), better at every β and both halves. Next: C1 on DEV2 (needs DASM), then ONE TEST2 scoring (`benchmark/gold/test2_final.py`) once more tagger batches are in. `docs/prereg_round13_detector_push.md`
- [ ] Human annotator (29 Sept): `tagger/` package (100 web clips d001–d100, manual tool, guide) zipped and sent by Adam. Private source map `benchmark/gold/delegation_sources.json`, QC `benchmark/gold/delegation_qc.json`. When the export (`tagger_<name>_<date>.json`) comes back: check it loads with `score_per_sound.load_gold`, spot-check quality, and fix `resolve_label` silently mapping free text like 'shopping trolley rattling' to 'Ping' before any scoring. If all 100 are good, Adam wants another 100.
- [x] Round 12 STOPPED at step 1 (28 Sept, job 31330706): under the corrected cost v2 (MID names, depictable filter, runs merged <= 2 s, video hit window) ours is WORSE than showing nothing on the 280 (1.650 vs 0.986, d +0.664 [+0.379, +0.950]); its 159 false spans alone cost 1.136. By the prereg the AudioSet harness cannot track the task; no cell re-scored, no new cell, nothing on 415/DEV/fresh, no src change. `docs/prereg_round12_v2.md`
- [ ] Detector round 9: J2 PASSED the 415 (ΔC −0.106) but FAILED the DEV ship check (hits 14→13, wrong 24→25) → not shipped; fresh-set score still runs as pre-registered (report only). Ship only if fresh passes AND DEV hits don't drop AND DEV wrong drops. `docs/prereg_round9_contrast.md`
- [x] Fresh confirmation set caches complete (422 clips). AudioSet harness is retired as a decision test (round 12), so the fresh set is report-only now.

- [ ] Merged sets (Adam 30 Sept): DEV = DEV + DEV2, TEST = TEST + TEST2; final candidate scored once on merged TEST at the end. Build merged scoring after the batch-2 prep chain (jobs 31563919-23).

- [ ] Thesis: insert `docs/thesis/detector_rounds_13_15.md` after §5.8 of ch5; ch5 §5.14 add the R13-1 TEST exposure (29 Sept, 11th), §5.15 row 11; §5.7 stale 'not re-autopsied' line (TO1+F7F8 has a full trace). Arm count on DEV = 106 full-pipeline arms (not ~60).


- [ ] ComfyUI nodes (`comfyui_nodes/__init__.py`) still configure `config.use_v4("590")` (the old shipping row): switch them to `config.use_shipped()` and call `src/listener_prep.ensure_listener_inputs` + `config.set_listener_split` before stage 4, so ComfyUI runs the shipped TO1+F7F8 + N2b + DR2 + K-V4.

## Later
- [ ] Thesis limitation note: `src/audioset_parents.json` keeps one parent per label, but the official ontology has 38 multi-parent labels (e.g. Hiss → Cat/Snake/Steam). Scoring's same_family and the stage-5 family rule use the single-parent file (frozen, not changed); state it and, if time, count how many scored sounds are multi-parent labels (DEV only).
- [ ] (Only if a fixed generic method wins a new test) mistake mining for all 215 labels at night. Possible fixes for a new test: more specific object slot, allow hands for human-action makers, better mining judge.
- [ ] Final check on the fresh set for whatever passes round 8 (and 7b), then decide on the detector (BEATs stays unless something passes).
- [ ] Next detector ideas (only after the DEV check; confirmation then needs TEST-once or new videos, since DEV is used): (1) short-sound path: FlexSED peak whose run is < 0.5 s -> make a 1-s span, keep only if DASM agrees (22 of 61 band sounds are blocked by the 0.5-s minimum); (2) weak-twin rule.
- [x] (28 Sept, round 13 start) Synced cluster stage4 (self-veto block). Was URGENT: sync `src/stage4_audio_event_detection/__init__.py` (cluster copy lacks the self-veto; use_shipped() sets PANNS_VETO 0 → NO veto at all on FlexSED-only spans) and `config.py`, when no job imports them.
- [ ] Thesis ch5/ch6: add round 13 (listener verifier negative result: finds 5–8 dropped sounds at ~3 wrong per hit, every question form; R13-1 TEST same; B1 vs B0 on TEST).
- [ ] Update thesis ch5/ch6 with: external baselines, confidence floor, BEATs self-veto, family-rule fix, display rules, picture check + maker rule, rounds 4–12 (AudioSet test retired, DEV confirmatory), sensitivity rows, sitting cancelled, picture wording test.
- [ ] Update the supervisor page (`docs/supervisor_2026-09-26/index.html`) with rounds 6–12, the audits, the retired AudioSet test, the picture wording test and the DEV check results.
- [ ] Future-work note for the thesis: a visible bird/macaw that is not the sound's source (gate error Adam accepts for now); COSED (Sept 2026, no code yet) for the FlexSED slot.

## Waiting on Adam
- [ ] OK to re-download the official DASM weights (`CPF2/detect_any_sound`, 0.64 GB; code copy is local) — the best arm's F8 vote needs DASM on new clips; I deleted `~/Transformer4SED` on 28 Sept.
- [ ] Close the detector search? None of the 8 frozen candidates passes DEV. If not closed, next ideas below need TEST-once or new videos.
- [ ] Look at the picture wording contact sheet `docs/picture_sense_test/index.html` (confirm/override the blind by-eye verdicts); decide: drop the generic method or run a new test with its fixes.
- [ ] Supervisor meeting: second annotator, ethics for extra raters, thesis format/length, results chapter date.

## Done
- 2026-10-01 FineLAP step in `src/listener_prep.py` (`finelap_screen.py split`, ~/venv_flap subprocess, skips existing npz; old live splits get only this step). Check (job 31598689): DEV tg_d007 npz identical to finelap_cache (labels, fs/fe/scores max diff 0); live as_explosion npz built, `_require_caches` passes under use_shipped.
- 2026-09-30 On-the-spot listener inputs for new videos (ComfyUI): `src/listener_prep.py` runs the benchmark harness for one clip; `main.py` calls it when no `--listener-split` is given. Check: as_explosion answers identical to the shipcheck run (yes/no 95/95, variants 37/37, AF 37/37), ~5 min per clip on an H200.
- 2026-09-30 `use_shipped()` reverted to the PANNs veto 0.05 (scored config); BEATs self-veto worse on DEV and TEST. Fable supports.
- 2026-09-30 Round 14 amendments C, D, I, J, K (+K2 with the 27 missing listener answers, 58e9420) and the BEATs weak-band P3 screen: none beats TO1+F7F8 on DEV. `docs/prereg_round13_detector_push.md`
- 2026-09-29 Round 13 (12-h detector push): only R13-1 twin-max passed DEV (+1 hit); TEST = same (no picture changed), nothing ships. Audio-LLM (Qwen3-Omni) rescue: +5..8 real dropped sounds but ~3 wrong per hit under yes/no, MC, paired-cut, localisation, open-list questions; worse on both DEV halves. B1 self-veto worse than PANNs on TEST too (2.83 vs 2.63, p 0.043). `docs/prereg_round13_detector_push.md`
- 2026-09-28 Cluster: PE-A-Frame old revision deleted (7.7 GB; 37 GB free); `~/Transformer4SED` deleted.
- 2026-09-28 DEV confirmatory detector check: all 8 candidates (EAT-R, DASM D1, I4, I6, I7, R1, R6, R7) worse than ours; best I7 −0.08 cost but loses 1 hit; R1 +3 hits but +7 wrong. `~/Transformer4SED` deleted. `docs/dev_candidates_check_2026-09-28.md`
- 2026-09-28 Picture wording test: hand table 7/7 vs new generic 3/7 on known cases, tie on 25 unseen → hand table stays.
- 2026-09-28 Round 10 (280): DASM agreement filters band rescues well (7 rescued, 12 false); 9 of 61 reachable, 22 blocked by the 0.5-s minimum (4b5f446, f6d2cc3). Round 11 (280): masker tests separate nothing (latest commit).
- 2026-09-28 Label sense traps: makers from all ontology parents; multi-sense labels never guessed; Honk → goose; horn sound-wave wording (0429a0e).
- 2026-09-28 Detector round 7b: SAM-Audio retry fails QC again (music not removed); separation idea closed (6133f4f).
- 2026-09-28 Detector round 8: 17 cells, none passes the 415; closest I7 local-contrast veto (114fae3).
- 2026-09-28 Detector rounds 4 (BEATs self-veto replaces PANNs, 84de50b), 5 (EAT/Dasheng fail, 0ba10a6), 6 (DASM fails, 7315606), 7 (SAM-Audio QC stop, fd5444c); audit (20160d4).
- 2026-09-28 Picture check-and-redraw (5 refined tries), smoke/rattle/horn fixes, maker rule (8209491, bf54684, 64f336d).
- 2026-09-28 Sealed picture sitting cancelled (3d0d734); supervisor page updated (3509de1, 32750a4).
- 2026-09-28 Fresh confirmation set downloaded (931ecea).
