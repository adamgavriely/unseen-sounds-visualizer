# TODO

Living list of work we decided to do later. Add when something is postponed; move to "Done" (with date and
commit) when it is finished; delete only if Adam drops it. Deadline: thesis 3 Oct 2026.

## Running now (28 Sept)
- [ ] Round 13: 12-h detector push (Adam, 28 Sept). DEV = development, ONE TEST exposure of one candidate on top of B0. `docs/prereg_round13_detector_push.md`
  - Status 29 Sept: R13-1 twin-max is the only DEV-eligible rule (+1 hit, cost 2.78 -> 2.69). Yes/no listener rescue +8 hits but +41 wrong (not eligible). Running: 4 stricter listener variants (amendment A) on DEV + TEST features. TEST caches + gold-free TEST listener cache ready (D0 120/120, D5 60/60). TEST scorer not built yet (the one exposure).
  - If Adam wants more annotation: draft a job offer for ~50-60 new clips (same tool/labels as DEV) to confirm without spending TEST.
- [x] Round 12 STOPPED at step 1 (28 Sept, job 31330706): under the corrected cost v2 (MID names, depictable filter, runs merged <= 2 s, video hit window) ours is WORSE than showing nothing on the 280 (1.650 vs 0.986, d +0.664 [+0.379, +0.950]); its 159 false spans alone cost 1.136. By the prereg the AudioSet harness cannot track the task; no cell re-scored, no new cell, nothing on 415/DEV/fresh, no src change. `docs/prereg_round12_v2.md`
- [ ] Detector round 9: J2 PASSED the 415 (ΔC −0.106) but FAILED the DEV ship check (hits 14→13, wrong 24→25) → not shipped; fresh-set score still runs as pre-registered (report only). Ship only if fresh passes AND DEV hits don't drop AND DEV wrong drops. `docs/prereg_round9_contrast.md`
- [x] Fresh confirmation set caches complete (422 clips). AudioSet harness is retired as a decision test (round 12), so the fresh set is report-only now.

## Later
- [ ] Thesis limitation note: `src/audioset_parents.json` keeps one parent per label, but the official ontology has 38 multi-parent labels (e.g. Hiss → Cat/Snake/Steam). Scoring's same_family and the stage-5 family rule use the single-parent file (frozen, not changed); state it and, if time, count how many scored sounds are multi-parent labels (DEV only).
- [ ] (Only if a fixed generic method wins a new test) mistake mining for all 215 labels at night. Possible fixes for a new test: more specific object slot, allow hands for human-action makers, better mining judge.
- [ ] Final check on the fresh set for whatever passes round 8 (and 7b), then decide on the detector (BEATs stays unless something passes).
- [ ] Next detector ideas (only after the DEV check; confirmation then needs TEST-once or new videos, since DEV is used): (1) short-sound path: FlexSED peak whose run is < 0.5 s -> make a 1-s span, keep only if DASM agrees (22 of 61 band sounds are blocked by the 0.5-s minimum); (2) weak-twin rule.
- [x] (28 Sept, round 13 start) Synced cluster stage4 (self-veto block). Was URGENT: sync `src/stage4_audio_event_detection/__init__.py` (cluster copy lacks the self-veto; use_shipped() sets PANNS_VETO 0 → NO veto at all on FlexSED-only spans) and `config.py`, when no job imports them.
- [ ] Update thesis ch5/ch6 with: external baselines, confidence floor, BEATs self-veto, family-rule fix, display rules, picture check + maker rule, rounds 4–12 (AudioSet test retired, DEV confirmatory), sensitivity rows, sitting cancelled, picture wording test.
- [ ] Update the supervisor page (`docs/supervisor_2026-09-26/index.html`) with rounds 6–12, the audits, the retired AudioSet test, the picture wording test and the DEV check results.
- [ ] Future-work note for the thesis: a visible bird/macaw that is not the sound's source (gate error Adam accepts for now); COSED (Sept 2026, no code yet) for the FlexSED slot.

## Waiting on Adam
- [ ] Revert `use_shipped()` to the PANNs veto 0.05 (the scored config)? On DEV the BEATs self-veto is worse: hits 13 vs 14, wrong 30 vs 24, cost 3.10 vs 2.78 (B0 - B1 = -0.33 [-0.69, -0.04]). Self-veto lets through 8 FlexSED-only pictures (5 Insect, Coin, Bicycle, Snoring). Recommend yes.
- [ ] Close the detector search? None of the 8 frozen candidates passes DEV. If not closed, next ideas below need TEST-once or new videos.
- [ ] Look at the picture wording contact sheet `docs/picture_sense_test/index.html` (confirm/override the blind by-eye verdicts); decide: drop the generic method or run a new test with its fixes.
- [ ] Supervisor meeting: second annotator, ethics for extra raters, thesis format/length, results chapter date.

## Done
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
