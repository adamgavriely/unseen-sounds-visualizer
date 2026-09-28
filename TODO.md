# TODO

Living list of work we decided to do later. Add when something is postponed; move to "Done" (with date and
commit) when it is finished; delete only if Adam drops it. Deadline: thesis 3 Oct 2026.

## Running now (28 Sept)
- [ ] DEV check (report-only) of EAT-R, DASM D1, I4, I6, I7 — jobs 31330562 (caches) → 31330563 (stages+score) queued. When done: read D5 first, scp benchmark/gold/dev_candidates_check.json + data/work/devcand/stage4.json + */_stage5_log.json, write Results in `docs/dev_candidates_check_2026-09-28.md`, delete ~/Transformer4SED. Baseline B1 (shipped) primary, B0 beside.
- [ ] Round 12 (Adam: "do everything good, within reason"; approved bark-runs-as-one + video hit window): cost v2 → shipped must beat silence → re-score all past cells on 280 → new cells N1 weak-twin, N2 short-sound path, N3 combo, N4 ontology vetoes (DASM-filtered) → ≤3 picks to 415 → DEV ship rule → fresh (once) → ship behind a flag. Also sync cluster stage4/config. `docs/prereg_round12_v2.md`
- [ ] Detector round 9: J2 PASSED the 415 (ΔC −0.106) but FAILED the DEV ship check (hits 14→13, wrong 24→25) → not shipped; fresh-set score still runs as pre-registered (report only). Ship only if fresh passes AND DEV hits don't drop AND DEV wrong drops. `docs/prereg_round9_contrast.md`
- [ ] Fresh confirmation set caches (job 31330209 RUNNING; BEATs, PANNs, PE-A-Frame 422/422; FlexSED 20/422) — then check 4 cache folders × 422 `.npz` (do not read logs/freshc_*.out). Nothing may be scored on it until a candidate passes the 415.
- [ ] Picture wording 3-arm test (jobs 31330354 prep → 31330355 A+B, 31330356 C): when done, pull results, blind verdicts BEFORE opening key.json, `scripts/picture_sense_sheet.py --phase blind/table/sheet`, results into `docs/picture_sense_test_2026-09-28.md`. Watch the weak slot forms (dental drill "patient's mouth").

## Later
- [ ] Thesis limitation note: `src/audioset_parents.json` keeps one parent per label, but the official ontology has 38 multi-parent labels (e.g. Hiss → Cat/Snake/Steam). Scoring's same_family and the stage-5 family rule use the single-parent file (frozen, not changed); state it and, if time, count how many scored sounds are multi-parent labels (DEV only).
- [ ] **Mistake mining for all 215 drawable labels** (one-time, ~4–7 GPU h, at night) — only if the a+b test adopts the new method. Re-run whenever the picture model changes.
- [ ] Final check on the fresh set for whatever passes round 8 (and 7b), then decide on the detector (BEATs stays unless something passes).
- [ ] URGENT before any cluster pipeline run: sync `src/stage4_audio_event_detection/__init__.py` (cluster copy lacks the self-veto; use_shipped() sets PANNS_VETO 0 → NO veto at all on FlexSED-only spans) and `config.py`, when no job imports them.
- [ ] Update thesis ch5/ch6 with: external baselines, confidence floor, BEATs self-veto, family-rule fix, display rules, picture check + maker rule, rounds 4–8, sensitivity rows, sitting cancelled.
- [ ] Update the supervisor page (`docs/supervisor_2026-09-26/index.html`) with rounds 6–8 and the picture results.
- [ ] Future-work note for the thesis: a visible bird/macaw that is not the sound's source (gate error Adam accepts for now); COSED (Sept 2026, no code yet) for the FlexSED slot.
- [ ] Cluster disk: delete the PE-A-Frame `perception_models` revision (6.1 GB, fetched for 7b) after fresh caches job 31330209 ends (only that revision's snapshot/blobs); sam-audio-large + 7b wavs being deleted now; `shipped_v2_*`, `shipped_v_*_r1` folders.

## Waiting on Adam
- [ ] Supervisor meeting: second annotator, ethics for extra raters, thesis format/length, results chapter date.

## Done
- 2026-09-28 Round 10 (280): DASM agreement filters band rescues well (7 rescued, 12 false); 9 of 61 reachable, 22 blocked by the 0.5-s minimum (4b5f446, f6d2cc3). Round 11 (280): masker tests separate nothing (latest commit).
- 2026-09-28 Label sense traps: makers from all ontology parents; multi-sense labels never guessed; Honk → goose; horn sound-wave wording (0429a0e).
- 2026-09-28 Detector round 7b: SAM-Audio retry fails QC again (music not removed); separation idea closed (6133f4f).
- 2026-09-28 Detector round 8: 17 cells, none passes the 415; closest I7 local-contrast veto (114fae3).
- 2026-09-28 Detector rounds 4 (BEATs self-veto replaces PANNs, 84de50b), 5 (EAT/Dasheng fail, 0ba10a6), 6 (DASM fails, 7315606), 7 (SAM-Audio QC stop, fd5444c); audit (20160d4).
- 2026-09-28 Picture check-and-redraw (5 refined tries), smoke/rattle/horn fixes, maker rule (8209491, bf54684, 64f336d).
- 2026-09-28 Sealed picture sitting cancelled (3d0d734); supervisor page updated (3509de1, 32750a4).
- 2026-09-28 Fresh confirmation set downloaded (931ecea).
