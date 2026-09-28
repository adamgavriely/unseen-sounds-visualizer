# TODO

Living list of work we decided to do later. Add when something is postponed; move to "Done" (with date and
commit) when it is finished; delete only if Adam drops it. Deadline: thesis 3 Oct 2026.

## Running now (28 Sept)
- [ ] Detector round 9: J2 (local-contrast check) PASSED the 415 (ΔC −0.106 [−0.178, −0.039]); now: fresh-set confirmation (after caches) + DEV simulation. Ship only if fresh passes AND DEV hits don't drop AND DEV wrong drops. `docs/prereg_round9_contrast.md`
- [ ] Fresh confirmation set caches (job 31330209 RUNNING; BEATs 180/422, then PANNs, PE-A-Frame, FlexSED) — then check 4 cache folders × 422 `.npz` (do not read logs/freshc_*.out). Nothing may be scored on it until a candidate passes the 415.
- [ ] Label "sense traps" (Honk → goose, Toot → car, Caw → crow, …): audit + code the subject/maker/checker on the ontology parent chain; car-horn "sound-wave lines" redraw.
- [ ] Picture wording 3-arm test: hand table vs VLM free text vs new (a) slot form + official ontology descriptions, (b) mistake mining. `docs/picture_sense_test_2026-09-28.md`

## Later
- [ ] Thesis limitation note: `src/audioset_parents.json` keeps one parent per label, but the official ontology has 38 multi-parent labels (e.g. Hiss → Cat/Snake/Steam). Scoring's same_family and the stage-5 family rule use the single-parent file (frozen, not changed); state it and, if time, count how many scored sounds are multi-parent labels (DEV only).
- [ ] **Mistake mining for all 215 drawable labels** (one-time, ~4–7 GPU h, at night) — only if the a+b test adopts the new method. Re-run whenever the picture model changes.
- [ ] Final check on the fresh set for whatever passes round 8 (and 7b), then decide on the detector (BEATs stays unless something passes).
- [ ] Sync the cluster copy of `src/stage4_audio_event_detection/__init__.py` (missing the BEATs self-veto block) and `config.py` after the round-7b/8 jobs finish.
- [ ] Update thesis ch5/ch6 with: external baselines, confidence floor, BEATs self-veto, family-rule fix, display rules, picture check + maker rule, rounds 4–8, sensitivity rows, sitting cancelled.
- [ ] Update the supervisor page (`docs/supervisor_2026-09-26/index.html`) with rounds 6–8 and the picture results.
- [ ] Future-work note for the thesis: a visible bird/macaw that is not the sound's source (gate error Adam accepts for now); COSED (Sept 2026, no code yet) for the FlexSED slot.
- [ ] Cluster disk: delete the PE-A-Frame `perception_models` revision (6.1 GB, fetched for 7b) after fresh caches job 31330209 ends (only that revision's snapshot/blobs); sam-audio-large + 7b wavs being deleted now; `shipped_v2_*`, `shipped_v_*_r1` folders.

## Waiting on Adam
- [ ] Supervisor meeting: second annotator, ethics for extra raters, thesis format/length, results chapter date.

## Done
- 2026-09-28 Detector round 7b: SAM-Audio retry fails QC again (music not removed); separation idea closed (6133f4f).
- 2026-09-28 Detector round 8: 17 cells, none passes the 415; closest I7 local-contrast veto (114fae3).
- 2026-09-28 Detector rounds 4 (BEATs self-veto replaces PANNs, 84de50b), 5 (EAT/Dasheng fail, 0ba10a6), 6 (DASM fails, 7315606), 7 (SAM-Audio QC stop, fd5444c); audit (20160d4).
- 2026-09-28 Picture check-and-redraw (5 refined tries), smoke/rattle/horn fixes, maker rule (8209491, bf54684, 64f336d).
- 2026-09-28 Sealed picture sitting cancelled (3d0d734); supervisor page updated (3509de1, 32750a4).
- 2026-09-28 Fresh confirmation set downloaded (931ecea).
