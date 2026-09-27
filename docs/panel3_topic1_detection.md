# Topic 1 — Sound detection (3 reviewers × 3 rounds)

Read `docs/panel3_common_context.md` first. Adam: *"we HAVE to recognize more sounds — losing so many sounds is very bad."*

## What the detector is and how it performs
- **Stack:** BEATs (AudioSet-2M, 527 classes, 2-s windows / 0.25-s hop) ∪ FlexSED (Dasheng + CLAP text queries over 215
  drawable family names, frame-level, cached) at bar 0.8, twin rule (a same-family BEATs span within 1 s absorbs a FlexSED
  span, earlier start kept), FlexSED veto τ 0.3 (drop a label FlexSED never hears in the clip), PANNs CNN14 veto τ₂ 0.05
  (drop a FlexSED-only span PANNs never hears), occlusion onset refinement clamped (never earlier than the anchor).
  Code: `src/stage4_audio_event_detection/__init__.py`, `flexsed_infer.py`, `beats_infer.py`.
- **Recall is the ceiling:** DEV miss autopsy — 11 of 21 misses never detected, 5 timing, 3 gate, 2 label filter. With the
  annotator's own sound list, the gate's F1 gain becomes significant (+0.067); the cost at β=2 would fall 2.63 → 1.23.
- **Ceiling in the caches (DEV-49, 36 needed sounds, best same-family score in [onset −0.5, +1.0] s):** shipped bars catch
  21; looser bars (BEATs 0.2 / FlexSED 0.5 / PANNs 0.3) 30; only 4 below every detector (hammer, clang, two golf hits).
  This is an oracle-bar ceiling read on the sounds it would be scored on — no bar may be chosen from it.
- **Why the recall does not convert (reviewer F1's diagnosis, DEV caches):** the lost sounds are masked by speech/music;
  FlexSED hears them (0.44–0.70) but BEATs and PANNs — the filters that make a low FlexSED bar affordable — score them
  0.000–0.026. Without those filters FlexSED under speech/music gives 2–5 false labels per clip.
- **AudioSet-Strong, 280 human-labelled clips (out of sample, descriptive):** BEATs 0.35: consequential recall 54.5 %,
  6.41 false spans/min; shipped stack 51.3 %, 4.56; FlexSED at 0.8 recovers 0 of 31 masked consequential events.
- **Detector round today (amendments 22–23, `docs/prereg_v4.md`; Stage 0 on the 280 clips,
  `benchmark/detector_round_stage0.json`):**

      cell                          conseq onset-recall  recall   masked(31)  false/min
      shipped (FlexSED 0.8)               25.0 %         50.4 %    38.7 %       4.46
      FlexSED 0.6 / 0.5                   25.4 / 25.4    52.2/53.6  38.7        4.89 / 5.12
      + time-aligned corroboration        25.4 / 25.4    52.2/53.6  38.7        4.61 / 4.80
      E: cascade — weak BEATs spans       46.4 %         58.9 %    45.2 %       6.90
         (0.175–0.35) promoted when FlexSED ≥ 0.3 or PANNs ≥ 0.05 agrees within 1 s
      F: twin-rule bug fix                25.9 %         51.3 %    38.7 %       4.56
  By the rule written first nothing was picked (no cell at ≤ shipped false spans raised recall). The cascade E is now a
  separate, disclosed question tested on DEV renders (amendment 23: blind arm must gain ≥ 2 hits at viewer cost ≤ shipped);
  renders are running.
- **Tried and dropped (13 pre-registered attempts, `docs/LEDGER_2026-09-26.md`):** PANNs alone, CLAP verifier, Qwen2-Audio
  verifier, Demucs/HTDemucs/DeepFilterNet separation views, FLAM ×2, PretrainedSED (5 backbones, ensemble), fusion V1–V3,
  AST/CED votes, hysteresis onset, rank admission, prompt ensemble, per-family FlexSED bars.
- **Web check (27 Sept):** no newer detector with public weights; boundary-aware SED (PSDS1 49.6, arXiv 2601.04178) and
  COSED (open-vocabulary, arXiv 2609.30083) release no code or checkpoints; DASM has no inference code.
- **Constraint to question:** the project has been "training-free". Is light adaptation (e.g. fine-tuning a detector head
  on AudioSet-Strong, or a small fusion model trained on the 280 calibration clips) acceptable or wise in 6 days?

## Questions (round 1 — ≤ 10 lines each, concrete: files, models, cost, decision rule)
1. What is the most promising way to recognise more of the needed sounds this week — and what is its false-alarm price?
   (Consider: the cascade E; audio-LLM verification of weak candidates; text-queried separation then re-detection;
   fine-tuning/training allowances; temporal context; ensembles; anything else with public weights.)
2. How should a detector change be judged without TEST (DEV-49, the 280 calibration clips, a new held-out AudioSet-Strong
   download, synthetic mixes)? Exact decision rule.
3. Is the "recognise more sounds" goal right for the viewer, given that each added detection is right only ~23 % of the
   time (break-even p > β/(4+β))? What should the thesis say about the detector ceiling?
4. The one thing not to do.
