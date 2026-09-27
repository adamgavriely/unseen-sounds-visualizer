# Panel 2026-09-27 — can the detector still be improved in 6 days? (one round, then a plan)

Adam: *"is there a way we can improve the detector? that's the main part for me atm."* Deliver by 3 Oct.

## New facts since your last round
- **Final TEST table scored** (amendment 21): F1 null; precision, wrong pictures, cost, clean clips significant after Holm.
  Its committed rule: *no further TEST number is read.* A 6th deliberate TEST read would need Adam's explicit yes and
  would be disclosed as such.
- **B.1** (`flexsed_recheck_dev49.json`): on the clean DEV-49, amendment 8 passes rules 1 and 3, fails rule 2 by 0.011.
  **At FlexSED's shipped bar 0.8 only 1 of 7 masked DEV sounds is recovered** (6/7 at ≥ 0.35).
- **D6** (`benchmark/audioset_stage4_report.json`, 280 AudioSet-Strong calibration clips, human frame labels, out of
  sample): BEATs 0.35 conseq recall 54.5 %, 6.41 false spans/min; shipped stack 51.3 %, 4.56; union without vetoes 58.9 %,
  8.83; **FlexSED 0.8 recovers 0 of 31 masked consequential events.**
- DEV miss autopsy: 11 of 21 misses never detected, 5 timing, 3 gate, 2 filter. Oracle: with the annotator's sound list
  the gate's ΔF1 is significant (+0.067). Cost curve at β=2 with a perfect sound list: 1.23 vs 2.63 today.
- **Web check (today):** no newer detector with public weights. Boundary-aware SED (arXiv 2601.04178, PSDS1 49.6 vs
  46.5) and COSED (arXiv 2609.30083, open-vocabulary) release no code or checkpoints. DASM has no inference code.
- Cached frame scores available: gold 139 clips — BEATs (`benchmark/gold/beats_fw`), PANNs (`panns_fw`), FlexSED
  (`data/work/flexsed_cache`, 215 family queries; also a 3-prompt cache); calibration 280 clips — BEATs, PSED ×5
  backbones + ensemble, SSLAM, FlexSED (`data/work/flexsed_calib`), PANNs (new today). The login node has internet
  (more AudioSet-Strong clips could be downloaded).
- Rules: the project is training-free (calibrating thresholds is allowed; training a network is not); a shared stage is
  chosen by the BLIND system's own DEV F1 (or a detector-level metric), never by ΔF1; every change needs a bar written
  before it runs; Adam's time is scarce.

## Questions (≤ 8 lines each, concrete, cite files)
1. Is there an honest detector improvement worth building this week? Name at most two, ranked, with the expected effect
   and why earlier attempts (13 failed) do not already rule it out. Candidates to consider (or reject): per-family FlexSED
   bars (amendment 11 fitted them on the 280 clips — recall 0.455 at cost 4.65, not selected), a lower single FlexSED bar
   with the vetoes, a calibrated late fusion of cached detectors (BEATs, FlexSED, PANNs, PSED, SSLAM) with per-family
   thresholds fitted on the 280 calibration clips, the 3-prompt FlexSED cache, speech-aware thresholds, others.
2. **Evaluation without TEST:** what evidence would count? E.g. (a) detector-level on a larger held-out AudioSet-Strong set
   downloaded now (how many clips, which split, which metric: onset-recall at matched false spans/min), (b) end-to-end on
   DEV-49 only, (c) a 6th TEST read with Adam's yes. Which combination is defensible, and what exact decision rule?
3. If the improvement passes: does the thesis's system change (new headline configuration), or is it reported as a
   "detector upgrade" beside the frozen system? What does that do to the final TEST table?
4. The one thing NOT to do.

Return your answer under your heading.

## Addendum (measured while you work, DEV-49 caches, best same-family score in [onset − 0.5, onset + 1.0] s)
36 needed DEV sounds. BEATs ≥ 0.35: 13, ≥ 0.2: 20, ≥ 0.1: 23. FlexSED ≥ 0.8: 11, **≥ 0.5: 27**, ≥ 0.3: 30. PANNs ≥ 0.3: 10,
≥ 0.1: 16, ≥ 0.05: 21. Shipped bars (BEATs 0.35 or FlexSED 0.8) reach **21**; looser bars (BEATs 0.2 or FlexSED 0.5 or
PANNs 0.3) reach **30**; only **4** are below every detector's floor (hammer, clang, two golf whacks). The recall is in
the caches; the question is false alarms. Note: FlexSED bars 0.7 / 0.6 were costed (4.61 / 5.43 vs 3.63) — check
whether that sweep had the PANNs veto (amendment 16 came later) and whether a corroboration rule (e.g. FlexSED ≥ 0.5
only where BEATs or PANNs also rises) was ever tried.
