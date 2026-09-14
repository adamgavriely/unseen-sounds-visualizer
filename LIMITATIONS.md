# Limitations

What the numbers in the report do and do not support. Seeded 2026-09-14 from
`docs/failure_catalogue.md` and the Limitations section of `docs/project_notes.tex`; the
report's Limitations section mirrors this file.

## Evaluation

- **The proposal's reference was circular.** The model-derived reference (tags `v2`, `v3`) is
  written by the system's own VLM from the detector's own events, so it rewards whichever
  system repeats the detector. Found in review on 2026-09-14. The corrected primary is the
  independent reference (`*_indep`): Qwen2-Audio + CLAP verification + Idefics3 + Llama-3.1
  writer, none of which sees any system's output. Both are reported.
- **The independent reference's sentinel is uncalibrated.** The MiniLM cosine threshold that
  rejects unsupported sentences is fixed at τ = 0.50; calibration on DCASE gold is planned but
  not done.
- **One annotator.** Blind re-adjudication gave κ = 0.60 against the annotator's own earlier
  labels, with the instability in the positive class the gate is judged on.
- **One judge decides the headline.** A second judge reproduces the gated-vs-blind gap
  (κ = 0.753) but moves the captioning baseline by +0.61; rankings involving the text
  baseline are a property of the judge.
- **100 clips, 25 per category.** A per-category mean carries roughly ±0.2; paired
  bootstrap CIs (`scripts/paired_stats.py`) are the numbers to read, not the means.
- **No human study.** An LLM scoring a VLM's description of a picture is a proxy for a deaf
  viewer's comprehension; the correlation is unmeasured.
- **Onset timing evaluated on 7 hand-labelled onsets** (3 clips): +0.9 s → +0.3 s mean error is
  indicative only. A DCASE-scale onset evaluation is future work.

## Detector (stage 4)

- **Vocabulary.** AudioSet has no "phone alert"; no model trained on it can name one
  (catalogue #3).
- **Masking by music and speech.** A siren under a film score is heard only at its close-up
  (#4); a tonal alert is mis-heard as a tuned instrument. Source separation inside the detector
  is the general remedy and is not yet in the shipping configuration.
- **Confident phantoms.** Dog 0.60 that nobody hears (#5), ice-cream truck at a station (#6),
  horse/train/truck at a quarry blast (#7), sheep in a jungle (#2). No threshold reaches them and
  the plausibility veto that would have caught some was turned off after 2 right / 2 wrong (#16).
- **Confident mislabels.** A door opening detected as "Gunshot" 0.50 with the right timing (#17).

## Visibility (stage 5)

- **Timid toward on-screen sources.** On DCASE 2025 gold (258 events): 97% of off-screen
  sounds are correctly kept, only 35% of on-screen sources are silenced (66.7% agreement).
  DCASE frames are 360° captures, so "on screen" there is defined by a field of view.
- **Unstable to frame choice.** A 0.5 s shift in frame sampling flipped a gunshot and a police
  siren between silenced and shown (#13).
- **The 7B model has a position bias**; every two-way question is asked in both orderings and
  agreement is required, which costs recall.

## Deduplication and depiction

- **Text similarity between short phrases is coarse** (SigLIP): insect buzzing merged into bird
  chirping at 0.82 (#14). Similarity merges now require time overlap; the coarse similarity
  remains.
- **Pictures are generated, not verified against the video**; a forced-choice check confirms
  the phrase reads as the sound, not that the rendered image does.

## Benchmark

- **274 clips, not 300.** Usable yield was 15–20% across four sourcing strategies.
- **Part of the queue was model-enriched.** The first 209 tags are the unbiased sample; later
  candidates were pre-screened.

## Process

- **The evaluation software produced plausible wrong numbers** several times (charitable score
  for an empty augmentation; a reference that could not say "nothing is missing"; a similarity
  that returned 0.0 for everything on a library upgrade). Each is guarded now; a single
  automatic number deserves less trust than its precision suggests.
- **Knobs were set once on the dev split** after 2026-09-13 (`docs/plan_robustness.md`); the
  demos in the notes illustrate and do not tune.
