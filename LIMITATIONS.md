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
- **The independent reference's verification gate was set once, on one comparison.** CLAP
  decides which of the audio LM's claims survive. The first gate (cosine above the mean + 2σ
  of 64 random AudioSet text decoys) left 45% of clips with an empty reference against 25%
  tagged `no_ambient`: a reference biased toward silence, which rewards the gated system's
  abstention — the mirror image of the circular bias the reference exists to remove. The
  shipping gate keeps a claim if this clip's audio ranks in the top 20 of the 100 benchmark
  clips for that claim's text and the cosine is positive. Top-5 (p < 0.05) was tried first
  and fails for a reason worth stating: the benchmark's clips are correlated (some twenty
  carry wind), so a top-5 rank among them is a test against siblings, not decoys. The cut was
  chosen on the empties-vs-`no_ambient` comparison and nothing else; sensitivity on the same
  log (667 claims, 81 clips with claims; "clock" = the audio LM's habitual hallucination):

  | gate | claims kept | clips left empty (of 81) | clock claims kept (of 59) |
  |---|---|---|---|
  | audio rank top-5 | 109 | 34 | 1 |
  | audio rank top-10 | 171 | 24 | 4 |
  | audio rank top-20 (shipping, with cos > 0) | 254 | 12 | 8 |
  | audio rank top-30 | 320 | 7 | 15 |
  | text mean + 2σ (first form) | 99 | 32 | 0 |
  | cos > 0.05 | 194 | 17 | 1 |

  Top-10 to top-30 all beat the 2σ bar on empties; the result is not knife-edge. Add the 19
  clips where the audio LM heard nothing to every row for the clip-level total.
- **The independent reference cannot score the gate decision.** Its "nothing beyond the
  picture" verdict agrees with the human tag on 55/100 clips (chance) under both visibility
  variants tried: list-and-match (Idefics3 list, one word per clip; MiniLM cosine) and a
  per-sound forced question to the same VLM. The cause is the rule "silent only if every
  verified claim is visible" meeting five to eight audio-LM claims per clip, several of which
  ("video game sound", "wind noise (microphone)", "clock ticking") are never visible. So it is
  biased toward "something is missing" and rewards the blind baseline on seen-ambient clips.
  Reported as such; the human-grounded reference remains the only one that scores the gate.
  Future work, a taxonomy decision rather than a tuned constant: claims that name a recording
  artefact (microphone wind noise, video-game sound, sound effect) should not count as
  missable, and near-duplicate claims of one source should be one claim.
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
- **Onset timing.** On 255 DCASE gold onsets (`benchmark/eval_dcase_onset.json`) hysteresis cuts
  MAE from 0.43 s to 0.31 s and matches 16 more events; the occlusion refinement adds nothing on
  average (MAE 0.34 s both ways, better on 25 / worse on 39, CI includes 0). The 7-onset
  hand-labelled number (+0.9 → +0.3 s) overstated it. Only 66 of 255 gold events are matched at
  the display bar: detector recall on 5 s clips of laughter, footsteps and domestic sounds is low.

## Detector (stage 4)

- **The detector bounds the headline result.** Reading the cached gate decisions
  (2026-09-14): of 11 test clips where a picture was due and none was shown, 7 never reached
  the display bar (ambient sound under Speech 0.84 / Music 0.81; textures at 0.22–0.31) and
  4 are label ambiguity (visible cars vs off-screen traffic); on the dev split ~80 of 100
  redundant pictures are confident phantoms (whale at a Christmas market, roaring cats at a
  helicopter arrival, electric toothbrush in an alarm clip) with no source anywhere. A
  clip-level tagger's confidence is not evidence that a sound is in the scene.
- **A CLAP second opinion does not work.** Calibrated on 255 DCASE gold events (family rank
  over 328 ontology families, declared bar: 95% recall at k ≤ 30): recall@30 = 50%, footsteps
  median rank 261. Same model verified claims at chance in the evaluation reference. Not shipped.
- **Source separation was assessed, not run.** Demucs removes speech only (4 of the 7 masked
  misses are under music) and keeping the higher of mix/residual scores can only add
  phantoms. First future-work item, with declared bars (report §Conclusion).

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
  that returned 0.0 for everything on a library upgrade; a renamed processor keyword
  silently ignored by transformers 5, so the audio LM answered "none" for 100 clips it never
  heard -- the code now asserts on the processor's output). Each is guarded now; a single
  automatic number deserves less trust than its precision suggests.
- **Knobs were set once on the dev split** after 2026-09-13 (`docs/plan_robustness.md`); the
  demos in the notes illustrate and do not tune.
