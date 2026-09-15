# Limitations

What the numbers in the report do and do not support. Seeded 2026-09-14 from
`docs/failure_catalogue.md` and the Limitations section of `docs/project_notes.tex`; the report's Limitations section mirrors this file. Numbers like #3 are entries in
`docs/failure_catalogue.md`. "Current rule" means the configuration v3 was scored with.

**In short.** The headline score is an automatic proxy, not a study with deaf viewers; it
was produced by one judge on 100 clips labelled by one person, and it cannot see when the
sound detector misses a sound. The detector, not the visibility decision, is what limits the
result, and six ways of improving it were measured and did not help. Everything below is the
detail behind those sentences.

## Evaluation

- **The proposal's reference was circular.** The model-derived reference (tags `v2`, `v3`) is
  written by the system's own VLM from the detector's own events, so it rewards whichever
  system repeats the detector. Found in review on 2026-09-14. An independent reference
  (result files ending in `_indep`; four models: Qwen2-Audio listens, CLAP checks each claim
  against the audio, Idefics3 looks at the frames, Phi-3.5 writes the sentence — none of
  them sees any system's output) was built to replace it but cannot score the gate decision (below);
  the human-grounded reference is the one that scores the gate, and all three are reported.
- **The filter that decides which of the audio model's claims survive was set once, on one
  comparison.** CLAP scores each claim against the clip's audio. The first filter (a claim was kept if its similarity score was more than two standard
  deviations above its average score against 64 random, unrelated AudioSet sound names) left
  45% of clips with an empty reference, although only 25% are tagged `no_ambient` — so it
  was discarding real sounds: a reference biased toward silence, which rewards the gated system's
  abstention — the mirror image of the circular bias the reference exists to remove. The current filter keeps a claim if this clip's audio ranks in the top 20 of the 100 benchmark
  clips for that claim's text and the cosine is positive. Top-5 (p < 0.05) was tried first
  and fails for a reason worth stating: the benchmark's clips are correlated (some twenty
  carry wind), so ranking a clip in the top 5 among them compares it with look-alikes, not with unrelated
  audio. The cut was
  chosen on the empties-vs-`no_ambient` comparison and nothing else; sensitivity on the same
  log (667 claims, 81 clips with claims; "clock" = a sound the audio model often claims to hear when it is absent):

  | gate | claims kept | clips left empty (of 81) | clock claims kept (of 59) |
  |---|---|---|---|
  | audio rank top-5 | 109 | 34 | 1 |
  | audio rank top-10 | 171 | 24 | 4 |
  | audio rank top-20 (current, with cos > 0) | 254 | 12 | 8 |
  | audio rank top-30 | 320 | 7 | 15 |
  | text mean + 2σ (first form) | 99 | 32 | 0 |
  | cos > 0.05 | 194 | 17 | 1 |

  Top-10 to top-30 all beat the 2σ bar on empties; the choice of cut does not change the outcome much. Add the 19
  clips where the audio LM heard nothing to every row for the clip-level total.
- **The independent reference cannot score the gate decision.** Its "nothing beyond the
  picture" verdict agrees with the human tag on 55/100 clips (chance) for both methods tried: (a) Idefics3 lists what is on screen and the list is matched to
  claims by word similarity (MiniLM); (b) the same VLM is asked a direct yes/no per sound. The cause is the rule "silent only if every
  verified claim is visible" meeting five to eight audio-LM claims per clip, several of which
  ("video game sound", "wind noise (microphone)", "clock ticking") are never visible. So it is
  biased toward "something is missing" and rewards the blind baseline on seen-ambient clips.
  Reported as such; the human-grounded reference remains the only one that scores the gate.
  Future work, a taxonomy decision rather than a tuned constant: claims that name a recording
  artefact (microphone wind noise, video-game sound, sound effect) should not count as
  missable, and near-duplicate claims of one source should be one claim.
- **The judge cannot see a detector miss.** On the 7 test clips tagged unseen/mixed where the
  detector heard nothing, the grounded reference reads "nothing beyond the picture" and every
  system scores 4. The clip-level cost-sensitivity analysis (report §Cost sensitivity) counts
  them as misses for all systems; the judge score does not.
- **The cost-sensitivity analysis is secondary and coarse.** Added after the test results were seen. The cost of a redundant picture relative to a
  missed one (r) is not fixed; results are shown for a range, and the gate is ahead once r
  exceeds 0.40. It scores only
  whether the gate opened at least once when it should have, not when, for which sound, or
  with what picture.
- **The independent reference's "does this sentence match the audio?" check is uncalibrated.**
  It uses a fixed similarity cut-off (MiniLM cosine 0.50) never tuned against ground truth; calibration on DCASE gold is planned but
  not done.
- **One annotator.** Re-labelling 60 clips without seeing the first labels agreed at κ = 0.60 (Cohen's kappa,
  "moderate"); most disagreement is on the "picture due" clips, the ones the gate is judged
  on.
- **One judge decides the headline.** A second judge reproduces the gated-vs-blind gap
  (κ = 0.753) but moves the captioning baseline by +0.61; rankings involving the text
  baseline are a property of the judge.
- **100 clips, 25 per category.** A per-category mean carries roughly ±0.2; read the confidence intervals (from paired resampling, `scripts/paired_stats.py`), not the
  bare averages.
- **No human study.** An LLM scoring a VLM's description of a picture is a proxy for a deaf
  viewer's comprehension; the correlation is unmeasured.
- **Onset timing.** On 255 DCASE gold onsets (`benchmark/eval_dcase_onset.json`) hysteresis cuts the mean timing error from 0.43 s to 0.31 s and matches 16 more events;
  the occlusion refinement adds nothing on average (0.34 s both ways, better on 25 / worse
  on 39; the confidence interval includes zero, so no real difference). The 7-onset
  hand-labelled number (+0.9 → +0.3 s) overstated it. Only 66 of 255 gold events are matched at
  the display bar: detector recall on 5 s clips of laughter, footsteps and domestic sounds is low.

## Detector (stage 4)

- **The detector bounds the headline result.** Reading the cached gate decisions
  (2026-09-14): of 11 test clips where a picture was due and none was shown, 7 never reached
  the display bar (ambient sound under Speech 0.84 / Music 0.81; steady background sounds such as traffic ("textures") scored only 0.22–0.31, below the
  0.35 bar) and
  4 are label ambiguity (visible cars vs off-screen traffic); on the dev split ~80 of 100
  redundant pictures are confident phantoms (whale at a Christmas market, roaring cats at a
  helicopter arrival, electric toothbrush in an alarm clip) with no source anywhere. A
  clip-level tagger's confidence is not evidence that a sound is in the scene.
- **A CLAP second opinion does not work.** CLAP ranked all 328 AudioSet sound families for each of 255 DCASE gold events; the
  declared bar was the true family in the top 30 for 95% of sounds. It reached 50%; for
  footsteps the true family was typically ranked 261st. Same model verified claims at chance in the evaluation reference. Not adopted.
- **Two-tagger agreement does not work either.** Keeping a BEATs detection only if PANNs
  (a second AudioSet tagger, different architecture) ranks the same family in its top 10
  for the same audio removes 24 of 77 dev phantoms (declared bar: 40) while losing 1 of 20
  unseen-clip pictures and 2 of 66 DCASE gold events. The phantoms are shared AudioSet
  confusions ("roaring cats" on a glacier hike heard by both), not one model's quirk.
  Not adopted (`benchmark/panns_agree.json`).
- **A persistence-plus-rank decision rule does not beat max-over-windows.** Replacing "max
  window score ≥ 0.35" by "top-3 non-speech class in every window of a ≥1 s run, score ≥
  0.15" (27-cell grid on DCASE gold) finds the same share of real sounds for the same number of false alarms as the current
  rule (21.6% found at 4.6 false alarms per minute vs 21.2% at 5.2); the declared +10-point recall
  bar failed. On dev it removes 40/77 phantoms and loses 1/19 unseen pictures — a trade
  along the same curve, not a better detector. Not adopted (`benchmark/persist_rank_setting.json`).
- **An audio language model as second opinion does not work either.** Qwen2-Audio-7B,
  given the clip's audio and BEATs' own candidates ("which of these do you hear?"): on a
  20-sound smoke test drawn beforehand it vetoed 6/10 phantoms and kept 9/10 real sounds
  (pass); on the full dev split it removed 25/77 phantoms (declared bar 40) and would also
  have removed real sounds on test (a baby crying, a basketball bounce). Often it simply repeats the candidate list instead of judging it. Not adopted (`benchmark/audio_llm_eval.json`).
- **Source separation was assessed, not run.** Demucs removes speech only (4 of the 7 masked
  misses are under music) and taking the higher confidence from the original and the speech-removed audio can only add
  false detections. First future-work item, with declared bars (report §Conclusion).

- **Vocabulary.** AudioSet has no "phone alert"; no model trained on it can name one
  (catalogue #3).
- **Masking by music and speech.** A siren under a film score is heard only at its close-up
  (#4); a tonal alert is mis-heard as a tuned instrument. Source separation inside the detector
  is the general remedy and is not yet in the current configuration.
- **Confident phantoms.** Dog 0.60 that nobody hears (#5), ice-cream truck at a station (#6),
  horse/train/truck at a quarry blast (#7), sheep in a jungle (#2). No threshold reaches them and
  the plausibility veto that would have caught some was turned off after 2 right / 2 wrong (#16).
- **Confident mislabels.** A door opening detected as "Gunshot" 0.50 with the right timing (#17).

## Visibility (stage 5)

- **Too reluctant to silence on-screen sources.** On DCASE 2025 gold (258 events): 97% of off-screen
  sounds are correctly kept, only 35% of on-screen sources are silenced (66.7% agreement).
  DCASE videos are 360° recordings, so "on screen" there means inside a chosen viewing
  window.
- **A larger VLM makes the visibility check worse under the rubric.** Qwen2.5-VL-32B on the
  same 258 DCASE gold events (same three-vote question): on-screen recall 67% (7B: 35%) but
  off-screen sounds wrongly silenced 25% (7B: 3%); agreement 70.9% vs 66.7%. Declared bar
  (≥50% on-screen AND ≤+2 pts off-screen) failed on the second condition; at the rubric's
  4:1 costs the 32B check roughly doubles the loss. Not swapped
  (`benchmark/eval_dcase_visibility_32b.json`).
- **Unstable to frame choice.** A 0.5 s shift in frame sampling flipped a gunshot and a police
  siren between silenced and shown (#13).
- **The 7B model tends to pick whichever answer is listed second ("position bias")**; every two-way question is asked in both orderings and
  agreement is required, which costs recall.

## Deduplication and depiction

- **Text similarity between short phrases is coarse** (SigLIP): insect buzzing merged into bird
  chirping at 0.82 (#14). Similarity merges now require time overlap; the coarse similarity
  remains.
- **Pictures are generated, not verified against the video**; a forced-choice check confirms
  the phrase reads as the sound, not that the rendered image does.

## Benchmark

- **274 clips, not 300.** Only 15–20% of candidate clips from each of four search methods were usable.
- **Some clips were pre-selected by a model.** The first 209 labelled clips were chosen
  without model help; later candidates were model-screened, so they are not a random sample.

## Process

- **The evaluation software produced plausible wrong numbers** several times (charitable score
  for an empty augmentation; a reference that could not say "nothing is missing"; a similarity
  that returned 0.0 for everything on a library upgrade; a renamed processor keyword
  silently ignored by transformers 5, so the audio LM answered "none" for 100 clips it never
  heard -- the code now asserts on the processor's output). Each is guarded now; a single
  automatic number deserves less trust than its precision suggests.
- **Settings were fixed once on the development split** after 2026-09-13
  (`docs/plan_robustness.md`); the demo clips are examples only and were not used to choose
  settings.
