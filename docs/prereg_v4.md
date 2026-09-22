# Pre-registration: v4 — the SOTA configuration, one swap at a time

*Committed 2026-09-18, before any v4 run. Decided with Adam after a Fable review and a live
check of the public leaderboards (arena.ai 7–13 Sept 2026, Artificial Analysis Sept 2026, HF
Open ASR). The supervisor's ask: the best models, a clear win over the baselines.*

## What v4 is

| stage | v3 (frozen, tag `v3-thesis-2026-09-17`) | v4 | why this one |
|---|---|---|---|
| 2 on-screen hints | OWLv2 (2023) | **SAM 3** (`facebook/sam3`, Nov 2025, text-prompted) | the recognised successor; context-only stage |
| 3 speech (context) | faster-whisper base | **Granite Speech 4.1-2B** (`ibm-granite/granite-speech-4.1-2b`, Apr 2026, WER 5.33, #1 Open ASR) | English assumed (6 languages; Whisper 99) |
| 4 sound detection | BEATs (2022), 527 classes | **FLAM** (Adobe, ICML 2025) with a descriptive query vocabulary and per-query calibration (§4) | the measured bottleneck; the only recognised frame-level open-vocabulary detector with released inference |
| 5 visibility + depiction + describer | Qwen2.5-VL-7B | **Qwen3.8-27B** (Aug 2026; Vision Arena open #4, 1244 ± 10) | already verified on an A100; 7B → 27B + thinking |
| 6 picture | FLUX.1-schnell (2024) | **Qwen-Image-2512** (Dec 2025, Apache-2.0; the only open model in the top 3 of both arenas) | quality; clean licence |
| 7 judge | Mistral-7B-Instruct (2024) | **Gemma-4-31B-it** (text mode; a different family from the Qwen describer and generator) | the describer and the generator are both Qwen in v4; the judge must not be |

Not in v4: MiniMax-H3 (video arm, "later", stays in the plan as the future section); DASM
(weights only, inference code unreleased — cited as a peer).

## Order and attribution

Four model swaps at once would make any gain unattributable. The swaps land in this order,
each with its own bar, on the frozen 100-clip test set and (when it exists) the gold set:

1. **+4 FLAM** → 20-clip check, then the full protocol (`v4a`)
2. **+5 Qwen3.8-27B** → 20-clip check, then full (`v4b`)
3. **+6 Qwen-Image-2512** → full (`v4c`)
4. **+3 Granite, +2 SAM 3** bundled last (`v4`) — context-only stages, expected Δ ≈ 0; run
   for freshness and reported as such

Each step is reported cumulatively (v3 → v4a → v4b → v4c → v4) and, at the end, as a
leave-one-out row per stage. The judge for every v4 row is Gemma-4-31B-it with Qwen3.8-27B
as describer; v3's panels are re-described and re-judged once under the same pair so the v3
row is comparable (the Mistral-judged v3 numbers stay in the thesis as the v3 chapter).

## The headline bar (what "significant improvement" means)

The claim is against the **baselines**, not against v3: under the same evaluation, the
gated system must beat the blind audio-to-image baseline. Pre-declared:

- **primary:** gated − blind judge score on the 100 test clips, paired bootstrap 95% CI
  excluding 0 and ≥ +0.25 on the 0–4 scale (v3: +0.01, a tie);
- **cost:** panel-on time and wasted seconds not worse than v3's ratio to blind (47% vs 60%);
- **gold set (when available):** per-sound "drawn / not drawn" accuracy of the gate vs the
  blind rule, and the human sentence as the reference.

If the primary bar fails after all swaps, the thesis says so with the numbers; the
cumulative table is the result either way.

## §4 FLAM, second attempt (the part that is a new experiment)

The first attempt (docs/prereg_flam.md) used the 527 bare AudioSet names as queries and
failed on precision (FP/min 693–2728 vs BEATs 5.2) while finding 70–85% of buried sounds.
Declared here, before the run:

- **Vocabulary:** a fixed list of ~85 descriptive queries ("a dog barking", "an emergency
  vehicle siren wailing"), each mapped to one AudioSet label so everything downstream
  (families, gate, dedup, depiction) is untouched. Written from the ontology and the
  pipeline's own families; five labels were added so that every *real* labelled detection
  on the dev split has a query (dev is the tuning split; disclosed). The list is
  `src/stage4_audio_event_detection/flam_queries.py` and does not change after the run.
- **Calibration (per query):** on DCASE 2025 task 3 **dev-train-tau** audio (600 five-second
  clips, one per mix and start time, never used for evaluation), the bar for query *q* is the
  lowest bar in {0.05, 0.10, …, 0.95} whose spans not overlapping a gold event of a matching
  class number ≤ 5.2 / |Q| per minute — the BEATs false-positive rate shared equally. A
  query that meets it at no bar gets 0.95. Scores are then rescaled piecewise-linearly so
  that the bar maps to `DISPLAY_THRESHOLD` (0.35) and 1 stays 1; the shipping span rule
  (reach the bar, extend through half of it, ≥ 0.5 s) runs unchanged on the rescaled scores.
- **Evaluation:** the same five bars as the first attempt, on **dev-test-tau** (the 255
  events) and the labelled dev detections:

| measure | BEATs | FLAM-v2 must reach |
|---|---|---|
| masked-event recall (under speech/music) | 9.5% | **≥ 24.5%** |
| clear-event recall | 32.6% | **≥ 32.6%** |
| false positives / min | 5.2 | **≤ 5.2** (a real test now, not by construction) |
| dev real detections still fired (of 23) | 23 | **≥ 21** |
| dev phantoms no longer fired (of 77) | 0 | **≥ 40** (a phantom whose label has no query counts as gone; disclosed) |

PASS iff all five. On pass, `AED_MODEL = "flam"` is v4a. On fail, attempt nine in the
table with the numbers, and v4 continues from stage 5 with BEATs.

## §5 Qwen3.8-27B — the visibility check already ran

docs/prereg_qwen38_visibility.md, thinking-off arm (job 30546271): on-screen recall
**56.3%** (bar ≥ 50% ✔), off-screen recall **82.6%** (bar ≥ 95% ✘) — the same trade the
32B made. The thinking arm (30546328) is still running. Whatever it says, the swap goes
ahead by decision (best models), the numbers are disclosed next to the 7B's, and no prompt
or vote rule is changed for it. The gold set's per-sound labels are the arbiter.

## Not done

No tuning on the test clips; no prompt changes after seeing numbers; the gate bar stays;
v3 stays in every table.

## §4 outcome (added 2026-09-18 evening, after the run) — FAILED as pre-registered

`benchmark/flam_v2_setting.json`, `benchmark/flam_calibration.json` (600 clips, 50 min;
median bar 0.53; 24 queries met the budget at no bar and were left at 0.95 as declared).

| measure | bar | FLAM-v2 |
|---|---|---|
| masked-event recall | ≥ 24.5% | **69.8%** ✔ |
| clear-event recall | ≥ 32.6% | **88.4%** ✔ |
| false positives / min | ≤ 5.2 | **61.6** ✘ |
| dev real detections kept | ≥ 21/23 | **15/23** ✘ |
| dev phantoms gone | ≥ 40/77 | **59/77** ✔ |

Diagnosis (exploratory, after the result): the false positives are impact and human
generics — Knock 5.9/min, Footsteps 4.6, Clapping 4.0, Doorbell 3.4, Cough 3.1, Bicycle
bell 3.1, Door 3.0, Bell 2.8, Rain 2.5 — and the calibration did not transfer from
dev-train to the busier, event-selected dev-test clips (with the 24 ceiling queries off the
rate would still be 34/min). Of the 8 lost dev reals, 4 had FLAM score ≈ 0 in the span
(Aircraft, Gunshot, Explosion, Zipper: FLAM has blind spots of its own), 3 sat under a
ceiling bar, 1 was marginal. BEATs' 5.2/min was never measured at matched recall, so the
bar is asymmetric; disclosed. Recorded as attempt nine. Two attempts say the same thing:
FLAM hears under masking what BEATs misses, with an unusable false-positive rate on
generic impacts — complementary, not a substitute.

## Added 2026-09-18 night (before any number): gold slice B and the detector table on it

Slice B = AudioSet-Strong *eval* clips with a consequential sound at least half covered by
speech or music (`benchmark/gold/audioset_slice.py`: fixed vocabulary of consequential
families, ≤ 12 clips per family, seed 7; clips gone from YouTube are listed). Human-timed
labels are the pre-fill; annotators add only visibility / "draw?" and the sentence. It is
reported separately from the 100-clip slice, never pooled, and is used for **detector**
measures only. `benchmark/audioset_detector_eval.py` gives one table for BEATs (v3),
PretrainedSED (v4 candidate) and FLAM-v2 (the failed attempt) on the same clips: recall of
masked consequential events, of all consequential events, of all events; false spans per
minute (every sound in these clips is labelled, so this is a real false-alarm rate); onset
error. No pass bar: it is the held-out check that the DCASE-chosen detector setting
transfers to real-world video. The PretrainedSED decision stays with docs/prereg_psed.md.

## Added 2026-09-19 00:40, after the PretrainedSED result and before any protocol score: the detector arm

PretrainedSED failed its eighth bar (docs/prereg_psed.md) and per that pre-registration
**BEATs stays the v4 detector**. Observed after the fact: that bar's population is BEATs'
own detections judged real by a human (23 items), so it measures agreement with BEATs
rather than recall of ground truth; on gold slice B (declared above, no pass bar, 111
held-out real-world clips, 236 masked consequential events) PSED was ahead of BEATs on
every measure (masked-consequential recall 62.7% vs 50.0%, all events 53.0% vs 37.9%, false
spans 2.6 vs 6.8 per minute, onset MAE 1.14 vs 1.48 s). Because this was seen after the
result it does not alter the v4 decision. Instead a second comparison is declared here,
with its rule fixed before any protocol score exists:

- rows **v4a** = PSED + Qwen2.5-VL-7B and **v4ab** = PSED + Qwen3.8-27B are run through the
  protocol next to v3_q38 (BEATs + 7B) and v4b (BEATs + Qwen3.8), same describer and judge;
- **PSED is adopted for v4 iff** the gated score of v4ab exceeds that of v4b (mean paired
  difference > 0) **and** the 95% paired bootstrap CI of that difference does not lie
  entirely below 0; an exact tie or a negative mean keeps BEATs;
- the dev-real bar is retired for future detector attempts in favour of slice B's
  exhaustive labels (a design correction, not a retroactive waiver).

**Outcome (2026-09-20 09:00, job 30723676 render/describe on A100 + 30724662 judge on L4; PSED at
the calibrated bar 0.15, `benchmark/protocol_results_v4ab_grounded_rubric.json`):** gated
v4ab **2.60** vs v4b **2.85**; paired mean difference **−0.25**, 95 % bootstrap CI
[−0.53, −0.01] — negative mean, CI entirely below 0 → **by the rule PSED is NOT adopted;
BEATs stays the v4 detector.** All three systems drop under PSED-0.15 (blind 2.46 vs 2.75,
caption 2.46 vs 2.66): at 0.15 PSED fires far more often (5.4 false spans/min on the
calibration set against BEATs' target rate), so every system shows more pictures and
captions the judge counts as redundant or wrong; gated − blind is +0.14 [−0.13, +0.40],
the widest positive gap of any row, but the pre-registered test is the absolute gated score.
Per scenario (proposed): acoustic-event 2.44 vs 2.96, ambient 2.72 vs 2.88, mixed 2.52 vs
2.68. The discarded wrong-bar run (PSED at 0.35, 2.88) is kept in the table only as a
labelled sensitivity row: a stricter bar recovers the score, which says the detector's
*operating point* for the pipeline is not the one that maximises recall on slice B. The
per-sound metric (F1-strict on Adam's annotations, both gold sets) is the second, pre-declared
view of the same question and is still to come.

### Post-hoc diagnosis of the v4ab drop (2026-09-20 09:30, read on the benchmark outputs — disclosed) and DRAFT row v4ab2 (not run; awaits Adam's approval and the per-sound F1 gate)

The first explanation ("PSED fires more") is false: on the calibration set BEATs@0.35 fires
15.0 spans/min, PSED@0.15 12.3/min (`benchmark/psed_span_match.json`). The outputs show two
*integration* defects instead, both BEATs-specific assumptions the pipeline made:
1. **Vocabulary leak.** The "salient non-speech" filter (`src/labels.py`) lists BEATs' 527
   label names; PSED's AudioSet-Strong names slip through it. Blind-system label counts,
   v4ab vs v4b: "Breathing" 17 vs 0, "Video game sound" 15 vs 0, "Human voice" 8 vs 0,
   "Laughter" 6, "Footsteps" 12 — human/vocal and meta classes the pipeline should never
   depict. 2.33 events per clip vs 2.01.
2. **Span rule.** Median picture span 10.0 s under PSED vs 4.0–5.7 s under BEATs (spans like
   [0, 16 s]): onset refinement is off for PSED and the hysteresis low bar (bar/2 = 0.075)
   lets a span run through the clip, so pictures stay up far longer — the rubric judge counts
   that as redundant.

**Draft v4ab2 (Fable-reviewed; to be fixed on the calibration set only, applied to BOTH
detectors, and re-tested under the unchanged adoption rule; v4ab stays in the table as the
failed row):**
- Filter by ontology *branch*, not by class name: exclude the AudioSet branches
  "Human voice" (speech, shouting, laughter, singing …), "Respiratory sounds" (breathing,
  cough, sneeze), "Channel, environment and background" and the meta-class branch that
  holds "Video game sound" / "Sound effect". **Open question for Adam:** "Crying, sobbing"
  and "Baby cry" sit under Human voice; footsteps under Human locomotion. Option A excludes
  the whole Human voice branch (loses baby cry); Option B keeps the Crying subtree and
  Human locomotion (declared exceptions). Recommended: B.
- Span rule, detector-agnostic: hysteresis low bar = bar (no extension below the bar); a
  span ends after 1.0 s continuously below the bar; hard cap = the 90th percentile of BEATs'
  span length on the calibration set (value to be computed and written here before the run).
  ONSET_CAM stays off for PSED; bar 0.15 unchanged; nothing else.
- If the filter changes BEATs' output, v4b is re-rendered under the same filter as the
  control (same clips, same judge), otherwise the comparison is confounded.
- Gate before running: the per-sound F1-strict on Adam's annotations. v4ab2 is run only if
  PSED's F1-strict ≥ BEATs' on the filtered class set on the 100-clip benchmark; if PSED is
  weaker on the sounds that matter, integration fixes cannot rescue it and the row is closed.
- Thesis wording: "The pre-registered detector swap failed (v4ab). Inspection of outputs,
  disclosed as post hoc, revealed vocabulary coupling: the salient-sound filter and the span
  rule assumed the BEATs label set and clip-level scores. Both were generalised to
  detector-agnostic rules, fixed on the calibration set, and the swap re-tested as v4ab2
  under the original adoption rule." v4ab and v4ab2 reported side by side.

### v4ab2 / v4b2 — DECLARED 2026-09-20 12:30 (Adam approved; before any run)

Per-clip attribution of the v4ab drop (`benchmark/aug_summary_v4b_v4ab.json`): clips with a
leaked-label picture fell by −0.44 on average against −0.15 for the others (proposed system;
34 vs 66 clips), so the vocabulary leak carries most of the loss; picture length does NOT
show an effect (median span ≥ 8 s: −0.23; < 8 s: −0.27), so the span-rule fix from the draft
is dropped — the hysteresis rule (low bar = bar/2) is already the same for both detectors.
A third pattern, PSED drawing "Vehicle" on city-walk clips the human tagged as needing
nothing, is an operating-point question and is what the per-sound metric will measure.

Three-Fable panel verdict on "does the evaluation favour the old models?": partly — through
integration glue shaped around BEATs (hand label lists, the bar inherited from BEATs'
false-alarm rate, SAM 3 at its default bar with OWLv2's phrases) and small n, not by design.

**Two fixes, both detector-neutral, both applied to both detectors; nothing tuned on the
benchmark:**
1. Label filter = speech, music, and the AudioSet branch "Channel, environment and
   background" (+ "Silence", + "Sound effect" — Adam, 20 Sept 13:00, an edited-in sound with
   no source; neither detector ever emitted it on the benchmark) only
   (`config.LABEL_FILTER = "branch"`, V4 stage "8"). Adam's
   rule (20 Sept): "only speech and music we don't draw"; the branch was added after checking
   that no important gold sound in slice B or the calibration set falls inside it. Note:
   "Wind" becomes drawable under this rule (Adam: leave it, decide later).
2. PSED's bar = the argmax of span-level F1 on the AudioSet-Strong calibration set
   (`scripts/psed_f1_bar.py` → `benchmark/psed_f1_bar.json`): **0.22** (P 0.78, R 0.62,
   F1 0.694; BEATs' best on the same set is 0.580 at 0.24). BEATs keeps its shipped 0.35.

Rows: **v4ab2** = PSED@0.22 + Qwen3.8 + filter (STAGES=458), **v4b2** = BEATs@0.35 + Qwen3.8
+ filter (STAGES=58) — the fair control, re-rendered under the same filter. Same describer,
same Mistral judge, rubric-enforced. Adoption rule unchanged: PSED is adopted iff gated
v4ab2 > gated v4b2 with the paired CI not entirely below 0. v4ab stays in the table as the
failed first row. The per-sound F1-strict on Adam's annotations is the second view.

Stage-2 addendum: at a bar chosen on the DCASE dev set SAM 3 reaches 72.4 % agreement
(bar 0.16) and OWLv2 74.0 % (bar 0.15); at equal on-screen recall to OWLv2@0.20, SAM 3
(bar 0.16) 72.4 % vs 68.8 %. Read: a tie within noise; the default-bar comparison above
favoured OWLv2 through the bar, not the model. The swap stays dropped for lack of a gain,
not because SAM 3 loses.

**Outcome of v4ab2 / v4b2 (2026-09-20 22:10; jobs 30726600/30726776 and 30726601/30726777; Mistral judge, rubric-enforced):**

| row | gated | blind | caption | gated − blind |
|---|---|---|---|---|
| v4b2 (BEATs 0.35 + branch filter) | 2.80 | 2.60 | 2.65 | +0.20 [−0.03, +0.41] |
| v4ab2 (PSED 0.22 + branch filter) | 2.35 | 2.24 | 2.50 | +0.11 [−0.14, +0.33] |

Detector-arm test: gated v4ab2 − v4b2 = **−0.45 [−0.69, −0.22]** → **PSED is not adopted;
BEATs stays** (second time, now under detector-neutral rules). Every system drops (blind
−0.36, caption −0.15); per scenario the gated system loses most on the acoustic-event clips
(2.04 vs 2.88), the very clips where a picture is due.

What the outputs show (`benchmark/aug_summary_v4b2_v4ab2.json`): under the loose filter
PSED's most frequent pictures are "Generic impact sounds" (29), "Mechanisms" (29), "Wind"
(27), "Wind noise (microphone)" (12), "Video game sound" (11), "Breathing", "Laughter" —
meta, noise and vocal classes the old hand lists used to block — at a median span of 16 s
(BEATs 5.2 s), 2.16 pictures per clip against 1.25. PSED's AudioSet-Strong head fires these
coarse classes on ambient audio where BEATs' clip-level tagger stays quiet.

A confound found while reading the judgements, disclosed here: the judge's *reference*
("what a hearing viewer gets that a deaf viewer misses") is model-derived — it names what
the row's own stage-4 heard, corrected only by the human clip tag (see
`grounded_reference`). A noisier detector therefore writes a noisier answer sheet for its
own row (e.g. the v4ab2 reference for the golf course asks for "generic impacts and
breathing"). The two rows were judged against different references, so the −0.45 is not a
clean measure of the detector. Two remedies, both pre-declared here before running:
(1) a same-answer-sheet check — each row's cached descriptions judged against the OTHER
row's grounded references (`slurm/job_xref_judge.sh`, `--ref-tag`), reported beside;
(2) the per-sound F1-strict against Adam's annotations (independent of every detector),
which is the decisive test of the detector question. The independent reference of v2
(`protocol_reference_indep_v2*.json`) is not used because it was found at chance on silence.

**Same-answer-sheet re-judge outcome (2026-09-21 00:40, job 30785966):** swapping the
references changes nothing — v4ab2 judged against BEATs' references: gated 2.33 (own: 2.35);
v4b2 judged against PSED's references: 2.82 (own: 2.80). Gated PSED − BEATs on BEATs'
sheet −0.47 [−0.72, −0.23], on PSED's sheet −0.47 [−0.70, −0.26]. So the model-derived
reference is **not** what separates the rows; the judge is reacting to the pictures and
descriptions themselves (undrawable texture labels, whole-clip spans). The confound is
disclosed but does not rescue PSED.

### v4ab3 / v4b3 — DECLARED 2026-09-21 00:30 (Adam approved; before any run): the ten-Fable panel's decisive test

Ten independent Fables (domain shift, filter mismatch, span length, judge reference,
statistics, plumbing audit, gate/painter, practitioner, per-sound expectation, devil's
advocate for PSED) converged: the detector-only evals scored PSED *after* the old hand list
removed its texture classes, while the pipeline rows drew them; on the benchmark 40–48 % of
PSED's events are Wind / Generic impact sounds / Mechanisms / Wind noise / Breathing / Video
game sound, 56 % of its events span the whole 20–28-s clip (BEATs 24 %; on 10-s clips this
could not show). No plumbing bug (rescale and hysteresis identical in both evals).
Principle to state: a picture channel wants a conservative, object-level nominator; recall
of ambient texture is a liability.

**Rows** (same describer, Mistral judge, rubric-enforced; STAGES 459 / 59):
- **v4ab3** = PSED@0.22 + Qwen3.8 + depictable filter + 8-s cap;
- **v4b3** = BEATs@0.35 + Qwen3.8 + depictable filter + 8-s cap (fair control).

**Depictable filter** (`config.LABEL_FILTER = "depictable"`, applied to both): the "branch"
rule plus, by ontology branch and category name only (no single-class picks): the Wind
subtree (texture), Respiratory sounds, the whole Source-ambiguous branch (Generic impact
sounds, Onomatopoeia, Bang …), Human voice except the Crying subtree (baby cry stays — Adam's
option B), the category names of GENERIC_LABELS (Animal, Mechanisms, Sounds of things,
Domestic sounds, Human sounds …), "Video game sound". **Span cap** (`config.MAX_SPAN = 8.0`):
an event never runs longer than 8 s from its onset, both detectors. Bars unchanged (0.22 /
0.35). Adoption rule unchanged: PSED is adopted iff gated v4ab3 > gated v4b3 with the paired
CI not entirely below 0. v4ab, v4ab2 stay in the table. Same-answer-sheet re-judge and the
per-sound F1 on Adam's annotations are reported beside; the per-sound F1 remains the
detector-independent verdict.

**Outcome v4ab3 / v4b3 (first run, filter as declared above; 100 clips, Mistral judge, rubric-enforced):**
v4b3 gated 2.96 / blind 2.82 / caption 2.67 (gated − blind +0.14 [−0.07, +0.34]); v4ab3 gated
2.65 / blind 2.71 / caption 2.71 (−0.06 [−0.32, +0.19]). Paired gated PSED − BEATs = **−0.31
[−0.61, +0.01]** (v4ab2: −0.45): the filter + cap helped both detectors, BEATs more. Per scenario
gated PSED vs BEATs: acoustic 2.28 vs 3.04, ambient 3.16 vs 2.88, mixed 2.00 vs 3.04. By the
rule PSED is **not adopted**; v4b3 is the best row so far. The per-sound F1 on Adam's
annotations stays the detector-independent verdict.

**Amendment 2026-09-21 (Adam, after seeing the filter drop laughter): "human sounds are
drawn; only speaking is not."** The depictable filter is changed *before* the per-sound
scoring and the rows are re-rendered on the affected clips only (stamps of the other clips
stay; same seeds, same judge). New rule (`src/labels.py`, mode "depictable"): the Human voice
and Respiratory subtrees are drawable (laughter, cough, sneeze, gasp, sigh, snoring, whoop
…); still blocked: the speech labels (incl. Shout / Screaming / Yell, in `SPEECH_LABELS` since
v1 — open question for Adam), music (singing, humming, whistling count as music), the bare
category names "Human voice" / "Respiratory sounds", and **"Breathing"** (texture; PSED fired
it 17× on wind-like audio in v4ab; Adam: "Breathing out"). What the first-run filter had
removed from the two rows' detections: PSED — Breathing 17, "Human voice" 8, Laughter 8,
Sneeze 1, Whoop 1; BEATs — Snicker 2, Sigh 1, Gasp 1. So the re-render touches ≈10 PSED clips
and ≈4 BEATs clips. Both scores (first run above; re-run below) are kept in the record; the
re-run is the row in the table. Adam's clip-level tags and the judge set (the same 100 clips)
are unchanged.

**Amendment 2 — 2026-09-21 12:00 (Adam + three Fable panels; declared before the re-run, which
waits for the full per-sound annotation).** The examiner Fables (10-panel) ruled that the first-run
v4ab3 / v4b3 scores above stay the **confirmatory** rows and adoption verdict; the re-render under
the amended filter is reported as a **labelled sensitivity row**, not as a replacement (the judge
scores each clip on its own, so the untouched clips are unchanged by construction). Filter rule
(`src/labels.py`, mode "depictable"), the same for every detector: *draw a label only if a
captioner would write it as a bracket tag and it names a source one can picture.* In order:
(1) steady textures are out on the raw name — Wind subtree, Breathing, Rumble, Hum, Whir, Rustle;
(2) the label is mapped to its family first (Bang → Explosion, Beep → Alarm, Smash/Breaking →
Glass, Clickety-clack → Train, Ding/Ringing → Bell, Sizzle → Cooking, Creak → Door) — the first
run applied the filter *before* the map, so those mappings were dead code and a gunshot heard as
"Bang" was silently dropped (bug, found by the ML-engineer Fable); (3) on the mapped name: speech
with words (Speech, Whispering, Chatter, Hubbub and subtrees) and music (incl. singing, humming,
whistling) are never drawn, nor the "Channel, environment and background" branch, Silence, Sound
effect, Video game sound, the bare category names, and any Source-ambiguous label left unmapped
(Clatter, Scrape, Screech, Twang …). Non-word vocal events — Shout, Yell, Screaming, Children
shouting (moved out of `SPEECH_LABELS`), Laughter, Cough, Sneeze, Snoring, Gasp, Crying — are
drawn (Adam: "useful to DHH when not heard, just like laugh"; SDH tags [screaming]). Blocked
labels: 239 of 625 (first run: 281). Not changed: bars, cap, gate, judge, clip set, clip tags.

**Amendment 3 — 2026-09-21 evening (engineering corrections; Adam: "fix A–C now, rerun after I
export"). Ten Fable bug-hunters (one per stage + three integration reviewers) found thirteen
silent defects; a further Fable checked the fix plan before coding and one reviewed the code
after. Fixed, with the effect on the first-run rows:**
- *Judge.* (A1) The reference prompt received Stage 2's whole-clip visible list and offered the
  sentinel "nothing beyond the picture" — the gate is graded by its own input. On 9 of the 50
  unseen/mixed clips of v4b3 the gated system was silent and scored 4; re-scored, gated − blind
  falls from +0.14 to +0.02. Now the prompt only names what the sounds tell; visibility is settled
  by the human clip tag alone; on unseen/mixed clips where the detector heard nothing the
  reference is a fixed sentence ("An off-screen sound matters here; its source is not known") so
  silence scores 0. Disclosed: on mixed clips the reference now includes the visible sounds too.
  (A2) `--ref-tag` was never read by the judge phase: the two `*_xref_*` "same-answer-sheet" rows
  above are **void** (each row was judged against its own references). Implemented now. (A3)
  ASR emitted "Music"/"You" as text on music-only clips, so the reference said "someone is
  speaking"; a stoplist removes such tags. Sentinel test is now exact in the judge as in the cap.
- *Spans.* (B1) Hysteresis halved `AED_THRESHOLD` (0.05 → 0.025) instead of the display bar, so a
  span grew from the noise floor, began near 0 s and the 8-s cap then removed the loud part. Now
  one pass at bar/2 = 0.175 (`config.AED_THRESHOLD`), as lines 182/217 declare. (B2) The cap was
  applied per raw span and undone by the family merge and the display chain; it is now also
  enforced on the displayed picture. (B3) The display chain measured the gap between repeats from
  the stretched end (raw end + 1.5 s dwell), so barks 2.2 s apart became one picture and every
  later onset was unmatched; the gap is now from the real end, and `MERGE_GAP` = 2.0 s = the
  annotation split rule (0.8 s reported as sensitivity). (B4) The speech-rescue band was dead
  since v3 (everything below the bar was dropped before the specs existed); marginal families
  without a strong firing now reach the gate. (B5) The "kind of a visible source" rule ignored
  time; it now requires overlap. (B6) The panel's overflow re-pack dropped quiet sounds that
  overlapped nothing. Not changed (design): the VLM is asked about the family ("Vehicle"), not
  the detail ("horn").
- *Per-sound scorer.* Families at ontology depth 1 (Vehicle, Water, Alarm, Explosion, Glass)
  and the six hand-written family names (Footsteps, Gunshot, Boat, Cattle, Dishes, Cooking)
  never matched; free-text gold names are resolved to ontology names; a second same-family
  sound covered by one picture counted as a miss; the caption baseline was scored on its
  strongest burst only; pictures the panel never drew (row overflow) were counted; placeholder
  panels are counted as shown and reported.
- *Plan.* Both rows are re-run **from scratch** under new tags **v4b4 / v4ab4** (same bars,
  filter of amendment 2, cap 8 s) after the annotation export; the first-run v4b3/v4ab3 rows
  stay in the record as "pre-fix". The adoption rule and the per-sound F1 as the
  detector-independent verdict are unchanged. Latent items left as limitations: late audio
  streams lose their offset (≤ 0.13 s in this data), PSED frame cache is not invalidated on clip
  change, ~14 PSED names unmapped, slice-B clips carry a fixed "unseen" tag.

**Amendment 4 — 2026-09-22 (importance rule; declared before the re-run and before the
re-rate).** The tagging note "seen sounds: put importance 1" made the mixed class impossible
(a dog on camera barking + a siren off screen came out "unseen"), because importance was a
function of the screen while visible/obvious already record the screen. New rule (Adam + two
Fables, two rounds): importance is a property of the sound (1 steady noise of the place, 2 an
event you can say in one sentence, 3 danger or a key story moment; unsure → lower). Score:
needed sounds rated 1 are don't-care (no onset → the onset rule cannot judge them; a matching
picture is absorbed); headline F1-strict unweighted over needed sounds rated 2–3; any picture
of a visible/obvious sound is a false alarm of weight 1; weighted F1 (2/3) a side column with
n(level 3); the earlier rule reported as a sensitivity row (`--old-rule`). Adam re-rates the
visible sounds he had set to 1 under the old note (tool filter "re-rate") and off-screen
textures he had set to 2. The category split of the 100 clips may shift; the judge set stays
frozen; per-sound rows are reported per category.

**Amendment 5 — 2026-09-22 13:00 (the gold set and the selection protocol; declared after the
annotation export was received and before any render finished or any per-sound score was read;
two Fables × two rounds, converged).**
- *Gold.* `benchmark/gold/annotations/gold_AG.json` (export 2026-09-22 09:23 UTC, one annotator):
  139 usable clips (done, not marked bad) = **103 benchmark clips** (54 of the frozen judge set,
  49 tagged afterwards — some found by the VLM-only pre-screen, disclosed) + **36 slice-B clips**
  (AudioSet-Strong, human-timed labels). 283 sounds; 132 needed sounds rated 2–3 (38 rated 3) in
  73 clips; 117 visible/obvious sounds rated 2–3; 44 sounds rated 1 (don't-care). Free-text names
  that no ontology name contains (25 rows) are mapped by a fixed alias table in the scorer
  (`ALIASES`), written before scoring. Gold rows whose label the depictable filter blocks
  ("Sound effect", "Generic impact sounds", "Drum", "Wind", "Mechanisms") are neither hit nor miss
  (12 rows, 4 of them needed ≥ 2); counted and disclosed.
- *Rows.* v4b4 = the declared configuration (BEATs, display bar 0.35, Qwen3.8-27B gate, majority
  of 3 votes, depictable filter of amendment 2, 8-s cap, FLUX) rendered **for real** on all 139
  clips for proposed / blind_a2i / audio_caption (sharded over GPUs, `slurm/job_gold.sh`); SILENCE
  = zero pictures, no render. v4ab4 = the PSED detector arm, same everything, rendered in parallel
  (arm rule of 2026-09-19 unchanged: adopt iff gated ΔF1 > 0 with the clip-bootstrap CI not below
  0). The gated system is not derived from the blind row by masking (stretch cuts, scene-shaped
  prompts and panel packing differ); the render logs the raw gate votes per sound per stretch
  (`gate_votes.json`) so the silence rule can be re-decided on CPU afterwards.
- *Headline.* Unweighted F1-strict over needed sounds rated 2–3 on the **103 benchmark clips**,
  proposed vs blind_a2i, with the **paired** clip-bootstrap CI of ΔF1 (same resampled clips for
  both systems, 2000 draws, seed 0); SILENCE and audio_caption rows beside. Pre-declared
  breakdowns: old-54 / new-49 / VLM-pre-screened subset; by category (mixed, unseen, seen,
  no-ambient); slice B (36) as the external check; all 139 pooled as one supplementary line;
  weighted F1, `--old-rule`, F1-phantom, late windows 1 s / 3 s, coverage, collisions, clean-clip
  accuracy as sensitivity rows; the LLM judge (Mistral, rubric-enforced) as the secondary metric.
- *Per-stage tests and knob selection (Adam: "choose the models that work best for our data").*
  DEV = the 54 old judge clips (every earlier choice already touched them); TEST = the 85 new
  clips, opened once. (4) Detector: BEATs vs PSED × display bar {0.25, 0.30, 0.35, 0.40} scored at
  detector level (recall of needed gold sounds at the onset window, false alarms per clip) on DEV
  + slice B as a diagnostic; the row-level verdict stays the arm rule. (5) Gate: per-sound
  visibility accuracy of the logged votes against the annotator's visible/obvious ticks, rule
  {majority, unanimous} × kinds {on, off}, dry-scored on DEV only. Filter and cap: not tuned. A
  setting replaces the declared one **only if** its DEV gain in needed-sound F1 exceeds the DEV
  bootstrap half-width and survives a 5-fold cross-fit over the 139; the frozen choice is committed
  before TEST is scored; the tuned variant is reported as a secondary row, never as the headline.
- *Selection hygiene.* No TEST clip is scored twice under different settings; the pre-registered
  row is the headline whatever the tuned row shows.

**Correction to amendment 5 — 2026-09-22 15:10 (before any TEST number was read; arithmetic and one
scorer line, not the protocol).** (a) The population counts above were taken over done clips
including the 20 marked bad. Correct: 139 usable = **109 benchmark clips** (49 DEV = usable old
judge clips; 60 TEST) + **30 slice B** (the AudioSet-Strong ids of `audioset_slice.json`).
(b) The scorer read only the *obvious* tick for "needed"; the declared rule (amendment 4, metric
doc, the tool) is needed = not visible **and** not obvious. 22 rows were visible-only; fixed in
`load_gold` before scoring. Corrected gold: **110 needed sounds rated 2–3 (28 rated 3) in 64
clips; 139 visible/obvious rated 2–3**; by category 32 mixed / 33 unseen / 44 seen / 30
no-ambient; DEV 36 needed, TEST-bench 43, slice B 31. (c) The PSED dry-run grid is PSED's raw bar
{0.10, 0.15, 0.20, 0.25, 0.30} mapped onto display bar 0.35 (the arm's own mechanism), because
the display bar cannot move PSED's operating point. (d) Scorer subsets are named without counts
(`bench`, `dev`, `test`, `test_bench`, `sliceB`); each system is scored on the clips it has
rendered and the paired ΔF1 on the intersection, with the clip counts printed.

**Re-run outcome (v4b4 / v4ab4), 2026-09-22 17:45 — full detail in `docs/GOLD_RERUN_2026-09-22.md`.**
v4b4 on the 109 benchmark clips (79 needed sounds), gated vs blind, paired clip bootstrap:
ΔF1 **+0.028 [−0.022, +0.074]** (primary, null) · ΔP +0.062 [+0.016, +0.109] · ΔR −0.076
[−0.141, −0.026] · ΔFA/clip −0.55 [−0.78, −0.37] · Δ clean-clip accuracy +0.172 [+0.082, +0.273] ·
ΔF0.5 +0.052 [+0.006, +0.098] (declared secondary, not promoted) · ΔwF1 +0.006 (null). Versus
SILENCE ΔF1 +0.290 [+0.199, +0.377]; versus CAPTION +0.031 [−0.019, +0.078]. DEV (49) and TEST (60)
agree; slice B shows the same false-alarm drop and no F1 gain. Judge (grounded + rubric-enforced):
2.68 vs 2.70, paired −0.014 [−0.223, +0.194]; by category +0.68 on seen-only clips and −0.55 on
unseen clips. **Detector arm: PSED not adopted** — gated PSED − gated BEATs ΔF1 −0.007
[−0.115, +0.098] on the benchmark clips (TEST −0.063, slice B +0.043). **Gate rule unchanged**
(majority of 3 best on DEV; Qwen3.8-27B ≥ Qwen2.5-VL-7B ≫ OWLv2 at chance). Conclusion as declared:
the primary endpoint is null; the gate's effect is a significant precision / false-alarm / silence
gain against a significant recall loss.

## Judge v4 (rubric-enforced) — declared 2026-09-19 01:50, before any 100-clip v4 score

Observed on v3 (100 clips) and the v4b 20-clip check: on clips tagged seen / no-ambient
(grounded reference "nothing beyond the picture") the LLM judge gives a *redundant* picture
4 on most clips (blind 3.80 mean on the 10 no-due clips of the check), the same as correct
silence. The protocol already codes one side of the declared rubric (docs/plan_robustness.md,
2026-09-14: wrong silence costs 4, redundant picture ~1) — an empty panel on a no-due clip
is 4 in code — but not the other. This completes it, in code, not in the prompt:

> For clips whose grounded reference is "nothing beyond the picture": empty panel = 4
> (unchanged); non-empty panel = min(LLM score, 2), applied in code after the LLM call.
> Caption row: empty caption = 4; a caption naming any sound = min(score, 2). All other
> clips: unchanged. Applied identically to gated, blind, caption and the oracle, and
> retroactively to v3 (100 clips) and v4b (20 clips); the uncapped judge is reported as
> "permissive judge" beside it on every row. Primary score for v4 = rubric-enforced judge.
> The gate's dev sweep uses the same asymmetry (redundant picture −2, missed picture −4).
> No further metric change after this declaration.

Risk, stated: the cap is exactly the lever that separates gated from blind and it is
declared after numbers that favoured blind. Defence: the 09-14 written asymmetry, the
pre-existing one-sided code rule, both judges on every row, and the judge-free check —
the gold set's per-sound "draw?" accuracy — which must agree.

## Correction, 2026-09-19 morning: the PSED rows ran at the wrong bar

The v4a/v4ab rows of the night used PSED's scores against the pipeline default bar 0.35,
not PSED's own bar 0.20 (chosen on DCASE by the matched-false-alarm rule before slice B or
the test clips were scored). On slice B PSED at 0.35 recalls 44% of masked consequential
sounds (BEATs 50%) — the misses on "picture needed" clips came from this. The rows are kept
under the tag `v4ab_bar035` as a record and **discarded**; PSED's scores are now rescaled so
that 0.20 lands on 0.35 (`psed_infer.rescale`), and v4ab is re-run. Fable's review: keep
the DCASE-chosen 0.20 (rule declared first); the slice-B sweep (0.05–0.50) is reported as a
held-out check, not used for selection.

## Hardware note (2026-09-19)

Runs are placed on whatever GPUs are free (Adam's rule): a 27B model may be sharded across
3× L4 (24 GB) with `device_map="auto"` instead of one A100-80. Fable's review: same
experiment in design (same weights, bf16, code, prompts, greedy decoding); not bit-identical
across GPU types (different bf16 kernels → rare token flips on near-ties), no systematic
bias; reported CIs cover it. Disclosure for the thesis: "Inference ran on mixed NVIDIA
hardware (A100-80 / 3× L4 sharded); bf16 numerics differ slightly across GPU types, so
outputs are reproducible in distribution, not bit-for-bit." Each row's log records
torch/CUDA/GPU. The PSED row (v4ab) runs on 3× L4.

## Calibration on AudioSet-Strong (declared 2026-09-19 before any number on the calibration set)

Adam: "why DCASE and not AudioSet-Strong?" DCASE is synthetic indoor audio; it was kept only
because the threshold rule was written before slice B existed. From now on every detector's
bar is chosen on a **calibration set** = a random sample of ~320 AudioSet-Strong *eval* clips
(seed 11), disjoint from slice B's 150 ids, not filtered (ordinary YouTube audio), by the same
rule: the loosest bar in {0.05 ... 0.95} whose false spans per minute on the calibration clips
do not exceed BEATs' at its shipping bar 0.35 on the same clips. Slice B (111 clips, Adam's
annotations) stays the untouched check. DCASE remains only for the visibility check until
Adam's visibility annotations replace it. The PSED row (v4ab) is re-run at the new bar; the
DCASE-chosen 0.20 row is kept as a record. The five-model average is judged by the same rule
and the same slice-B pass rule (docs/prereg_psed_ensemble.md), with its bar from the
calibration set.

### Calibration result (2026-09-19, `benchmark/detector_calib.json`)

280 calibration clips (47 min). BEATs at 0.35 makes **6.39** false spans/min there (DCASE: 5.2),
so the matched bars are **PSED 0.15** (was 0.20 on DCASE) and **five-model average 0.10**.
Slice B, untouched, at those bars:

| detector | bar | masked-consequential recall | all events | false spans/min | onset MAE |
|---|---|---|---|---|---|
| BEATs | 0.35 | 50.0% | 37.9% | 6.76 | 1.48 s |
| PSED | 0.15 | 64.8% | 57.0% | 4.05 | 1.31 s |
| five-model average | 0.10 | 68.2% | 62.4% | 5.78 | 1.57 s |

Reading: at BEATs' own false-alarm rate the average finds 3.4 points more hidden sounds than
PSED, at 1.7 more false spans per minute — a trade along the curve, not a better detector
(slice B cannot resolve a 3-point difference). By its declared rule (≥ 65.7% AND ≤ 2.59/min)
the average **fails** on the false-alarm side; the ensemble attempt is closed. PSED at the
DCASE bar (0.20) stays the running v4ab row; the protocol is not re-run at 0.15 now — one
final detector configuration (docs/prereg_detector_v5.md) will be run once through the
protocol when the last detector attempt closes.

## Stage-2 swap check (declared 2026-09-20, before running): SAM 3 vs OWLv2 on the DCASE visibility set

SAM 3 was chosen as the stage-2 detector (v4 "2") but never measured. Dev-only check, same
258 DCASE 2025 Task 3 events, frames and seed as the VLM visibility runs
(`benchmark/eval_dcase_visibility.py`, seed 7); one concept phrase per DCASE class
(`benchmark/eval_dcase_visibility_det.py`, PHRASE), the same wording style as OWLv2's
DETECT_QUERY; best presence score over six frames; bars = each model's pipeline default
(SAM 3 0.5, OWLv2 0.20), nothing tuned. Reported: agreement, on-screen recall, off-screen
recall (the one that costs the viewer wrong pictures), AUROC of the raw score, per class.
Decision rule: SAM 3 replaces OWLv2 in stage 2 iff its agreement is higher AND its
off-screen recall is not lower by more than 2 points; otherwise OWLv2 stays and the "2"
swap is dropped. The DCASE label is geometric (in field of view), so both models are judged
against the same imperfect gold; only the comparison is read, not the absolute numbers.

**Outcome (2026-09-20 06:30, jobs 30723804/30723805, L40S / RTX 6000 Ada):** on the DCASE
visibility set (256 / 253 of the 258 events scored; a few clips yielded no frames)
SAM 3 (bar 0.5): agreement **66.8 %**, on-screen recall 42.1 %, off-screen recall 90.8 %,
AUROC 0.767. OWLv2 (bar 0.20): agreement **69.2 %**, on-screen recall 62.7 %, off-screen
recall 75.6 %, AUROC 0.798. SAM 3's agreement is lower and its raw score separates on/off
worse (AUROC), so by the declared rule **OWLv2 stays in stage 2 and the "2" swap is dropped**.
Noted for the discussion: SAM 3 is the more conservative detector (fewer "visible" calls →
90.8 % off-screen recall), which is the direction that avoids wrong suppressions, but it
misses 58 % of the sources DCASE marks on screen; on the same 258 events the Qwen3.8
reasoning check reached 69.8 % agreement (`eval_dcase_visibility_q38_direct.json`), so
the VLM stays the stronger visibility witness and the stage-2 detector remains a supporting
vote under `VISIBILITY_RULE`.


## Amendment 6 — 2026-09-22 19:30 (declared after the oracle test and the miss autopsy, before the runs below; three Fables × three rounds)

**What the diagnostics showed.** (a) Oracle-gate test: with the annotator's own sound list in place
of the detector, the gate is significant on every subset (ΔF1 +0.067 [+0.012, +0.117] on the 109
benchmark clips, +0.060 on TEST-60, +0.084 on slice B; +0.097 with the extra "obvious" question).
(b) Half-oracle decomposition (`benchmark/gold/oracle_gate.py` + the row-B variant): giving the
system the right sounds **while keeping every false alarm the real detector produced** already makes
the gate significant — ΔF1 +0.075 [+0.029, +0.119]. So the end-to-end null is caused by **missed
sounds, not by phantom labels**. (c) Miss autopsy of the 43 missed needed sounds: 16 timing (a
picture of the right family exists outside the ±1 s window), 15 sub-threshold (BEATs 0.05–0.35),
9 detector-blind (< 0.05), 2 display bar, 1 label filter.

**Declared rows (before running).**
1. **v4b5 — display bar 0.15** (`AED_THRESHOLD` = 0.075), everything else exactly as v4b4, rendered
   for real on all 139 clips for proposed and blind_a2i. Rationale, declared here: the division of
   labour in this architecture is detector = recall, gate = precision, and the half-oracle shows
   phantoms do not destroy the gate's benefit. This row is **not** a tuned point estimate: the
   result is reported as the pair (0.35, 0.15) for both arms, with ΔF1 as a function of detector
   recall. The pre-registered v4b4 row stays the headline whatever v4b5 shows.
2. **Gate variant "majority + obvious"** (the fourth VLM question, the annotator's own rubric):
   declared now as a secondary row, to be rendered end to end if GPU allows. It was measured on the
   gold set in the oracle setting, so it is exploratory and can never become the headline; the
   sensitivity analysis (3-vote vs 4-vote on all three subsets) is reported with it.
3. **No metric change.** The ±1 s onset window, the primary F1 and the population are unchanged. The
   +3 s and +8 s windows stay sensitivity rows (ΔF1 +0.032 and +0.054 [+0.002, +0.105]).
4. **Not run** (declared as closed): CLAP top-k verifier (failed its DCASE calibration), Qwen2-Audio
   verifier (failed its declared bar in September), PretrainedSED (rejected three times), bigger
   VLM for the gate (27B ≈ 7B on gate accuracy), CED-base (clip-level, cannot fix onsets),
   audio-visual segmentation / MLLM localisation (they assume the source is on screen).
