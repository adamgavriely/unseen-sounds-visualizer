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


## Amendment 7 — 2026-09-22 21:00 (Adam: "we MUST improve the detection ... report per category ... only work with the gold set I annotated; you can split train/dev but only on what I did and keep variety")

- **Split.** `benchmark/gold/split.py` writes `benchmark/gold/split.json` once: clips are grouped by
  (category, population, sourcing wave) and dealt alternately into **DEV (79 clips, 69 needed sounds
  rated ≥ 2)** and **TEST (60 clips, 41)** with seed 7, so both halves carry the four categories, his
  benchmark clips and the AudioSet slice, and every wave. From here on every stage-level choice is
  made on DEV and TEST is read once. No external tuning set is used.
- **Stage-4 metric** (`benchmark/gold/detector_bench.py`, reported per category):
  *onset-recall* (a same-family detection starting within [−0.5, +1.0] s of a needed sound's onset —
  the picture metric's own window), *found-recall* (named anywhere while it sounds), *false labels
  per clip* (a detected family no gold sound of that clip has), *median onset error*.
- **Declared stage-4 grid:** BEATs bar {0.35, 0.25, 0.15, 0.10, 0.05} × {raw onsets, novelty-snapped
  onsets, lower hysteresis floor} × {BEATs alone, BEATs ∪ FlexSED}. Selection on DEV by onset-recall
  at the knee of the false-label curve; the pre-registered 0.35 row stays the headline.
- **Declared stage-5 grid** (free, from the cached votes): stretch rule {all, majority, any} × vote
  rule {majority of 3, obvious only, obvious OR majority, obvious AND majority, any of 4}. Selected
  by the oracle-gate ΔF1 on DEV (an end-to-end criterion, not a proxy), with the shipped rule kept
  as the headline.


## Amendment 8 — 2026-09-22 23:30: the detector changes (declared before the end-to-end run, after the stage-4 numbers and before any v4b6 score)

**Diagnosis.** At every onset where BEATs scored a needed sound below 0.05, BEATs' own top labels
were Speech 0.58–0.81 or Music 0.48–0.58 — the sound is masked, not out of vocabulary. Compensating
for that arithmetically (dividing by 1 − max(Speech, Music)) lifts those nine sounds only to
0.018–0.100, so the fix has to be a detector that is asked about one label at a time.

**FlexSED passed all three rules declared before it was run** (DEV half, 79 clips, 65 needed sounds
rated ≥ 2; 215 depictable family names as queries, fixed in advance):
1. *Recovers ≥ 4 of the 9 masked sounds:* **7 of 9** on the complete cache — siren 0.042 → 0.853,
   alarm 0.029 → 0.711, footsteps 0.016 → 0.695, whistle 0.044 → 0.586 and 0.011 → 0.585, door
   0.027 → 0.574, civil-defence siren 0.006 → 0.493; the two misses are hammer (0.019 → 0.194) and
   crow (0.049 → 0.032).
2. *Union adds ≥ 0.05 onset-recall at matched false labels:* on the complete cache BEATs 0.35 ∪
   FlexSED 0.8 gives **onset-recall 0.57 at 1.61 false labels per clip** — the same recall BEATs
   alone reaches only at bar 0.15 and **3.05** false labels, i.e. the same recall at **half** the
   false labels, and +0.12 over the shipped bar at 2.2× its false labels. The stricter union
   (FlexSED 0.7) reaches **0.62 at 2.78**, beating BEATs 0.15 on both axes.
3. *No-ambient false-label rate ≤ 2× BEATs:* 1.33 vs 0.72 per clip = **1.85×**.

**Adopted: `FLEXSED_BAR = 0.8`, union with BEATs at the unchanged bar 0.35** (`config.V4["0"]`,
`src/stage4_audio_event_detection/flexsed_infer.py`). A family both detectors report at the same
moment keeps the earlier onset; a clip without a FlexSED cache falls back to BEATs alone.

**Declared row v4b6** = STAGES `590` (depictable filter + 8-s cap + Qwen3.8 gate + the union),
rendered for real on all 139 clips for proposed and blind_a2i. The pre-registered v4b4 row remains
the headline; v4b6 is the declared detector-improvement row, and v4b5 (bar 0.15, now superseded) is
reported as the single-detector alternative at a comparable false-label budget.

**Per-category stage-4 table (DEV), the comparison Adam asked for:**

| config | onset-recall | found-recall | FA/clip | mixed | unseen | seen | no-amb |
|---|---|---|---|---|---|---|---|
| BEATs 0.35 (shipped) | 0.45 | 0.60 | 0.73 | 1.00 | 0.33 | 0.83 | 0.72 |
| BEATs 0.15 | 0.57 | 0.71 | 3.05 | 4.58 | 1.72 | 2.71 | 3.22 |
| **BEATs 0.35 ∪ FlexSED 0.8 (adopted)** | **0.57** | 0.69 | **1.61** | 2.21 | 1.61 | 1.33 | 1.33 |
| BEATs 0.35 ∪ FlexSED 0.7 | **0.62** | 0.74 | 2.78 | 4.11 | 2.56 | 2.12 | 2.50 |

(Complete 139-clip cache. Onset-recall on the two categories that carry needed sounds: mixed
0.44 → 0.50, **unseen 0.45 → 0.64** with the adopted union and **0.73** with FlexSED 0.7.)


## Amendment 9 — 2026-09-22 (viewer cost; Adam approved after seeing that equal-weight F1 cannot separate the systems — declared post hoc and labelled as such)

**Why.** F1 prices a picture of a sound that is not there exactly like a sound left undrawn. For a
deaf viewer those are not the same: a wrong picture misleads, a missing one leaves the viewer where
subtitles already leave them. F1 also cannot see the 74 of 139 clips that contain no needed sound —
more than half the set — where the only right answer is silence.

**Definition.** cost per clip = `COST_MISS` × (needed sounds rated ≥ 2 with no picture) + β ×
(pictures that are false alarms); lower is better. The weights are **not invented for this result**:
`COST_MISS = 4` and β = 2 are the numbers this project declared in September for the gate sweep
(`benchmark/gate_dev_sweep.py`), read off the judging rubric — a needed picture withheld scores 0
where it could have scored 4, and a picture shown where none is due is capped at 2 where silence
scores 4. The whole β curve is reported, not one chosen point.

**Result (109 benchmark clips, paired clip bootstrap).** At the declared β = 2: gated **3.36** vs
blind **4.24** per clip, **Δ −0.88 [−1.36, −0.46]**; on the untouched TEST-60, **−0.77 [−1.43,
−0.10]**. For the union row v4b6 the gap is larger: −1.23 [−1.80, −0.75] and −1.20 [−1.97, −0.50].
The gate is cheaper than blind for **any β > 0.40**, i.e. for any viewer who finds one wrong picture
at least a tenth as costly as one missed sound.

**And the finding that goes against the project, recorded here in full.** SILENCE costs **2.79** per
clip, which is *less than either system* at β = 2. Showing nothing wins whenever a wrong picture
costs more than about 1.4 (β), i.e. more than a third of a missed sound. Both systems beat silence
only in the window **0.40 < β < 1.41**. With the misses weighted by the annotator's own importance
(a level-3 danger sound counted 3, a level-2 event 2), the window moves to β < 1.0 and the gated
system beats both baselines at β = 0.25 and 0.5. So the honest statement is: **at its current
picture accuracy (23 % of the pictures it shows are right) this system is worth using only if a
wrong picture is cheap relative to a missed sound, and the gate always makes it cheaper than drawing
everything.** That is the number the thesis should argue about, and it is reported in both
directions.

## Amendment 10 — stage 4 as the joint bottleneck (2026-09-23, written BEFORE the sweep)

The error taxonomy on the adopted detector (v4b6, 109 benchmark clips) attributes 70 of 114 wrong
pictures (61%) to naming, and the corrected miss attribution on DEV attributes 11 of 21 misses to
"no event of this family covering the onset". Stage 4 therefore owns both sides of the viewer cost,
and the three knobs below are swept **jointly on DEV only**, with one TEST look afterwards.

Knobs (all read from caches already on disk; nothing is refitted per family, per clip or per label):

  tau   cross-detector veto. A picture is dropped when FlexSED's score for that family never
        reaches tau anywhere in the clip. Because FlexSED's own firing bar is 0.8, this can only
        touch labels FlexSED did not itself raise, so the seven sounds it was adopted to recover
        cannot be deleted by it. Grid: 0.0, 0.1, 0.2, 0.3, 0.5.
  bar   FLEXSED_BAR, the score at which FlexSED itself raises an event. Grid: 0.5, 0.6, 0.7, 0.8.
  duty  a picture is dropped when its family sits above the bar for >= duty of the clip (the
        "noise of the place"). Grid: none, 0.7, 0.5, 0.4.

**Selection rule, fixed now.** One cell is chosen on DEV: the cell with the lowest viewer cost per
clip (miss 4, wrong picture 2). Ties are broken by fewer misses, then by the smaller change from the
current setting (tau 0.0, bar 0.8, duty none). No other statistic selects the cell.

**Go/no-go for TEST, fixed now.** The chosen cell is adopted only if, on TEST:
  (a) viewer cost per clip is below the silence baseline on the same clips, and
  (b) recall loss against the current v4b6 setting is at most 2 hits, and
  (c) the sign of the DEV cost improvement is reproduced.
If any of the three fails, the result is reported as a failure and v4b6 stands. TEST is read once.

**Recorded in advance:** the display window [-0.5, +1.0] s is NOT changed. Three of the five
mistimed DEV misses are pictures that arrive 0.84-1.90 s EARLY, which the asymmetric window
penalises; widening it after seeing that would be fitting the metric to the result. It is reported
as a limitation instead, and any onset change must win inside the pre-registered window.

### Amendment 10, clarification (2026-09-23, still before any TEST look)

Condition (a) above conflated two different questions and would have made a real improvement
unreportable. They are separated here, before TEST is read:

  **Adoption of the stage-4 setting** (what amendment 10 decides). The chosen cell replaces v4b6
  only if, on TEST: the viewer cost per clip is lower than v4b6's on the same clips, the recall loss
  against v4b6 is at most 2 hits, and the sign of the DEV cost improvement is reproduced.

  **Whether the system beats silence** (the headline). Reported separately and unconditionally,
  whatever it shows. DEV currently stands at 3.59 against silence 2.69, and the bar sweep can
  recover at most about three of the six deaf sounds, so the honest expectation written down now is
  that the gated pipeline still costs a viewer more than showing nothing at beta = 2 on this gold
  set. That is a result about the picture budget, not a reason to withhold the stage-4 finding.

Tie-break confirmed as written: at equal cost, fewer misses wins. Breaking on fewest false alarms
would contradict the beta = 2 declared in amendment 9, which already prices a miss at two wrong
pictures.

**Corrected attribution of the 21 DEV misses** (an earlier draft of this analysis said timing was
48%; that was wrong and is recorded as such). Detection 11 (six where no detector hears the family
anywhere in the clip, five where the family is heard only at another moment), timing 5 (three of
them pictures that arrive 0.84-1.90 s EARLY), gate 3, label filter 2.

## Amendment 11 — a per-family bar for the second detector (2026-09-23, before the calibration runs)

**The problem, measured on presence only.** For the 78 needed sounds rated >= 2 whose family is in
FlexSED's 215-family vocabulary, FlexSED's score AT the sound has median 0.75 and spans 0.01-0.98.
The single global bar of 0.8 therefore refuses 46 of 78 real sounds (59%); 0.7 refuses 41%, 0.6
refuses 33%. The score scale is family-dependent -- Chainsaw, Bell, Train and Siren reach 0.96-0.99
while Dog never exceeds 0.56 and Owl never 0.34 anywhere in the 139 clips -- so one number cannot
serve every family. (A further 24 needed sounds are outside the vocabulary altogether; that is a
separate, reported limitation, not addressed here.)

**Why this is not the family whitelist again.** The whitelist failed (DEV 61% -> TEST 30%) because
it was a per-family binary fitted on DEV gold, about one sound per family. The bars here are fitted
on the 280-clip AudioSet-Strong calibration set in data/input/audioset_calib, which the gold never
touches: the overlap between those 280 ids and the 111 AudioSet ids in benchmark/gold/audioset_slice.json
was checked before anything ran and is **zero**.

**Criterion, fixed now -- the same one that set every other detector's bar** (benchmark/detector_calib.py,
"the loosest grid bar whose false spans per minute do not exceed BEATs at 0.35 on the same clips"),
applied per family instead of globally, on the grid 0.05..0.95 step 0.05.

**Minimum support, fixed now.** A family gets its own bar only where the calibration set contains at
least **K = 8** positive spans of that family. Otherwise the global bar stands. The number of
families that actually receive a per-family bar will be reported, whatever it is.

**The veto scales with the bar.** tau was frozen at 0.3 against a bar of 0.8. Where a family's bar
moves, its veto moves with it at the same ratio, tau_family = 0.375 x bar_family. This adds no new
free parameter.

**Both arms.** Per-family bars and the veto are stage-4 changes; blind_a2i is re-rendered with the
same detector, or the paired difference on TEST is not a comparison of gates.

**One TEST look.** Amendment 11 is folded into amendment 10's single look: the per-family cell joins
the DEV grid and is selected by the same minimum-cost rule. If it is not selected on DEV, it is
reported as tested and not adopted, and TEST is never read for it.

**Known threat, recorded before the result.** AudioSet-Strong labels are incomplete: a real but
unlabelled sound counts as a false positive, which pushes a bar upward. This biases exactly against
the quiet background families the amendment is meant to recover, so the procedure is conservative in
the direction that matters, and any failure to recover them cannot be claimed as evidence that the
sounds are absent.

### Amendment 11, clarification (2026-09-23, before any FlexSED calibration score is read)

Three definitions were underspecified and are fixed here, while the calibration job is still running
and no FlexSED score on the calibration set has been looked at.

**The reference is pooled, the measurement is per family.** BEATs at 0.35 makes 6.39 false spans per
minute on the calibration set *in total*. Split across 215 families that is 1-2 spans each, and for
a family BEATs is deaf to it is exactly zero -- so "the loosest bar whose per-family FP/min does not
exceed BEATs' per-family FP/min" would clamp FlexSED to silence on precisely the families it was
adopted to recover. Instead: the budget for one family is BEATs' GLOBAL false-span rate divided by
the number of eligible families, and a family's bar is the loosest grid bar whose own false-span
rate stays inside that budget.

**Eligibility, counted before the fit.** The calibration set has 46.7 minutes of audio and 235
labelled families, of which 109 are depictable. At K = 8 positive spans, **27 depictable families
are eligible**: Tap 113, Bird 102, Dog 99, Tick 69, Vehicle 61, Laughter 51, Alarm 35, Footsteps 33,
Cricket 22, Insect 20, Shout 19, Door 19, Clapping 19, Water 17, Telephone 16, Bell 15, Crowd 14,
Gunshot 14, Train 13, Whistle 12, Gobble 10, Snoring 9, Baby laughter 9, Thunder 9, Aircraft 8,
Basketball bounce 8, Typewriter 8. Everything else keeps the global bar. Hammer and Gasp -- two of
the six sounds no detector hears -- have no support and are therefore NOT helped by this amendment;
that is stated now so it cannot be presented later as a success.

**One fallback, not a grid.** Families without a per-family bar use 0.8 and nothing else. With 33
needed sounds on DEV, a per-family x fallback product would be fitting.

**The training-split question, answered.** The 280 calibration clips are from the AudioSet-Strong
**eval** split, which benchmark/audioset_detector_eval.py states is held out from the training of
every detector compared there. FlexSED is trained on AudioSet-Strong, so this matters: on the eval
split its scores are not inflated by memorisation, and the bars are not anti-conservative. The same
set set PSED's operating point, so the project's detector bars are all calibrated the same way.

**Incompleteness, kept as a stated limitation.** The criterion is relative -- FlexSED's false-span
rate against BEATs' on the same clips with the same incomplete labels -- so an unlabelled real sound
inflates both sides and only the difference in how the two models react to it survives. The
direction of that residual bias is against the quiet families the amendment targets, so a failure to
recover them is not evidence that they are absent.

## Amendment 12 — the speech-removal result, and onset refinement as its only surviving use (2026-09-23)

**The completeness test Adam approved, reported as it came out: the residual view FAILS its declared
rule.** residual = mix - DeepFilterNet3(mix), tagged by BEATs and OR-ed into the union as an extra
view. Go/no-go, written on 2026-09-22 before the audio was generated: onset-recall >= 0.62, at most
2.0 false labels per clip, and no rise on the quiet clips. On DEV (79 clips, 65 needed sounds):

    beats@0.35                                 onset 0.45  found 0.60  err 0.35s  false 0.73  (quiet 0.72)
    beatsres@0.35                              onset 0.60  found 0.72  err 0.15s  false 1.99  (quiet 2.44)
    beats+beatsres@0.35                        onset 0.63  found 0.75  err 0.15s  false 2.23  (quiet 2.89)
    beats+flexsed@0.8  (adopted v4b6)          onset 0.57  found 0.69  err 0.18s  false 1.61  (quiet 1.33)
    beats+flexsed+beatsres@0.35                onset 0.69  found 0.80  err 0.15s  false 3.03  (quiet 3.44)
    beats+flexsed+beatsres@0.65                onset 0.62  found 0.75  err 0.18s  false 1.80  (quiet 1.61)

Every configuration that reaches the onset bar either breaks the false-label budget or raises false
labels on the quiet clips, which is the condition that exists precisely to stop a detector buying
recall with noise. The residual view is **not adopted**. The two reviewers who ranked this idea last
and capped its value at about two sounds were wrong about the size of the recall gain (0.45 -> 0.60
alone) and right about the reason it cannot be used.

**What survives, and its rule, fixed now.** Removing the speech halves the median onset error,
0.35 s -> 0.15 s. That is a property of the residual's timing, not of its label list, so it can be
used for onset alone: an event already detected on the mix keeps its label, its family and its
existence, and only its ONSET is re-read from the residual's frame scores for that same family. No
label is added, none is removed, so the false-label count cannot change -- it is held constant by
construction, and this sidesteps exactly the condition the residual view failed.

Refinement rule: for an event (label L, onset t0) detected on the mix, take the residual's frame
scores for canonical(L); find the earliest frame within [t0 - 2.0, t0 + 2.0] s at which that score
crosses half its local peak; if such a frame exists, move the onset there, otherwise keep t0. The
+/- 2.0 s search window and the half-peak rule are fixed now and are not swept.

**Go/no-go, fixed now.** Adopt only if, on DEV, onset-recall rises by at least 0.05 with the false
labels per clip unchanged (they cannot change) and the median onset error not worse. It then joins
amendment 10's DEV grid as one cell and shares its single TEST look.

### Amendment 12, outcome (2026-09-23): onset refinement is REJECTED by its own rule

    beats@0.35+flexsed@0.8            onset 0.57  found 0.69  median err 0.18s  false 1.61
    the same, onsets re-read from the residual   onset 0.54  found 0.71  median err 0.28s  false 1.61

The rule required onset-recall to rise by at least 0.05 with the median error not worse. Onset-recall
FELL by 0.03 and the median error nearly doubled. The false-label count is identical, 1.61 on both
rows, which confirms the construction did what it claimed -- no label was added or removed -- and so
the failure is a clean statement about the timing itself.

Why, for the record: the residual's own detections have good onsets (0.15 s median), but that does
not transfer to an event detected on the mix. The half-peak rule inside +/- 2 s pulls a start
backwards into the residual's own noise floor, which is loud precisely where the speech used to be.
Both speech-removal ideas -- as a view, and as a source of timing -- are therefore closed, on their
declared rules, and the two reviewers' ranking of the idea as last is upheld even though their stated
reason (a gain of about two sounds) understated the recall it can reach.

## Amendment 13 — the gate decides per family, not per label (2026-09-23, declared before the test)

After the cross-detector veto, 18 wrong pictures on the benchmark are gate leaks: the sound is real
and sounding, and the annotator ticked its source visible or obvious. Split by the annotator's own
ticks, 13 of the 18 carry BOTH ticks -- the source is on screen AND a viewer with no sound would
assume it -- so these are the clearest cases in the gold, not annotation ambiguity.

Split by mechanism, three of the eighteen are a different failure with a one-line cause: the gate
looked at the video, decided that this family's source IS on screen, silenced that label, and the
pipeline then drew a SIBLING label of the same family.

    ambient_transport_subway_108   silenced Vehicle     -> drew Train
    m4_film_1917_33a               silenced Burst, pop  -> drew Explosion
    w8_dashcam_avalanche_road_2b   silenced Siren       -> drew Alarm

**The change.** A silence verdict propagates to every label of the same canonical family in the same
clip and overlapping moment. This is a coherence fix, not a fit to the result: the per-sound metric
already treats Train and Vehicle as one family, so a gate that decides at label level is answering a
different question from the one being scored. It adds no parameter.

**The risk, stated first.** Family-level propagation is aggressive. If a visible car silences
Vehicle while an unseen train is genuinely audible, the train is now suppressed and becomes a miss.
That is exactly what the go/no-go must catch.

**Go/no-go, fixed now.** Adopt only if, on DEV, the viewer cost per clip falls and at most ONE hit
is lost. Otherwise report as tested and rejected. It joins amendment 10's grid and shares its single
TEST look; it is applied to both arms.

**Recorded honestly:** the remaining 15 leaks are plain gate errors -- the vision model looked at a
visible source and said it was not there. That is not fixed by any rule change here, and the gate is
already the best of the 15 decision rules swept. It is the residual error of the visibility model,
and it is reported as such.

### Amendment 13, outcome (2026-09-23): REJECTED by its own rule

    DEV (49 clips)              F1     P      R      FA/clip  cost   hits  miss
    v4b6 as it stands           0.231  0.169  0.364  1.20     4.12   12    21
    + cross-detector veto       0.264  0.207  0.364  0.94     3.59   12    21
    + veto + family-level gate  0.228  0.196  0.273  0.76     3.47    9    24
    silence                                                   2.69

The cost did fall, 3.59 -> 3.47, but THREE hits were lost where the rule allowed at most one. The
risk written down before the test -- a visible car silencing Vehicle also suppresses an unseen train
-- is what happened. The family-level gate is not adopted. The three sibling escapes are real and
are reported as a known, unfixed error of the pipeline, because the only rule that removes them
costs more recall than it is worth at beta = 2.

### Disclosure: an unplanned look at TEST (2026-09-23)

The script that produced the table above also printed the same three rows for the 60 TEST clips,
because the loop was written over both halves. There is no way to un-see them, so they are recorded
here in full rather than left implicit:

    TEST (60 clips)             F1     P      R      FA/clip  cost   hits  miss
    v4b6 as it stands           0.301  0.243  0.395  0.88     3.50   17    26
    + cross-detector veto       0.316  0.288  0.349  0.62     3.10   15    28
    + veto + family-level gate  0.279  0.279  0.279  0.52     3.10   12    31
    silence                                                   2.87

**What this costs and what it does not.** No decision was taken on these numbers: amendment 13 was
already rejected on DEV before they were printed, and the veto's operating point tau = 0.3 was
selected on DEV and frozen in amendment 10 hours earlier. But the TEST result for the veto is now
known to the experimenter, so the single-look guarantee for amendment 10 is broken for that one
cell, and the honest description of the final TEST number for the veto is "confirmatory of a DEV
choice, seen once before the remaining grid cells were scored" rather than "a clean held-out look".
The bar and per-family cells of the grid have NOT been seen on TEST and their single look is intact.
Any report of this work states this paragraph rather than claiming an unbroken protocol.

### Amendment 10, second clarification (2026-09-23) — the selection rule was incoherent as written

Amendment 10 said "the cell with the lowest viewer cost per clip" with no recall constraint at
selection time, while its own go/no-go capped the recall loss at 2 hits. Applied literally to the
DEV sweep, the rule picks a cell its own go/no-go would then reject:

    tau 0.5  duty 0.5   cost 3.31   hits  6   miss 27     <- literal minimum cost
    tau 0.3  duty -     cost 3.59   hits 12   miss 21     <- the cell actually run
    tau 0.3  duty 0.7   cost 3.59   hits 10   miss 23

**Corrected rule:** the minimum-cost cell AMONG those that keep DEV hits within 2 of the current
v4b6 setting -- the go/no-go's own recall cap, applied at selection instead of only after it. Under
that constraint the minimum is tau 0.3 with no duty cap (3.59, tying with tau 0.3 / duty 0.7 and
winning the declared tie-break on fewer misses). The constraint is taken from a rule fixed before
any TEST number was seen, and the cell it selects is the cell that was already running.

**Timing, stated plainly:** this correction is written AFTER the accidental TEST look recorded above.
It changes no cell and no number, but a reader is entitled to know the order of events rather than
to trust it. The sentence in that disclosure -- "tau = 0.3 was selected on DEV and frozen in
amendment 10" -- was wrong: tau = 0.3 was the cell being run, and the rule that selects it is the
corrected rule written here.

**What was actually swept, and where.** The tau dimension and the rejection of the duty cap were
decided scorer-side, on pictures already rendered, because neither changes which sounds reach the
gate. The pipeline runs are the bar sweep (0.8, 0.7, 0.6) at the selected tau, with the bar 0.8 cell
serving as the replication of the scorer-side result. Amendment 10 described a full pipeline-side
tau x bar x duty grid; that is not what was run, and this paragraph replaces that description.

### Amendment 11, fitted bars and their stability (2026-09-23, before the DEV cell was scored)

All 27 eligible families receive a bar LOOSER than the global 0.8. Fitted values (support = positive
spans in the calibration set): Tap 0.35/113, Bird 0.40/102, Dog 0.60/99, Tick 0.25/69, Vehicle
0.70/61, Laughter 0.65/51, Alarm 0.75/35, Footsteps 0.45/33, Cricket 0.45/22, Insect 0.65/20, Shout
0.70/19, Door 0.35/19, Clapping 0.40/19, Water 0.35/17, Telephone 0.25/16, Bell 0.70/15, Crowd
0.70/14, Gunshot 0.40/14, Train 0.70/13, Whistle 0.15/12, Gobble 0.15/10, Snoring 0.25/9, Baby
laughter 0.40/9, Thunder 0.20/9, Aircraft 0.15/8, Basketball bounce 0.25/8, Typewriter 0.50/8.

**Stability, checked and reported before the result.** Re-fitting at half and at double the budget
moves every one of the 27 bars. The exact value of a family's bar is therefore NOT a robust
quantity, and no claim is made that these are optimal thresholds. What IS stable is the direction:
at every budget tried, essentially every family's bar sits below the global 0.8 (only Vehicle,
Alarm, Shout and Bell reach 0.80, and only at half budget). The per-family cell is consequently
tested as ONE configuration derived by a declared procedure, not as a tuned optimum, and if it wins
the DEV selection the honest claim is "a per-family bar helps", not "these are the right bars".

Implementation note: a family's bar is applied by rescaling that family's score column so its own
bar lands on the global one, clipped at 1.0. The clip matters -- the peak survives as the event's
confidence, which ranks sounds for the panel's limited rows, and an unclipped Whistle scaled by
0.8/0.15 would reach 5.3 and outrank every unscaled sound for a slot. The rescaling also makes the
veto scale with the bar at exactly the ratio frozen earlier, with no new parameter.

## Amendment 14 — the gate's remaining leaks are opened and closed (2026-09-23)

The 18 gate leaks that survive the cross-detector veto were split by their own logged votes
(gate_votes.json), not by a bucket name. Three are sibling escapes (amendment 13, rejected). The
other fifteen:

  **11 of 15: the vision model saw nothing, in any stretch.** Thunder x4, Water x2, Bell, Alarm,
  Vehicle, Fireworks, Shofar -- every stretch returned zero of three "is the source on screen" votes
  while the annotator ticked the source both visible and obvious. Four of the eleven are Thunder in
  storm clips and two are Water in an aquarium walk, where the "source" is the sky or a whole tank
  rather than an object a detector can box; that is a real edge in the visible definition and is
  reported as one rather than counted as a pure model failure.

  **4 of 15: the every-stretch rule.** The pipeline silences a sound only when the majority sees its
  source in EVERY stretch, so one dissenting stretch lets the picture through (Gunshot 2/1, Fire
  2/3/1, Glass 1/2, Siren 1/0/3).

**The looser rule was tested and is exactly cost-neutral.** Silencing when ANY stretch sees the
source, on DEV:

    silence only if EVERY stretch sees it (current)   hits 12  wrong 46  cost 3.59
    silence if ANY stretch sees it                    hits 10  wrong 42  cost 3.59

It removes four wrong pictures and loses two hits. At the declared beta = 2 a wrong picture costs 2
and a lost sound costs 4, so a rule must remove more than two wrong pictures per sound lost; this
one removes exactly two, and the cost is identical to the second decimal. It is therefore rejected
on the cost function alone, with nothing fitted.

**Candidate C is closed.** After the veto, what is left of the gate is eleven cases of a vision
model failing to see a source that a person sees immediately, and one aggregation rule whose
alternative is worth exactly nothing at the declared trade-off. This is a limitation of the
visibility models (OWLv2 + the VLM) and of the visible/obvious definition at the edges, and is
reported as such rather than presented as future work with a proposed fix.

## Amendment 15 — the object detector's verdict is computed and thrown away (2026-09-23, declared before the test)

Found while reading a running log: for london_protest_01 the stage-2 concept pass printed
`[stage2/owlv2] visible: ['Train', 'Vehicle']` and the gate then recorded
`visible? Vehicle -> no (name=no a/b=no desc=no)`. The picture was drawn, and that Vehicle picture is
one of the eleven leaks amendment 14 attributed to "the vision model saw nothing".

It is not that the vision model saw nothing. `SceneContext.visible_entities` is produced by every
stage-2 backend (owl.py, sam3.py, siglip.py). **Correction (2026-09-23, later the same day): an
earlier version of this paragraph said it is read by nothing. That is wrong.**
`plan_augmentations` does read it, but in the shipping configuration the verdict is deliberately
deferred to the per-sound VLM check (`defer = redundant and VLM_VISIBILITY and
DEPICTION_REASONING`), for a reason recorded in the code: stage 2 is a whole-clip object pass and it
once silenced a fire alarm because the pull station was visible somewhere in the video. The effect
on the gate is the same -- the final word is the VLM's -- but it is a design decision, not an
oversight. The gate's verdict is the VLM's alone, and an open-vocabulary object
detector that already located the source on screen has no vote.

**The change to test.** A sound is silenced if the VLM's majority says its source is visible in every
stretch (the current rule) OR the stage-2 concept pass lists its family among the visible entities.
This is an OR of two independent visibility opinions, so it can only silence more, never less.

**The risk, stated first.** OWLv2 is run at a low threshold over six frames and reports what it finds
anywhere in the clip, with no time alignment. A car parked in frame one will silence a passing
ambulance heard in frame six. That is the same failure mode that sank the family-level gate
(amendment 13), and the go/no-go must catch it.

**Go/no-go, fixed now.** Adopt only if, on DEV, the viewer cost per clip falls AND at most one hit is
lost -- the same cap amendment 13 was rejected against, so the two are judged identically. It joins
amendment 10's grid and shares its single TEST look, and is applied to both arms.

**Measurement note.** The verdicts for the cells already rendered are recovered from the run logs,
which interleave each clip's `Done -> ..._augmented.mp4` line, so an OWLv2 line is attributable to
the clip that follows it. Going forward the pipeline writes them to scene.json so no future analysis
depends on parsing a log.

### Amendment 15, outcome (2026-09-23): REJECTED by its own rule

    DEV (49 clips)                      F1     P      R      hits  wrong  FA/clip  cost
    veto only (the current best cell)   0.264  0.207  0.364   12     46    0.94    3.59
    veto + OWLv2 also silences          0.222  0.188  0.273    9     39    0.80    3.55
    silence                                                                        2.69

The cost moved by 0.04 and three hits were lost where the rule allowed one. The reason is the one
written down before the test: the concept pass reports what it finds anywhere in six frames with no
time alignment, so a vehicle parked at the start of a clip silences an ambulance heard at the end.
Seven wrong pictures removed is not worth three sounds at beta = 2, and it is the same failure that
sank the family-level gate.

**Correction to amendment 14.** That amendment said of eleven leaks "the vision model saw nothing".
That was true of the VLM but not of the pipeline: for some of them the object detector had located
the source, and its verdict was discarded. The accurate statement is that the gate's verdict rests
on the VLM alone, that the object detector's opinion is available and unused, and that ORing the two
has now been tested and costs more recall than it saves. The unused `visible_entities` field remains
a genuine oddity of the design and is reported as one.

**Score so far for the stage-4/stage-5 ideas tested today, all against rules fixed in advance:**
adopted 1 (the cross-detector veto), rejected 4 (speech-removal view, onset refinement from the
residual, family-level gating, the object detector as a second silencing vote), closed on evidence 1
(onset preference between detectors, only four comparable cases on DEV).

### Amendment 10, replication check (2026-09-23) — PASSES

Before reading any other grid cell, the bar 0.8 / tau 0.3 cell was scored as a replication: the
scorer-side result was obtained by dropping PICTURES whose family the second detector never hears,
while the pipeline drops the EVENTS themselves, which also changes what the gate is asked to vote on
and how the panel assigns its rows. If the two disagreed, every other cell would have to be re-read.

    DEV (49 clips)                 F1     P      R      FA/clip  cost   hits  miss
    v4b6 (current, no veto)        0.231  0.169  0.364  1.20     4.12   12    21
    bar 0.8 + veto (pipeline)      0.261  0.203  0.364  0.96     3.63   12    21
    the same veto, scorer-side                                   3.59   12    21
    silence                                                      2.69

Difference 0.04 in cost, identical hits and misses. The veto is therefore confirmed end to end, not
only as a scoring filter: it removes about a quarter of the wrong pictures (FA/clip 1.20 -> 0.96)
and loses no sound at all on DEV. The remaining grid cells can be read as intended.

## Amendment 16 — the veto made symmetric with a third detector (2026-09-23, declared before the DEV test)

The adopted veto is one-sided: it asks FlexSED about labels BEATs raised alone. Nothing asks about
labels FLEXSED raised alone, and after the veto those are the largest remaining error class
(wrong-family 33 + invented 14), dominated by sustained textures -- Insect, Bicycle, Ice cream van,
Power tool, Steam -- each at peak 0.87-0.94.

A third model answers them. PANNs CNN14 (the v1 detector, different architecture, different
training, already cached at benchmark/gold/panns_fw for all 139 clips) separates FlexSED's own right
and wrong pictures almost cleanly. Over the 69 benchmark pictures FlexSED itself raised (peak >= 0.8;
15 right, 54 wrong), PANNs' peak for the same canonical family:

    at FlexSED's RIGHT pictures   median 0.424   q25 0.256
    at FlexSED's WRONG pictures   median 0.026   q75 0.199

    tau2 0.02  drops 23 of 54 wrong, 2 of 15 right
    tau2 0.05  drops 34 of 54 wrong, 2 of 15 right
    tau2 0.10  drops 37 of 54 wrong, 3 of 15 right
    tau2 0.20  drops 40 of 54 wrong, 4 of 15 right

(BEATs as the second opinion is weaker -- 16 of 54 at the same tau2 -- because BEATs and FlexSED
already agree by construction wherever the union merged them.)

**The rule.** A span that FlexSED raised and BEATs did not is dropped when PANNs' peak for that
canonical family stays below tau2 anywhere in the clip. A span BOTH detectors raised is never
touched, exactly as in the adopted veto, so the construction stays asymmetric in the direction that
protects recall. Grid: tau2 in {0.02, 0.05, 0.10}, selected on DEV by amendment 10's corrected rule
(minimum cost among cells keeping hits within 2 of v4b6), never on the benchmark numbers above,
which include TEST clips and are therefore diagnostic only and are NOT used to choose tau2.

**Go/no-go for TEST.** As amendment 10: cost below v4b6 on TEST, recall loss at most 2 hits, DEV sign
reproduced. Both arms. Shares the single TEST look.

**Stated before the result:** the benchmark table above predicts a large win. The DEV-only numbers
may be smaller, and if DEV disagrees with it the DEV number governs and the disagreement is
reported.

### Amendment 16, DEV outcome (2026-09-23): SELECTED, and it crosses silence on the target category

    DEV (49 clips), selection only     F1     P      R      FA/clip  cost   hits  miss
    v4b6 (no veto)                     0.231  0.169  0.364  1.20     4.12   12    21
    + cross-detector veto (adopted)    0.264  0.207  0.364  0.94     3.59   12    21
    + PANNs veto tau2 0.02             0.286  0.250  0.333  0.67     3.14   11    22
    + PANNs veto tau2 0.05             0.297  0.268  0.333  0.61     3.02   11    22
    + PANNs veto tau2 0.10             0.282  0.263  0.303  0.57     3.02   10    23
    silence                                                          2.69

By amendment 10's corrected rule -- minimum cost among cells keeping hits within 2 of v4b6 (>= 10) --
tau2 = 0.05 and tau2 = 0.10 tie at 3.02 and the declared tie-break on fewer misses selects
**tau2 = 0.05**. Cost falls 4.12 -> 3.02 (-27%) and precision rises 0.169 -> 0.268 (+59%) for one
sound lost, where the rule allowed two.

**The result that matters, on the category the project exists for.** Cost per clip at the declared
operating point (a missed sound 4, a wrong picture 2), and the price of a wrong picture at which the
pipeline stops being worth using:

    unseen clips (14)        cost @ beta=2    silence    crossover
      + cross-detector veto      5.14          4.86      beta = 1.71
      + PANNs veto               4.00          4.86      beta = 5.00

    all DEV clips (49)
      + cross-detector veto      3.59          2.69      beta = 1.04
      + PANNs veto               3.02          2.69      beta = 1.47

On the unseen clips the gated pipeline is now CHEAPER TO A VIEWER THAN SHOWING NOTHING at the
project's own declared weights, and stays cheaper until a wrong picture is priced at five times --
that is, for any plausible viewer. On the benchmark as a whole silence still wins, for the reason
already recorded: 27 of the 49 DEV clips contain no sound that should ever be drawn, and silence
cannot be beaten there by construction.

**This is a DEV result and TEST has not been read for it.** The go/no-go fixed in amendment 16
governs. Both arms are re-rendered at tau2 = 0.05 before the single TEST look.

### Amendment 16, the crossover with a confidence interval (2026-09-23) — the win is NOT significant

A paired clip bootstrap (2000 draws, seed 0) on the same DEV clips:

    unseen (14 clips)   silence - ours at beta=2:  +0.86   95% CI [+0.00, +1.86]   NOT significant
                        crossover beta 5.00        95% CI [ 2.00,   8.00]
    all DEV (49 clips)  silence - ours at beta=2:  -0.33   95% CI [-0.86, +0.20]   not significant

**Correction to the DEV outcome above, and to what was reported to Adam.** The point estimate on the
unseen clips does cross silence (4.00 against 4.86) and the crossover does move from 1.71 to 5.00,
but on fourteen clips the interval on that gap runs from exactly zero to +1.86. The honest statement
is therefore: *the gated pipeline's cost on unseen clips is no longer distinguishable from silence,
having previously been worse, and the point estimate now favours the pipeline* -- not "the pipeline
beats silence". The crossover's own interval, [2.00, 8.00], has its lower bound at the declared
operating point, which says the same thing from the other side.

This is the clearest argument yet for more annotated clips, and it is an argument about power, not
about the method: the effect is the right size and the sample cannot resolve it. Fourteen unseen
clips in DEV is what the current gold set provides.

### Amendment 16, the TEST threshold written down before the read

From the accidental look recorded earlier, v4b6 on TEST has 17 hits. Go/no-go (b) is inherited from
amendment 10 and is measured **against v4b6 for the cell as a whole**, so the cell passes only with
**15 or more hits on TEST**. The veto alone already cost 2 hits there (17 -> 15), and the PANNs veto
cost one more on DEV, so failing (b) is a live possibility and is written down now so the read is a
check rather than an interpretation.

**What happens if (b) fails while cost and the unseen category both improve:** that is not a "v4b6
stands" outcome. It is reported as "the cross-detector veto adopted, the PANNs veto tested and not
adopted on the declared recall rule, with its cost and category results reported in full". The
decision rule is not renegotiated after the number is seen.

### Amendment 16, a disclosure about the grid endpoints

The tau2 grid {0.02, 0.05, 0.10} was chosen after reading the 109-clip diagnostic table, which
contains TEST clips: 0.20 was left out of the grid because that table showed it losing 4 of 15 right
pictures. The cell WITHIN the grid was then selected on DEV alone. The grid's endpoints are
therefore mildly informed by TEST and the doc's earlier claim that those numbers were not used is
too strong. Recorded rather than corrected silently.

### A cell that had to be discarded, and the false result it nearly produced (2026-09-23)

**The diagnosis below was wrong and is corrected at the end of this section. The discard stands; the
reason does not.**

The bar 0.7 DEV cell scored 1 hit, zero false alarms and a cost of 2.61 -- BELOW silence at 2.69 --
which would have been the first cell all day to beat silence on the whole benchmark. It was wrong.
The job had run the FLUX generator instead of the placeholder (backend "generate" on 97 of its 98
augmented specs), the generator failed on the card it landed on, and 97 pictures were written with
no image file. The scorer counts only pictures the panel could actually show, so the cell scored as
near-silence and "won" by not drawing anything.

Checked across the grid before believing any of it: b8 45/45 placeholder with files, b6 88/88, the
per-family cell 77/77, bar 0.7 **1/98**. Only bar 0.7 is affected; it is re-run, and its earlier
numbers are void.

Recorded because the failure mode is the dangerous kind: a broken cell does not look broken in a
cost table, it looks like a win, and the metric rewards a system that shows nothing on a benchmark
where most clips should show nothing. Any cell that appears to beat silence is checked for pictures
that exist before it is believed.

**Correction (same day, within the hour).** The bar 0.7 job was NOT misconfigured. Its own startup
line reads `[cfg] phase=render ... gen=placeholder`, exactly like the cells that worked. The cell was
simply **still rendering** when it was scored, and every check used to decide it was finished was
unreliable:

  * the work directory is created when a clip STARTS, so counting directories counts started clips;
  * `augmentations.json` is written TWICE by src/pipeline.py -- once after gating (line 95) and again
    after the pictures are made (line 114) -- so the file exists, with a full list of specs, before a
    single image has been generated;
  * `backend: "generate"` is the value a spec carries before stage 6 touches it, not a record that a
    generator ran.

So the cell scored as near-silence because its pictures did not exist YET. The only sound completion
test is the one that was applied afterwards: every `augment: true` spec has an `image_path` and that
file is on disk (b8 45/45, b6 88/88, per-family 77/77, bar 0.7 **1/98**).

The conclusion that matters is unchanged and is worth more than the incident: **a cell that draws
nothing scores below silence on this benchmark, so any cell that appears to beat silence must first
be shown to have drawn pictures that exist.** That check is now enforced in
benchmark/gold/grid_select.py rather than promised in prose, because the next cell to fail this way
will look exactly like a win.

### Amendment 10/11/16 — the complete DEV grid and the selected cell (2026-09-23)

Every cell rendered end to end, both arms, placeholder generation, and every one passing the
pictures-exist guard:

    cell                          F1      P      R      FA/clip  cost   hits  miss
    v4b6 (current, no veto)       0.231  0.169  0.364   1.20     4.12   12    21
    bar 0.8 + veto [replication]  0.261  0.203  0.364   0.96     3.63   12    21
    bar 0.7 + veto                0.182  0.130  0.303   1.37     4.61   10    23
    bar 0.6 + veto                0.187  0.123  0.394   1.90     5.43   13    20
    per-family bars + veto        0.238  0.161  0.455   1.59     4.65   15    18
    veto + PANNs veto 0.05        0.289  0.256  0.333   0.65     3.10   11    22
    silence                                             0.00     2.69

By the corrected rule -- minimum cost among cells keeping hits within 2 of v4b6 -- the selected cell
is **veto + PANNs veto at tau2 = 0.05**, cost 3.10 against v4b6's 4.12, a 25% reduction, with
precision up from 0.169 to 0.256 and one sound lost where two were allowed. The scorer-side estimate
for this cell was 3.02, so the pipeline-side replication holds a second time (0.08 apart).

The per-family cell is the opposite trade and is reported alongside rather than discarded: it is the
only configuration that RAISES recall (0.364 -> 0.455, three more sounds found), and it wins below
beta ~ 1.2. The two cells together are the project's operating curve -- the conservative cell for a
viewer who dislikes wrong pictures, the per-family cell for one who minds them less.

**TEST is deliberately NOT read yet.** Adam's instruction was to push the pipeline further before
the single look, and reading TEST now would spend it on a cell that further work may supersede. The
look happens once, at the end, on whatever cell DEV selects then, under the go/no-go already fixed.

## Amendment 17 — agreement across independent models (2026-09-23, declared before any score is read)

The PANNs veto worked because PANNs is genuinely different from FlexSED: different architecture,
different training, so the two fail in different places. That argument does not stop at one extra
model. Two further AudioSet taggers are cached over the gold clips, in the same 527-label space as
PANNs and BEATs so their scores are directly comparable per canonical family:

    AST   MIT/ast-finetuned-audioset-10-10-0.4593   spectrogram transformer, AudioSet mAP ~0.459
    CED   mispeech/ced-base                         consistent ensemble distillation, mAP ~0.496

Neither is frame-level. For a veto that is irrelevant: the only question asked of a supporting model
is whether it hears the family ANYWHERE in the clip, which is exactly what a clip-level tagger
answers. They are never used to create a span, only to confirm or refuse one, so they cannot add a
false alarm and cannot move an onset.

**The rule.** A span raised by exactly ONE of the two span-producing detectors (BEATs, FlexSED) must
be supported by at least **k** of the supporting models {PANNs, AST, CED}. A span both BEATs and
FlexSED raised is never touched -- the same guard as amendments 10 and 16. Support means the model's
peak for that canonical family reaches tau2 somewhere in the clip.

**Grid, fixed now:** k in {1, 2, 3} x tau2 in {0.02, 0.05, 0.10}. Selection on DEV only, by
amendment 10's corrected rule (minimum cost among cells keeping hits within 2 of v4b6). Nine cells,
scored on pictures already rendered, so nothing new is generated to choose among them.

**Predicted before the result, so it can be wrong in writing.** First a correction to the obvious
reading: k = 1 over {PANNs, AST, CED} is LOOSER than the adopted PANNs-only veto, because a span
PANNs would refuse now survives if AST or CED hears the family. So k = 1 should lose fewer sounds
AND remove fewer false alarms than the adopted cell -- not behave like it. The interesting cell is
k = 2, and the real question is whether AST and CED disagree with PANNs where PANNs is WRONG
(diversity helps, and the ensemble is worth its cost) or merely agree with it where it is RIGHT
(all three are AudioSet-trained on the same data and are redundant, in which case the PANNs result
was about its architecture being unlike FlexSED's, not about ensembling).

k = 1 should behave close to the
adopted PANNs veto since PANNs already carries most of the signal; k = 2 should trade a little
recall for precision; k = 3 should be too strict and lose more than two sounds. If instead the extra
models add nothing at any k, that is a real finding about ensemble diversity -- that PANNs' value
was specifically its architecture being unlike FlexSED's, not that "more models" helps -- and it is
reported as such.

**This shares amendment 10's single TEST look.** No TEST number has been read for any cell.

### The primary comparison after the vetoes: ours vs blind on DEV (2026-09-23)

Both arms get the vetoes -- they are a shared stage-4 change, so the comparison stays paired and
fair. 49 DEV clips, clip bootstrap 2000 draws, seed 0:

    before today (v4b6)          F1     P      R      FA/clip  cost
      ours                       0.231  0.169  0.364  1.20     4.12
      blind                      0.199  0.130  0.424  1.92     5.39
      delta F1  +0.032  95% CI [-0.029, +0.090]   P(d>0) 0.853
      delta P   +0.039  95% CI [-0.008, +0.087]   P(d>0) 0.945     not significant
      cost advantage over blind  +1.27  95% CI [+0.61, +2.00]      significant

    after both vetoes            F1     P      R      FA/clip  cost
      ours                       0.289  0.256  0.333  0.65     3.10
      blind                      0.257  0.191  0.394  1.12     3.88
      delta F1  +0.032  95% CI [-0.044, +0.094]   P(d>0) 0.809
      delta P   +0.065  95% CI [+0.001, +0.121]   P(d>0) 0.976     SIGNIFICANT
      cost advantage over blind  +0.78  95% CI [+0.29, +1.27]      significant

**The gate's precision advantage over the blind baseline becomes significant for the first time**,
on DEV, and the equal-weight F1 difference stays exactly where it has always been (+0.032, not
significant) -- which is the structural point established earlier: F1 prices a picture of a sound
that is not there the same as a sound left undrawn, and cannot see the trade the gate makes.

**Reported against interest:** the gate's COST advantage over blind SHRANK, 1.27 to 0.78, and the
shrinkage is real rather than noise. The vetoes are a shared stage, so blind gets them too, and the
better the detector becomes the less there is for a visibility gate to remove. That is a genuine
finding about where the gate's value comes from -- it is worth most when the detector is worst --
and it belongs in the thesis next to the claim that the gate helps.

### Amendment 17, outcome (2026-09-23): more models do NOT help — PANNs alone wins

    DEV (49 clips)          F1     P      R      FA/clip  cost   hits  miss
    veto only               0.264  0.207  0.364  0.94     3.59   12    21
    any 1 of {P,A,C} 0.02   0.264  0.207  0.364  0.94     3.59   12    21
    any 1 of {P,A,C} 0.05   0.264  0.207  0.364  0.94     3.59   12    21
    any 1 of {P,A,C} 0.10   0.264  0.207  0.364  0.94     3.59   12    21
    2 of 3, tau2 0.02       0.275  0.234  0.333  0.73     3.27   11    22
    2 of 3, tau2 0.05       0.282  0.244  0.333  0.69     3.18   11    22
    2 of 3, tau2 0.10       0.270  0.244  0.303  0.63     3.14   10    23
    3 of 3, tau2 0.05       0.265  0.257  0.273  0.53     3.02    9    24   INELIGIBLE (recall cap 10)
    PANNs alone (adopted)   0.297  0.268  0.333  0.61     3.02   11    22
    silence                                               2.69

**The adopted PANNs-only veto beats every cell of the ensemble grid.** "Any one of three" has
literally no effect at any threshold -- at least one of PANNs, AST or CED hears the family every
single time, so the rule never fires. "Two of three" is strictly worse than PANNs alone at matching
recall (3.18 against 3.02 at 11 hits). "Three of three" matches PANNs' cost but at nine hits, which
the recall cap refuses.

**The prediction written in advance was right on the shape and wrong on the size.** It said k = 1
would be looser than PANNs alone -- it is, to the point of doing nothing -- and that k = 3 would lose
more than two sounds, which it does. It also posed the real question, and the answer is the
unflattering one for ensembling: AST and CED hear the families PANNs hears when PANNs is right, and
ALSO hear the ones it correctly refuses, so they can only weaken the veto.

**The finding, stated as the thing worth keeping:** PANNs' value was never that it was a third
model. It was that its architecture and training are unlike FlexSED's *in the specific way that
matters* -- a CNN on log-mel with clip-level supervision versus a text-queried SSL transformer -- so
it is quiet exactly where FlexSED's text query hallucinates a sustained texture. Two further
AudioSet transformers are not that. **Diversity has to be the right kind, and adding models is not a
substitute for it.** Amendment 17 is not adopted, and the AST and CED caches stay in the repository
as the evidence for this negative result.

### Amendment 17, CORRECTION (2026-09-23): the first reading was confounded by a broken score scale

The conclusion above (PANNs alone wins, ensembling does not help) survives. The MECHANISM stated
with it was wrong, and it was wrong in a way that a reviewer would have caught, so it is corrected
here rather than quietly edited.

**The confound.** The three models were thresholded at the SAME tau2 without checking that their
scores mean the same thing. They do not. Share of the 527 labels whose clip peak exceeds the
threshold, median DEV clip:

    model     t=0.02   t=0.05   t=0.10
    PANNs       3.2%     1.7%     0.9%
    AST         2.5%     1.3%     0.9%
    CED       100.0%   100.0%   100.0%

**CED puts every one of 527 labels above every threshold in every clip.** It therefore "supports"
everything, which is the entire reason "any 1 of 3" had literally no effect, and it silently turned
"2 of 3" into "PANNs or AST" and "3 of 3" into "PANNs and AST". The sentence written earlier -- that
AST and CED "hear the families PANNs correctly refuses" -- described a calibration artefact as if it
were a fact about hearing. Withdrawn.

**Re-run with CED dropped** (AST is calibrated comparably to PANNs, so the comparison is fair):

    cell                    F1     P      R      FA/clip  cost   hits
    veto only               0.264  0.207  0.364  0.94     3.59   12
    PANNs only (adopted)    0.297  0.268  0.333  0.61     3.02   11
    AST only                0.250  0.231  0.273  0.61     3.18    9
    PANNs or AST            0.282  0.244  0.333  0.69     3.18   11
    PANNs and AST           0.265  0.257  0.273  0.53     3.02    9

**PANNs alone still wins, and now the reason is supported rather than asserted.** AST removes the
same quantity of false alarms at the same threshold (0.61 per clip, identical to PANNs) but costs
three sounds instead of one -- it is not miscalibrated, it is simply a weaker discriminator for this
particular question. Combining them helps in neither direction: the OR is looser and costs more, the
AND is stricter and loses the same three sounds.

**What can honestly be claimed:** a second opinion from a model unlike the first is worth a great
deal (4.12 -> 3.02), and which model it is matters more than how many there are. What cannot be
claimed from this evidence is a general story about architectures; two AudioSet taggers were tried,
one helped and one did not, and one of the two was unusable at a shared threshold. CED could be
re-tested with a per-model threshold fitted on the AudioSet calibration set, exactly as amendment 11
did for FlexSED, and that is left as untested rather than reported as a failure of the model.

### The noise-name label filter: sized, and NOT changed (2026-09-23)

The picture check found that labels naming a noise rather than an object are almost never depicted
(1 of 3 for us, 0 of 4 for blind), which suggested a cheap precision win in src/labels.py. Sized
before writing any code, on the whole benchmark with both vetoes applied: **two pictures**, "Chink,
clink" and "Steam", both wrong. No needed gold sound carries such a label, so a filter would cost no
recall -- but it would buy two pictures out of 111 in exchange for a hand-written list of words,
which is a fitting surface for no measurable gain. **Not changed.** The depictable allow-list
(amendment "9") already removes nearly all of these; this is the residue, and it is smaller than the
noise in any statistic reported here.

Also declined, and worth recording because it was tempting: eight labels are drawn on this benchmark
and are never once correct (Thunder x4, Fire x2, Coin, Power tool, Shout, Snoring, Microwave oven,
Shofar -- twelve pictures). Removing them would look like a free win and would be **exactly the
family whitelist that failed before** (DEV 61% -> TEST 30%): a rule fitted to which labels happen to
be wrong on the data being scored. Not done.

## Split balance, checked before the TEST read (2026-09-23, Adam's question)

Adam asked whether DEV and TEST are equally representative by category, so that neither half is
biased. Checked rather than assumed. The split was made once (benchmark/gold/split.py, seed 7,
stratified by category x population x sourcing wave) and is not changed by this check.

    half      clips  unseen  mixed  seen  no_ambient   needed sounds  unseen/mixed share
    DEV        49     14      8     18      9              33          52% / 48%
    TEST       60     13     15     12     20              43          53% / 47%
    slice B    30      6      8     14      2              28          61% / 39%

**On the quantity the metric actually measures the split is well balanced.** The needed sounds --
the only thing that can produce a hit or a miss -- are 52/48 unseen-to-mixed on DEV and 53/47 on
TEST, and the share of clips containing nothing that should ever be drawn is 55% on DEV and 53% on
TEST. Neither half is loaded toward the easy or the hard side of the measurement.

**One real asymmetry, disclosed before the TEST numbers are read.** The two "nothing to draw"
categories are split unevenly, and they test different stages:

    seen        source IS on screen -> tests whether the GATE silences it     DEV 18, TEST 12
    no_ambient  no such sound at all -> tests whether the DETECTOR invents it  DEV  9, TEST 20

TEST therefore carries proportionally more of the clips that punish an inventing detector and fewer
of the clips that punish a leaking gate. Since everything adopted today is a DETECTOR change --
two vetoes whose whole purpose is to stop the detector naming sounds that are not there -- TEST is
loaded slightly in favour of the change being evaluated.

This is stated now, before the read, so it cannot be presented afterwards as either a triumph or an
excuse. It is a property of a split fixed before any of this work existed, and re-drawing the split
after seeing DEV results would be a far worse error than living with it. The correct treatment is to
report the TEST result per category as well as overall, so a reader can see the effect directly
rather than take this paragraph on trust.

### Correction to a number already reported (2026-09-23): the unseen crossing is 3.33, not 5.00

An earlier entry, and a report to Adam, put the crossing where the gated pipeline meets silence on
the unseen clips at beta = 5.00. That figure came from applying the PANNs veto to v4b6's pictures
inside the scorer. The rendered configuration -- the one that actually ships -- crosses at **beta =
3.33**, with a 95% interval of [1.33, 10.0] on 14 clips. The blind baseline crosses at 3.11 on the
same clips, so the gate's advantage in this category is small, not large.

The all-clips crossing is **beta = 1.37, 95% interval [0.71, 2.19]**. That interval CONTAINS the
rubric's asserted 2.0, which means the honest statement is stronger and duller than either side
would like: on this evidence neither "we beat silence at beta = 2" nor "silence beats us at beta =
2" is established. The point estimate favours silence; the uncertainty covers both.

Figure: benchmark/gold/beta_figure_dev.png (benchmark/gold/beta_figure.py). Two panels, all clips
and unseen clips, x capped at the derived 4.30, the band where the system is cheaper than silence
shaded, and the rubric's 2.0 drawn dashed so it reads as an assertion rather than a measurement.

# THE SINGLE TEST LOOK (2026-09-23) — taken once, under rules fixed before it

Completeness guard passed on all four arms before any number was read. All 60 TEST clips have a
complete oracle vote record.

## Go/no-go, in the order declared

    (a) cost per clip below v4b6's       2.53 vs 3.67      PASS
    (b) at least 15 hits                 16 (v4b6 has 18)  PASS
    (c) sign of the DEV improvement      DEV 4.12->3.10, TEST 3.67->2.53, both fell   PASS

    VERDICT: the cross-detector veto + the PANNs veto are ADOPTED.

## TEST, 60 clips

    system        F1      P      R      FA/clip  cost   hits  miss
    v4b6         0.298  0.231  0.419   1.00     3.67   18    25
    v4b6 blind   0.275  0.185  0.535   1.68     4.70   23    20
    OURS         0.395  0.421  0.372   0.37     2.53   16    27
    blind        0.348  0.278  0.465   0.87     3.27   20    23
    silence                                     2.87    0    43

Precision rises 0.231 -> 0.421 (+82%) and false pictures per clip fall 1.00 -> 0.37 (-63%), for
three sounds lost, named in full below.

## The primary comparison, ours vs blind (paired clip bootstrap, 2000 draws, seed 0)

    before (v4b6)        dF1 +0.022 [-0.045,+0.081]   dP +0.045 [-0.010,+0.105]    dcost +1.03 [+0.37,+1.73] *
    after (final cell)   dF1 +0.047 [-0.057,+0.134]   dP +0.143 [+0.034,+0.264] *  dcost +0.73 [+0.20,+1.33] *

**The gate's precision advantage over the blind baseline is SIGNIFICANT on TEST** (+0.143), where
before today's work it was not (+0.045, interval crossing zero). The cost advantage was already
significant and remains so, while shrinking from 1.03 to 0.73 for the reason recorded on DEV: the
vetoes are a shared stage, so the blind arm gets them too. Equal-weight F1 stays non-significant,
exactly as it has throughout, and for the structural reason established in September.

## Against showing nothing

    ALL 60 TEST clips     silence - ours = +0.33   95% CI [-0.33, +1.07]   not significant
    unseen clips (13)     silence - ours = +3.08   95% CI [+1.08, +5.38]   SIGNIFICANT
       (v4b6 on the same 13 clips was already +2.62 [+0.46, +5.23])

**On the off-screen clips the project exists for, the gated pipeline is significantly cheaper for a
viewer than showing nothing, at the operating point this project declared for itself in September.**
Across all 60 TEST clips the pipeline is cheaper on the point estimate (2.53 against 2.87) but the
interval contains zero, so no claim is made there.

## Per category

    category     clips   ours   blind  silence   ours P   ours R
    unseen         13    4.00   3.54    7.08     0.846    0.478
    mixed          15    6.00   7.60    5.33     0.250    0.250
    seen           12    0.50   2.50    0.00     -        -
    no_ambient     20    0.20   0.30    0.00     -        -

On the unseen clips precision reaches 0.846 -- five of every six pictures shown are the right sound
at the right moment. The crossing where the system stops beating silence on those clips is beyond
the search range (beta > 20, 95% CI [6.00, 20.00] on 13 clips), so at any plausible price of a wrong
picture it is worth using there. Mixed clips remain the weak category, and clips with nothing to
draw cost 0.20-0.50 against silence's zero, which is the residue the vetoes could not remove.

**Honouring the disclosure made before the look:** TEST was known in advance to be no-ambient heavy
(20 clips against DEV's 9) and therefore loaded in favour of a detector change, which is what was
adopted. The per-category table above is exactly what lets a reader discount for that, and the
effect is visible: on the 20 no-ambient clips our cost is 0.20, near-perfect, and those clips are a
third of TEST. The DEV/TEST difference in the all-clips comparison against silence (DEV 3.10 vs 2.69,
losing; TEST 2.53 vs 2.87, winning on the point estimate) is explained by that composition, and
neither half is claimed as significant against silence overall.

## Every sound the change costs

    m4_fire_bodycam_23a              Siren      at  3.00s   importance 3
    mc_bridge_scene                  Bow-wow    at  0.10s   importance 2
    w8_film_hunt_for_red_october_1b  Sonar      at  0.00s   importance 2

Three, of which one is rated 3 (the highest importance the annotator assigns). That is the price of
the 38 wrong pictures the vetoes removed.

TEST is now closed. No further number is read from it.

## Amendment 18 — the video side, tested (2026-09-23)

Adam asked three things: does the model need to see MOTION rather than stills, does it need to sync
with the audio, and is there a better model than OWLv2. Two are now answered with measurements.

### (a) Sending the frames as VIDEO rather than as loose images: NO EFFECT

`reason._ask` gives the VLM one `{"type": "image"}` per frame, so it never receives temporal
position ids or the spacing between frames. Re-asking the same question, on the same stretches, with
the same model and frames, changed only to `{"type": "video"}` with an fps:

    blind leaks that now name the source        0 / 8
    hits that would be lost (now named)         0 / 12

**Video encoding changes nothing.** The model answers "nothing" either way; in one case
(`mv_tornado_scene` Siren) the image mode named "air horn" and the video mode did not, so if
anything the video path abstains slightly more. Together with the vote log -- ten of eleven blind
cases answered "nothing" in EVERY stretch -- this closes the question: **the gate's blindness is
perception, not encoding and not prompting.**

*Two earlier runs of this test were void and are recorded as such: the first crashed on every call
(`fps` expected a float, was given a list) and the second left the chat template's reasoning mode
on, so the model emitted "The user is asking about..." which the flip test scored as success. The
result above is the third run.*

### (b) A TIME-ALIGNED object-detector vote: exactly cost-neutral at the declared bar

The earlier rejection of OWLv2 as a silencing vote was CLIP-level, with no time alignment. Per
stretch, over the same frames the VLM saw, at our config's own `OWL_THRESHOLD` of 0.20, on DEV:

                          OWL sees it     OWL blind
    VLM says visible          ...             ...
    VLM says NOT visible       24              51

Of those 24, silencing them would remove **16 wrong pictures** and lose **8 needed sounds** -- a
ratio of exactly **2.00** against the beta = 2 break-even of 2.0. It fails, by nothing at all. This
is the second rule today to land exactly on the break-even line (the every-stretch gate rule was the
other), which is worth noting as a property of the trade rather than a coincidence.

**Diagnostic, reported as post-hoc:** the two groups are not inseparable. Needed-sound scores top out
at 0.33 while leak scores run to 0.87, so a stricter bar does separate them:

    bar 0.20   16 leaks removed,  8 hits lost   ratio 2.00
    bar 0.25    8 leaks removed,  4 hits lost   ratio 2.00
    bar 0.30    4 leaks removed,  3 hits lost   ratio 1.33
    bar 0.35    4 leaks removed,  0 hits lost   clears
    bar 0.40    3 leaks removed,  0 hits lost   clears

A bar of 0.35 removes four wrong pictures and loses nothing. Selecting it on DEV is the same
procedure that selected tau = 0.3 and tau2 = 0.05, so it is legitimate -- but the gain is small:
four pictures on 49 clips is about 0.16 cost per clip, against the 1.14 the two detector vetoes
moved. It is recorded as available rather than pursued, because a stricter silencing bar is also the
kind of knob that looks free on DEV and costs recall on TEST, and the current cell has already been
through its single TEST look.

### (c) A better model than OWLv2: the strongest remaining video lever, not yet run

SAM 3 (`facebook/sam3`, Meta, ICLR 2026) scores more than double OWLv2's cgF1 on the open-vocabulary
SA-Co benchmark -- where OWLv2 is the paper's own strongest baseline -- and carries a concept through
frames rather than judging each alone. `src/stage2_video_understanding/sam3.py` has been in this
repository since the v4 plan and was never adopted, because stage 2's verdict is read by nothing.

The harness now takes `--backend sam3`, so the same 2x2 can be run against it. Two experiments with
different ceilings, and they must not be conflated:

  **supplement**  SAM 3 as an extra silencing vote. Bounded by the 11 leaks: at most -0.37 cost per
                  clip, and it needs the same better-than-2:1 clearance OWLv2 just missed.
  **replace**     SAM 3 INSTEAD of the VLM's visibility vote, with DISPLAY_THRESHOLD then lowered.
                  This is the version with the real ceiling, because a gate that can be trusted buys
                  RECALL rather than precision. It is a full amendment: DEV selection, both arms,
                  one TEST look.

Full review of what the pipeline does and does not do with the video: docs/video_understanding_review.md

### Amendment 18c, outcome (2026-09-23): SAM 3 is a better detector and lands on the SAME line

Same harness, same concept phrases, same stretches as the OWLv2 run, so the two are comparable one
for one. DEV, 118 (sound, stretch) decisions, SAM 3 at its own default bar of 0.5:

                             SAM 3 sees it    SAM 3 blind
    VLM says visible               28              15
    VLM says NOT visible            9              66

**SAM 3 is visibly the better model, and it shows in the right place.** It disagrees with the VLM on
only 9 stretches where OWLv2 disagreed on 24 -- it is not seeing more things, it is seeing the right
things. But the 9 split into **6 wrong pictures removed and 3 needed sounds lost: a ratio of exactly
2.00**, against the beta = 2 break-even of 2.0. And unlike OWLv2, no threshold separates them: the
ratio is 2.00 at every bar from 0.25 to 0.70.

**Three different mechanisms have now landed on exactly 2.00**: the every-stretch gate rule
(amendment 14), the per-stretch OWLv2 vote, and the per-stretch SAM 3 vote. That is no longer a
coincidence worth remarking on -- it is the finding. **A visibility signal trades leaked pictures for
lost sounds at the break-even rate whatever model produces it**, because the two populations are the
same events seen from two sides. Better vision does not move that line; it only makes the
disagreements fewer and sharper.

**Why SAM 3 loses the three sounds, which is the useful part.** All three are the same clip and the
same family: `mv_protest_scene_movie` Crowd, at 0.0 s, 2.1 s and 7.1 s, with SAM 3 confidence
0.60-0.90. A crowd IS plainly on screen. The crowd SOUND the annotator marked as needed is off-screen
chanting. SAM 3 answers "is a crowd visible" correctly and the gate needs "is the crowd we can hear
the one on screen" -- presence is not source. That is the wrong-family problem relocated into the
visual channel, predicted before the run, and it is why a concept detector cannot simply replace a
visibility judgement.

The six pictures it correctly removes are Insect x4 in a pet shop (there are visible insects),
Train in a subway, and a Gunshot -- all cases where the source really is on screen.

**Consequences, stated plainly:**

  * The **supplement** route is closed. Three models, one line, no separation. It is not worth
    another experiment.
  * The **replace** route (SAM 3 instead of the VLM's visibility vote, then lowering
    DISPLAY_THRESHOLD) is now much less attractive than it looked an hour ago: the Crowd failure is
    what a replacement would do systematically, and a replacement has no second opinion to catch it.
  * What the result DOES establish, and belongs in the thesis: the gate's remaining errors are not a
    model deficiency. They are the boundary between "the object is on screen" and "the sound you
    cannot hear comes from that object", which no current vision model is asked to answer.

## Amendment 19 — audio-visual synchrony: FAILS, and is barely measurable (2026-09-23)

Adam's idea, and the only visibility signal nobody had tried: a source that is on screen AND making
the sound should MOVE when the sound happens; a bystander object should not. Measured as the change
inside the detected object's box DIVIDED BY the change outside it, so a panning camera cancels, at
the instant of the onset against that object's own usual level. Measured only where the detector
already finds the object, because otherwise the answer would merely repeat the detector's verdict.

Go/no-go, fixed before the run: a threshold must remove more than two leaks per needed sound lost.

    DEV: 10 gate leaks, 12 hits
    stretches where an object was confident enough to measure:  3 leaks, 2 hits

    leaks (annotator says the source IS visible)   median +0.835   mean +0.906
    hits  (source is off screen)                   median +54.9    mean +54.9

    best threshold found: 1 leak removed per 1 sound lost -> ratio 1.00

**FAIL, on the declared rule.** But the number that matters is the one above it: **only 5 of the 22
stretches had an object the detector was confident enough about to measure synchrony at all.** The
signal exists exactly where a detector already sees something, which is the case the gate is already
handling. Even a perfect synchrony rule could touch a fifth of the cases.

Worse for the hypothesis, the direction is BACKWARDS in this sample: the off-screen-source cases show
MORE local change (+54.9 median) than the on-screen-source cases (+0.84). The extreme value is
`mv_protest_scene_movie` Crowd at +108 -- a protest crowd churns continuously whether or not the
chanting being scored comes from off camera. Motion is not causation, and a crowd is the case where
that is most obviously true.

**Stated honestly: with three leaks and two hits this is not a clean negative, it is an
inconclusive test on a sample too small to settle anything.** What it does establish is the
structural point -- synchrony is only measurable where an object is already detected, which is a
fifth of the relevant stretches, so it cannot be the lever even if the mechanism were sound.

### What the four visibility experiments together now say

    every-stretch gate rule          4 leaks removed / 2 sounds lost     ratio 2.00
    per-stretch OWLv2 vote          16 leaks removed / 8 sounds lost     ratio 2.00
    per-stretch SAM 3 vote           6 leaks removed / 3 sounds lost     ratio 2.00
    audio-visual synchrony           1 leak  removed / 1 sound  lost     ratio 1.00 (n = 5)

Three independent mechanisms on the break-even line, and the fourth not measurable. **The visibility
side of this pipeline is closed.** The gate's remaining errors are not a model deficiency and not a
missing signal; they are the boundary between "this object is on screen" and "the sound you cannot
hear comes from that object", which is a question no current model is asked and which this project
cannot answer with the data it has.

## Amendment 20 — SAM 3 re-tested on OUR question and OUR clips: still loses (2026-09-23)

Adam's objection to the September SAM 3 comparison was correct. That run used DCASE clips, where
"visible" means the source is geometrically inside the field of view. This project's question is the
annotator's: *can a viewer see the thing making this sound* — a car behind a wall is in frame and not
visible, a crowd on screen is not the source of chanting from around the corner. SAM 3 had never
been asked our question.

`benchmark/gold/visibility_models.py` (job 30991800, H200) asks all three models the same question
on the DEV clips, scored against the annotator's own `visible` tick on each gold sound of importance
2-3 (35 sounds: 16 visible, 19 not). Each model's own best bar, swept until the peak was interior:

    model            best bar   agreement   right when visible   false alarm
    OWLv2              0.10       71.4%          14/16               8/19
    SAM 3              0.10       62.9%          10/16               7/19
    the VLM (gate)      -         66.7%          10/16               5/17

The first sweep reported SAM 3 at 57.1% because both models peaked at the lowest bar tried; the
numbers above are from the re-sweep on the saved scores (`visibility_models_dev.json`), where each
peak is interior to its range. The conclusion does not change: **OWLv2 stays the stage-2 default,
and the shipping VLM gate is already within 5 points of the best detector.**

This is the second independent confirmation of amendment 19's finding, and the stronger one. SAM 3
scores roughly double OWLv2 on open-vocabulary detection benchmarks and *loses* here by 8.5 points.
A model gets better at *presence* without getting better at *source*, because presence is what it is
trained and measured on. Reported against interest: this was run to give SAM 3 its fair chance, and
it was the objection of the project's own author that prompted it.

## Amendment 21 — the final like-for-like TEST table (2026-09-25 evening, before any render)

Panel of five, four rounds (`docs/panel_2026-09-26_plan.md`, signed 5/5; D2 carried 4–1, P5 dissenting: "a fifth
read"). Proposed: render ours, blind and ungated text tags on TEST at the shipped configuration (onset rule on, cap
off), one card class (H200/A100), each arm copying the base env of `test_monocap_v31` (`V4=590 FBAR=0.8 VETO=0.3
PVETO=0.05`, `MAXSPAN=none`), checked in the `[v4]` log lines. **Both the render and the scoring wait for Adam's
explicit yes** (a TEST render of a baseline is a further look — `docs/NIGHT_REPORT_2026-09-25.md` §5).

Committed now: *If Adam says yes, this table replaces the 23 Sep table as the thesis's TEST table whatever it shows;
the 23 Sep table moves to the read history. If Adam says no, the 23 Sep table stays the headline and the onset fix
is reported as DEV-selected, TEST inconclusive.* The table is scored once, with the two Holm families of the plan
(§A.2–A.3) and the category rows, and the read joins the history as the fifth; no other TEST number is read after
it. Guards: no scorer, judge or audit is pointed at the TEST tag before the yes; the TEST directories are excluded
from every both-halves script; job logs are read for completion and errors only.

Adam's decision: **YES** — render and score (chat, 2026-09-25 21:50 JDT: "2 yes"). Second annotator: yes, DEV/TEST clips only, no slice B ("because they are weird").

### Amendment 21 — result (scored once, 2026-09-26; `benchmark/gold/holm_table.py`, `holm_test_final_v33_test_bench.json`)

Renders `test_final_v33` (ours and blind on H200, ungated text on A100; `[v4]` lines checked: FLEXSED_BAR 0.8,
PANNS_VETO 0.05, FLEXSED_VETO 0.3, ONSET_MONOTONE True, MAX_SPAN None); completeness guard 60/60 for every arm. (A
first call named the subset `test`, which includes slice B; the guard stopped it before any number was computed; the
declared 60 TEST clips are `test_bench`.)

    60 TEST clips, 43 needed sounds     F1     P      R      FA/clip  cost
    ours                                0.381  0.390  0.372  0.42     2.63
    blind                               0.322  0.253  0.442  0.93     3.47
    silence                             0      -      0      0        2.87

    PRIMARY  dF1 ours - blind +0.059 [-0.030, +0.144]  p 0.183   null
    FAMILY 1 (Holm over 7)        d      95% CI              p       Holm
      dP                       +0.137 [+0.043, +0.256]   0.002   0.010   survives
      dFA/clip                 -0.517 [-0.800, -0.283]  <0.001  <0.001   survives
      d cost/clip              -0.833 [-1.434, -0.300]   0.003   0.012   survives
      d clean-clip accuracy    +0.219 [+0.086, +0.364]   0.001   0.006   survives
      dF0.5                    +0.110 [+0.017, +0.210]   0.023   0.069   -
      dR                       -0.070 [-0.154, +0.000]   0.103   0.206   -
      dwF1                     +0.022 [-0.074, +0.103]   0.580   0.580   -
    FAMILY 2 (Holm over 3)
      F1 vs silence (all)      +0.381 [+0.217, +0.557]  <0.001  <0.001   survives (= ours' F1)
      cost vs silence (all)    -0.233 [-1.000, +0.467]   0.585   0.585   -
      cost vs silence (unseen, 13; pre-declared subgroup of a post-hoc metric)
                               -3.077 [-4.769, -1.385]  <0.001  <0.001   survives
    CATEGORY (CIs only): mixed d cost vs blind -1.73 [-3.73, -0.13]; seen -1.83 [-2.83, -1.00];
      unseen +0.00 [-0.62, +0.77]; no-ambient -0.10 [-0.30, +0.00]

By the committed rule this table **replaces the 23 Sep table** as the thesis's TEST table; the 23 Sep table moves to
the read history as read 4, this is read 5. No further TEST number is read. DEV like-for-like under the same code
(`holm_dev_monocap_v31_dev.json`): primary +0.048 null; family 1 survivors dFA, d cost, d clean-acc (dP p 0.020, Holm
0.080, does not survive); family 2 survivors F1 vs silence only (unseen cost -0.86, p 0.091).

### Correction (2026-09-26, after the fifth deliberate TEST read) — two DEV/TEST definitions

The sentence at l.1487 ("the split was made once … stratified") is incomplete. Two definitions coexist:
`benchmark/gold/split.json` (DEV 79 / TEST 60, seed 7; read by `detector_bench.py`) and the scorer's amendment-5 subsets
(DEV = gold ∩ `judge100.txt`, 49; TEST = `test_bench`, 60; slice B separate). **35 of the scorer's 60 TEST clips are in
split.json's DEV-79; 20 of the scorer's DEV-49 are split.json TEST** (recomputed 2026-09-26 from `split.json`,
`judge100.txt`, `audioset_slice.json`). FlexSED's bar 0.8 (amendment 8) and the detector-level rejections (amendment 12,
GOLD §13) were selected on split.json DEV-79. The vetoes (amendments 10, 16), the onset rule and every end-to-end DEV cell
used the clean DEV-49. Mitigation: the DEV-49 end-to-end grid also selected 0.8 (cost 3.63 / 4.61 / 5.43 for 0.8 / 0.7 /
0.6); amendment 8's criteria were gate-blind (a shared stage, applied to both arms). A re-check of amendment 8's three
rules on DEV-49 from caches is reported beside this note. The original line stays as written.

### Derived gated text arm on DEV (plan B.4; declared decision-free, 2026-09-26)
`derive_gated_text.py --src dev_monocap_v31 --new dev_gtext_v34`: spans identical to ours on 49/49 clips. Direct judge
(Gemma-4-31B, `judge_direct_dev_gtext_v34.json`, 98 rows, 0 unparsed); trust checks pass for both arms (text B2 gap
+0.78 [+0.12, +1.40]; pictures +0.98 [+0.24, +1.68]). Pictures − same-gate text tags = −0.04 [−0.22, +0.14] (paired
clip bootstrap, 2000, seed 0; the two differ on 7 of 49 clips): a tie, as expected (the judge reads the text tag as the
label itself). The ungated text row stays in the table as the baseline, with its confound stated.
Read count (2026-09-26): amendment 21 is the **fifth deliberate read** of TEST and the **tenth exposure** counting prints
and diagnostics (`docs/LEDGER_2026-09-26.md`, audit finding 1).

### Amendment 8 re-checked on the clean DEV-49 (week plan B.1, 2026-09-27; report only, rules unchanged)
`benchmark/gold/flexsed_recheck_dev49.py` → `flexsed_recheck_dev49.json`. Rule 1 (masked sounds recovered ≥ 44 %):
**6 of 7 = 86 %, PASS** (at FlexSED's shipped bar 0.8 only 1 of 7). Rule 2 (union ≥ +0.05 onset-recall over BEATs at
the same false-label rate): union 0.576 at 1.51 false labels/clip vs BEATs interpolated 0.536 → **+0.039, FAIL**
(narrowly). Rule 3 (no-ambient false labels ≤ 2×): 1.44 vs 0.78 = **1.86×, PASS**. By the rule written before the run,
nothing is reverted and the thesis states: *"FlexSED's bar was selected on a split (split.json DEV-79) that overlaps 35
of the 60 TEST clips; on the clean DEV-49 it passes two of its three adoption rules and misses the third by 0.011."*

### Per-vote table (week plan B.4, 2026-09-27; report only) — `gate_vote_table.py`, DEV gold sounds, importance ≥ 2
    vote alone             seen silenced   needed kept   balanced
    name (open naming)     0.44 (19/43)    0.83 (30/36)  0.64
    a/b (both orders)      0.26 (11/43)    0.94 (34/36)  0.60     the safety vote: almost never silences a needed sound
    desc (description)     0.37 (16/43)    0.75 (27/36)  0.56
    majority of 3 (shipped)0.37 (16/43)    0.86 (31/36)  0.62
All within the ±0.11 half-width of balanced accuracy on 79 sounds; nothing is changed (choosing "name alone" after
seeing this table would be selection on DEV, and the difference is noise).

### Cost curve of the final tables (plan §A.5, 2026-09-27) and a scoring-configuration fix
`cost_curve.py` called `config.use_v4("59")`, which (a) switches the 8-s display cap back on (stage "9") and (b) sets
LABEL_FILTER "depictable", under which gold sounds the filter never draws stop counting as needed (DEV 36 → 33). Every
per-sound table, including amendment 21, is scored with `score_per_sound`'s own configuration (no cap on uncapped rows;
those sounds still count as needed, and both systems miss them equally). New flags `--maxspan none --match-scorer`
make the curve reproduce the scored tables exactly (β = 2: TEST ours 2.63 / blind 3.47 / silence 2.87; DEV 2.78 / 3.59 /
2.94). Crossings: ours beats blind for β above 0.39 (TEST) / 0.46 (DEV); ours beats silence for β below 2.56 (TEST) / 2.33 (DEV).
`cost_curve_test_final_v33.png`, `cost_curve_dev_monocap_v31.png`. (The oracle line reuses the cached gate votes of
`gate_gold`, frames sampled around the gold sounds.)

### Stage 4 on the 280-clip AudioSet-Strong calibration set (plan B.4 / D6, 2026-09-27; descriptive, no selection)
`benchmark/audioset_stage4_report.py` → `audioset_stage4_report.json`. Shipped bars; stage 4's own union/veto rules on
the cached frame scores (onset refinement not re-run). Non-speech, non-music events; recall = matched by family with
≥ 0.5 s overlap; "consequential" and "masked" as tagged in the calibration set (31 masked consequential events).

    row                              masked-conseq  conseq   all    onset-recall  false/min  onset MAE
    A BEATs 0.35                         38.7 %      54.5 %  34.8 %    27.7 %       6.41      2.61 s
    B FlexSED 0.8 alone                   0.0 %      15.6 %  12.3 %    13.4 %       2.87      0.77 s
    C union                              38.7 %      58.9 %  38.3 %    30.8 %       8.83      2.41 s
    D + FlexSED veto 0.3                 38.7 %      53.1 %  35.4 %    27.7 %       6.21      2.41 s
    E + PANNs veto 0.05 (shipped)        38.7 %      51.3 %  33.7 %    25.9 %       4.56      2.49 s
Reading: on out-of-sample AudioSet-Strong clips the shipped stack keeps BEATs' recall within about 3 points and cuts
false spans per minute by 29 % (6.41 → 4.56) — the same direction as on the gold (precision up, recall slightly down).
FlexSED at bar 0.8 recovers none of the 31 masked consequential events here (its score scale is family-dependent; the
per-family bars fitted on this very set were not adopted), unlike on the gold DEV set; the union's masked recall equals
BEATs'. Onset errors are the extractor's (≈ 2.5 s mean on 10-s clips), not the shipped refined onsets.

### Gate frame-shift stability (week plan B.3, 2026-09-27; report only, rule written first: > 10 % flips = frame-sensitive)
`gate_gold.py --dev-only --shift 0.5` and a same-frames repeat (`--suffix _repeat`), Qwen3.8-27B, H200;
`gate_shift_compare.py` on the 79 DEV gold sounds rated ≥ 2: the repeat flips **0** verdicts (deterministic); a +0.5-s
shift of every frame flips **6 (7.6 %) → stable within 10 %**. Of the six, five move from silent to drawn (four of them
sources the annotator marked visible — a machine gun, a glass clink and shatter, a phone buzz — and one a vehicle marked
not visible), one from drawn to silent (visible water). The 1-of-60 TEST difference between card classes is consistent
with this. Reported as: the gate is repeatable and changes about one verdict in thirteen when its frames move by half a
second.
Note to the B.1 re-check (2026-09-27): at FlexSED's **shipped** bar 0.8 only 1 of the 7 masked DEV-49 sounds is recovered
(and 0 of 31 masked consequential events on AudioSet-Strong, D6). Amendment 8's "7 of 9 recovered" counted scores of
≈ 0.35–0.49 and above, below the adopted bar. The union's recall gain at bar 0.8 therefore comes from other sounds, not
from the masked sounds that motivated FlexSED; the thesis states this.

## Amendment 22 — the detector round (2026-09-27, written before any cell is scored; TEST is not touched)

Adam: *"we HAVE to recognize more sounds"*; *"try it on DEV without touching TEST"*. Panel (F1, F4, P2;
`docs/panel_2026-09-27_detector_brief.md`). Motivation, not a selection source: on DEV-49 the caches reach 30 of 36 needed
sounds at looser bars vs 21 at the shipped ones (only 4 below every detector) — an oracle-bar ceiling read on the very
sounds it would be scored on, so **no bar is chosen from it**.

**Cells (fixed; floors a priori, not swept):** every cell = the shipped stack (BEATs 0.35 ∪ FlexSED, FlexSED veto 0.3,
PANNs veto 0.05, onset rule, no cap, `config.use_shipped()`) with one change —
- A: FlexSED bar 0.6 · B: FlexSED bar 0.5 (the lower-bar × PANNs-veto cell was never run: the 0.7/0.6 costs of amendment
  10 had the FlexSED veto only);
- C: bar 0.6 + tier 2 · D: bar 0.5 + tier 2 — tier 2 (`FLEXSED_CORROB = (0.1, 0.05, 1.0)`): a FlexSED-only span is admitted
  only if BEATs ≥ 0.1 or PANNs ≥ 0.05 rises for the same family within 1 s of it (0.05 = the shipped PANNs veto; never tried
  time-aligned);
- E: D + tier 3 (Adam's cascade, `BEATS_LOWBAND_CORROB = (0.3, 0.05, 1.0)`): a BEATs span with peak in
  [AED_THRESHOLD 0.175, 0.35) — heard but too weak to show — is promoted to the display bar if FlexSED ≥ 0.3 (the shipped
  FlexSED veto τ) or PANNs ≥ 0.05 rises for the same family within 1 s.

**Stage 0 — pick (280 AudioSet-Strong calibration clips, detector level, cached scores):** the cell with the highest
consequential onset-recall among the cells whose false spans/min ≤ the shipped stack's on the same clips (4.56). If no
cell qualifies, nothing is picked and the round ends (reported). Masked recall reported, not used.

**Stage B — DEV-49 (real renders, both arms, H200, base env copied, `[v4]` lines checked), the BLIND arm decides:** the
picked cell passes iff the blind arm's hits ≥ shipped blind hits (17) **and** its viewer cost (β = 2) ≤ shipped blind cost
(3.59) **and** its blind F1 paired CI vs shipped blind is not entirely below 0. The gated arm (ours) is printed for every
cell, never selects.

**Stage A — held-out confirmation (only if Adam approves downloading a new, disjoint AudioSet-Strong evaluation set; N
and seed written before the download):** paired clip bootstrap Δ consequential onset-recall (picked − shipped) with lower
CI > 0, false spans/min ≤ shipped on the same clips, masked recall not lower.

**Outcome:** pass B (+ A if run) → reported as a **detector upgrade beside the frozen system** ("DEV-49 + AudioSet-Strong,
not confirmed on TEST"); the amendment-21 TEST table stays the thesis's TEST table; no TEST read. Fail → reported as a
negative, nothing changes. Not allowed: choosing by ours' F1 or ΔF1, choosing on the 36 DEV sounds' scores, any TEST
number.

**Amendment 22, addendum (2026-09-27, before any cell is scored).** (1) *Cell F — a union bug found by trace:* in the
twin rule a BEATs span too weak to be shown (peak < 0.35) absorbed a FlexSED span above FlexSED's own bar, and the merged
sound was then dropped at the display bar; on DEV-49 six FlexSED detections at 0.89–0.93 vanished this way (one at a
needed sound: rainforest insects). Cell F = shipped + `UNION_WEAK_TWIN = "ignore"` (only a displayable BEATs twin absorbs;
otherwise the FlexSED span stays FlexSED-only and faces the PANNs veto). It is judged by the same Stages 0/B/A; if it
passes it is reported as a bug fix. (2) *Disclosure:* while the rule was being written, reviewer F1 ran a detector-level
dry sweep of lower FlexSED bars on DEV-49 from the caches (bar 0.5 + both vetoes: onset-recall 0.545 vs 0.515, false
labels/clip 0.45 vs 0.24). It chose nothing: the pick is made on the AudioSet-280 set (Stage 0) and decided by the blind
arm's rendered DEV numbers (Stage B).

### Amendment 22 — Stage 0 result (2026-09-27; `benchmark/detector_round_stage0.py` → `detector_round_stage0.json`)
280 AudioSet-Strong calibration clips, stage-4 logic from caches (shipped stack recomputed by the same code: 4.46 false
spans/min; the D6 report's 4.56 omitted the weak-twin absorption).

    cell                    conseq onset-recall  conseq recall  masked (31)  false/min
    shipped (0.8)                 25.0 %            50.4 %        38.7 %       4.46
    A bar 0.6                     25.4 %            52.2 %        38.7 %       4.89
    B bar 0.5                     25.4 %            53.6 %        38.7 %       5.12
    C bar 0.6 + tier 2            25.4 %            52.2 %        38.7 %       4.61
    D bar 0.5 + tier 2            25.4 %            53.6 %        38.7 %       4.80
    E D + tier 3 (cascade)        46.4 %            58.9 %        45.2 %       6.90
    F twin fix (0.8)              25.9 %            51.3 %        38.7 %       4.56
**By the rule written first, nothing is picked:** no cell stays at or below the shipped false-span rate while raising
onset-recall. Amendment 22 ends here: no cell is adopted through it. The twin fix (F) changes almost nothing (+0.9 pts
onset-recall, +0.10 false spans/min). Tiers 1–2 (lower FlexSED bars, time-aligned corroboration of FlexSED) buy ≤ 0.4 pts
onset-recall — consistent with F1's diagnosis that the corroborating detectors are deaf where BEATs is deaf.

## Amendment 23 — the cascade (cell E) as a new question (2026-09-27, written after Stage 0, before any DEV number)

Stage 0 showed one cell with a large recall gain at a false-alarm price (E: consequential onset-recall 25.0 → 46.4 %,
recall 50.4 → 58.9 %, masked 38.7 → 45.2 %, false spans/min 4.46 → 6.90). Whether that trade is worth it to a viewer is a
new question, motivated by the Stage-0 table (disclosed) and tested on data that has not seen cell E: the DEV-49 renders of
E (`dev_fbar05cl_v35`, both arms, H200, base env, `[v4]` lines checked; F1's earlier DEV dry sweep did not include tier 3).

**Rule (the blind arm decides; the price of a wrong picture is the declared β = 2):** E passes iff, against the shipped
blind arm on DEV-49 (`dev_monocap_v31`: 17 hits, cost 3.59), the E blind arm has **more hits (≥ 19, i.e. + 2)** and a
**viewer cost at β = 2 not higher (≤ 3.59)**. Reported beside it: blind F1/P/R with paired CIs, the gated arm (ours) for
information, the β at which E and shipped cross, and every added hit and added wrong picture by name. Pass → reported as a
**detector upgrade on DEV + AudioSet-280 (fit set), not confirmed on held-out data or TEST**, beside the frozen system;
a held-out AudioSet-Strong confirmation (Stage A of amendment 22) is then required before any stronger wording, and needs
Adam's approval to download. Fail → reported as a negative. TEST is not touched either way.

## Amendment 24 — detector round 2: a new corroborator, a cost rule, a held-out set of complex scenes (2026-09-27, written before any number; TEST is not touched)

Source: the three-reviewer detection panel (`docs/panel3_topic1_rounds.md`, rounds 1–3, signed by all three). Adam's
direction (27 Sept): "do anything you can to improve the detector — we want complex-situation audio, not clean audio";
his yes to a new AudioSet-Strong download was given the same day.

**Why.** Amendment 22 showed the recall is in the caches but every route to it pays ≈ 2.4 false spans per gained onset
(≈ 6 per gained event) on the 280, above the 2.0 break-even at β = 2; the masked sounds are heard only by FlexSED, and
every corroborator tried (BEATs, PANNs, CLAP, PSED, AST, CED) is deaf there. One public-weights model not yet tried is a
text-queried frame-level detector with an encoder independent of FlexSED's: **PE-A-Frame** (`facebook/pe-a-frame-large`,
Apache-2.0, PE-AV arXiv 2512.19687; `transformers.PeAudioFrameLevelModel`, imports in env msproj, transformers 5.16.1).
Its cache: the 215 depictable families (`benchmark/gold/depictable_vocab.json`), query "The sound of {family}" (FlexSED's
wording), 48 kHz mono, one score per 40 ms (sigmoid of the model's own logit scale and bias), stored as FlexSED's npz
(`fw` [labels, T], `labels`, `fps` 25). Nothing in the cache is chosen from gold.

**Cells, fixed a priori** (stage-4 logic from caches, `benchmark/detector_round_stage0.py`):
shipped; F (twin fix); E (amendment 22: FlexSED bar 0.5 + tier 2 + tier 3); **E-F** as E, with tier 3 = (0.3, 1.01,
1.0) and (0.5, 1.01, 1.0), i.e. FlexSED-only corroboration; **E-AND** as E, with tier 3 = FlexSED ≥ 0.3 **and** PANNs ≥
0.05 within 1 s (closes tier 3's bypass of the PANNs veto);
**D+PE** FlexSED bar 0.5, PE-A-Frame ≥ θ within 1 s replaces the PANNs veto on FlexSED-only spans; **E+PE** as D+PE plus
PE-A-Frame ≥ θ as the tier-3 corroborator instead of FlexSED/PANNs; **U+PE** shipped stack plus PE-A-Frame spans at bar
θ_u (a third detector) kept only if FlexSED ≥ 0.3 for the same family within 1 s — **U+PE was added by the coordinator,
not proposed by a reviewer** (disclosed; multiplicity is still one cell to held-out). **Grid for θ and θ_u:** PE-A-Frame's
score is sigmoid(logit × learned scale + learned bias), so its scale is unknown; the grid is the 50/80/90/95/99th
percentiles of all per-frame, per-family scores over the 280, computed once from the cache and written to the log before
any cell is scored.

**Screen for PE-A-Frame** (before any PE cell is scored): on the 280's E-delta pool (spans cell E raises that shipped does
not show; positives counted and written into the log before scoring) — AUROC of PE-A-Frame's same-family peak within 1 s
reported with a 2000-draw bootstrap CI beside PANNs' on the same spans; at the loosest grid θ (percentile grid above) that keeps ≥ 80 % of true
items, kept-false/kept-true ≤ 0.84 (p > 1/3). Fail → no PE cell is scored; reported as a negative.

**Pick (fit set = the 280, called a fit set, not out-of-sample).** Per clip C = 4 × consequential events with no
same-family span overlapping (≥ 0.5 s or half the event) + 2 × false spans; C-onset the same with the [−0.5, +1.0] s onset
window. θ / θ_u per PE cell = argmin C-overlap on the 280. **The pick is the one cell with the lowest mean C-overlap that
also has C-onset below shipped's; none → the round ends and the negative is written.** Each cell is labelled "strict
pass" (false spans/min also ≤ shipped) or "trade pass". Only the pick goes on (multiplicity: one cell reaches held-out).

**Held-out set (new, disjoint).** AudioSet-Strong evaluation split (`audioset_eval_strong.tsv`), seed 23, disjoint from
the 280 (+ their 40 missing), slice B (111 + 39 missing) and every gold clip whose name is an AudioSet segment id (73 in
gold_AG.json, 4 of them outside slice B's list). **N = 500 requested: 300 complex + 200 random.** The panel signed
250 + 250; the coordinator moved it to 300 + 200 on Adam's direction of 27 Sept (complex scenes); the random part stays
because false alarms on quiet clips are part of C. Complex = Speech or Music covers ≥ 50 % of the clip **and** at least one labelled non-speech, non-music event lies ≥ half
under Speech/Music (masked). The id list is written to `benchmark/gold/audioset_heldout.json` by
`benchmark/gold/audioset_heldout.py --dry` and committed before any download. Caches: BEATs, PANNs, FlexSED, PE-A-Frame,
same code as the 280.

**Held-out test.** The pick passes iff the paired clip bootstrap of ΔC-overlap (pick − shipped), 2000 draws, seed 0, over
all held-out clips has **upper 95 % CI < 0**. Reported beside (no gate): ΔC-onset, Δ consequential recall, Δ onset-recall,
Δ masked recall, Δ false spans/min — each per stratum (complex / random) — and the β at which the pick crosses shipped.

**Then.** Pass → DEV-49 render of the pick, both arms, H200, base env, amendment 23's rule as a sign check, reported as
"detector upgrade beside the frozen system, DEV point estimate, not confirmed on TEST"; the amendment-21 TEST table stands.
Fail → the thesis reports the exchange rate on the 280 and held-out as the detector ceiling. Cell E's amendment-23 DEV
render runs anyway and is reported as its own question. No TEST number; no bar from DEV scores; this rule is not revised
after a number is seen.

**Amendment 24 — clarification 1 (2026-09-27, before any PE-A-Frame or round-2 number exists).** "At the loosest grid θ that
keeps ≥ 80 % of true items" is read as **the operating point at the 80 % recall floor: the highest grid θ that still keeps
≥ 80 % of the pool's true items**, where kept-false/kept-true is computed (the panel's wording, T1-a round 2 and T1-c
round 3: "at the operating point keeping ≥ 80 % of true items"). The script is `benchmark/detector_round2.py`
(steps `screen`, `fit`, `heldout`), committed with this line. Every cell is scored on the clips that have all four caches
(BEATs, FlexSED, PANNs, PE-A-Frame).

### Amendment 23 — result (2026-09-27; DEV-49 blind arms, `score_per_sound.py --subsets dev`, base env checked in `[v4]` lines)

    blind arm                    hits  miss  wrong pictures (visible / cross / phantom)   viewer cost β=2   P     R     F1
    shipped  dev_monocap_v31      17    19     50  (17 / 21 / 12)                           3.59           0.25  0.47  0.33
    cascade E dev_fbar05cl_v35    17    19     80  (25 / 35 / 20)                           4.82           0.18  0.47  0.26
**Fail on both parts of the rule:** no added hit (17 vs ≥ 19 required) and 30 more wrong pictures (cost 4.82 vs ≤ 3.59).
The recall gain cell E showed on the 280 (onset-recall 25 → 46 %) does not reach a single needed DEV sound; it matches
the panel's forecast (≈ 2 new sounds at most, most of the 280's gain was earlier starts on sounds already found). Cell E
is not adopted. Amendment 24 (committed before this number) is unaffected.

**Amendment 24 — clarification 2 (2026-09-27, before any screen or cell number).** A probe on 12 calibration clips (the fit
set; no cell, no pool, no gold) checked that the cache is sane: the labelled families rank 3rd–4th of 215 by clip peak
(76 % in the top 10), but the sigmoid saturates — a median of 148 of 215 families exceed 0.5 in every clip, and many peak at
1.000. So the cache stores the model's **logit** (its learned scale 2.30 and bias −10.0 applied), and every reader uses
**PE score = the family's logit minus the median logit over the 215 families in the same frame** (how much the family
stands out at that moment). This replaces the sigmoid everywhere in amendment 24 (screen, θ grid percentiles, D+PE, E+PE,
U+PE); no second variant is scored. The 3 sigmoid smoke-test files are deleted and recomputed.

### Amendment 24 — result (2026-09-27; `benchmark/detector_round2.py` → `detector_round2.json`; the 280 = fit set)

**Screen.** PE-A-Frame score grid (percentiles 50/80/90/95/99 over all frames and families): 0.00 / 2.25 / 3.72 / 5.20 /
8.73. E-delta pool: 216 spans, 118 true, 98 false (true = overlaps a labelled event of the same family). At the operating
point keeping ≥ 80 % of true spans (θ = 5.20): 99 true, 75 false kept, ratio 0.76 ≤ 0.84 → **passes by the written rule —
but only because the pool was already at 0.83 before filtering**: PE-A-Frame's AUROC on the pool is **0.53 [0.45, 0.61]**,
chance level, below PANNs' 0.64. Reported as it is: the rule let the PE cells be scored; the ranking says PE-A-Frame does not
separate true from false candidates here.

    cell       C-overlap  C-onset  recall  onset-recall  false/min
    shipped      3.071     3.886   50.4 %    25.0 %        4.46
    F            3.079     3.893   51.3 %    25.9 %        4.56
    E            3.614     4.014   58.9 %    46.4 %        6.90
    E-F 0.3      3.529     3.971   58.9 %    45.1 %        6.64
    E-F 0.5      3.321     4.107   58.9 %    34.4 %        6.02
    E-AND        3.286     3.757   58.5 %    43.8 %        5.87
    D+PE         3.136     4.050   53.6 %    25.0 %        4.95   (θ from the grid, argmin C-overlap)
    E+PE         3.329     4.257   58.5 %    29.5 %        6.00
    U+PE         4.750     5.536   52.7 %    28.1 %        9.71
**No pick: no cell lowers C-overlap below shipped's 3.071.** E-AND is the only cell that lowers C-onset (3.757 vs 3.886) and
it raises C-overlap. The round ends here, as written; **the held-out set is not scored** (its caches are kept, unread, for a
future pre-registered question). Reading: the shipped stack sits at the cost optimum of every training-free stack tried;
extra recall costs more false spans than β = 2 allows, and a second text-queried detector with an independent encoder
(PE-A-Frame) does not tell masked true sounds from false ones. This is the detector-ceiling result for the thesis.

Definition note to the result above: the screen's "true" is span-level (overlaps any labelled event of the same family,
consequential or not), so the pool's 118/98 is not the panel's ~48 true onsets / ~114 false spans (consequential onsets
only); C itself counts only consequential misses. This is one reason the ratio screen was near-trivial.

## Amendment 25 — last detector question: an audio "listener" as verifier (2026-09-27, after amendment 24's result, before any listener output)

**Trigger, disclosed as post hoc.** All three detection reviewers listed a per-candidate audio-LLM verifier as the fallback
"if PE-A-Frame fails its screen". PE-A-Frame passed amendment 24's ratio screen only by the letter (AUROC 0.53 [0.45, 0.61],
below PANNs' 0.64); the coordinator triggers the fallback on that, after seeing the screen. Adam authorised "anything you
can to improve the detector" (27 Sept).

**Listener.** Qwen3-Omni-30B-A3B-Instruct (Apache-2.0; `Qwen3OmniMoeForConditionalGeneration`, text output only), one
bf16 load on an H200/A100-80. Per candidate span: the clip's audio from start − 1 s to end + 1 s (clipped to the clip), 16
kHz mono, and the question "Is the sound of {family} present in this recording? Answer yes or no." Score = logit(yes) −
logit(no) at the first generated token (max over the "yes"/"Yes" and "no"/"No" token ids). **Null control:** each window
is also asked about one family from the 215 that is labelled nowhere in that clip (seeded, seed 0); the share of yes
(score > 0) on those is reported as the listener's yes-bias. **Load gate:** if 10 windows are not scored within 2 h of
wall time, stop and report.

**Pool (the 280).** Spans cell E raises that the shipped stack does not show (as amendment 24), each labelled **hit** (overlaps
a consequential event of its family by ≥ 0.5 s or half the event), **false** (overlaps no labelled event of its family) or
**neutral** (overlaps only non-consequential events of its family). Counts written to the log before any score.

**Screen (hard gate this time; the ratio-only screen of amendment 24 let a chance-level corroborator through).** On hit vs
false spans: listener AUROC with a 2000-draw bootstrap CI; pass iff the **lower 95 % bound ≥ 0.70 and the point AUROC > PANNs'**
on the same spans. Fail → no listener cell is scored; the round and detection are closed with a negative.

**Cells (only if the screen passes).** E+L and E-AND+L: E (resp. E-AND) where every span not shown by shipped is kept only if
the listener score ≥ θ; θ from the 20/40/60/80th percentiles of the pool's listener scores, argmin C-overlap on the 280.
**Pick, held-out test and DEV sign check: exactly amendment 24's** (lowest C-overlap that also lowers C-onset below shipped;
one cell; held-out 415 clips, paired ΔC-overlap upper 95 % CI < 0; then DEV-49 renders both arms as a sign check). No TEST.
This is the last detector question before submission; the rule is not revised after a number is seen.

**Amendment 25 — pool counts (written before any listener score):** the 280 give 214 candidate spans (union of the spans E
and E-AND add over shipped): **14 hit, 96 false, 104 neutral**. With 14 positives the AUROC interval is wide; the gate
(lower bound ≥ 0.70) stands as written.

### Amendment 25 — result (2026-09-27; `benchmark/listener_round.py screen` → `listener_round.json`, `listener_pool_calib.json`)

Load gate met (model loaded in 18 min; 10 windows in 1070 s wall < 7200 s). On the 280's pool (14 hit / 96 false spans):
listener AUROC **0.661 [0.525, 0.784]**, PANNs 0.574 on the same spans. The listener is a sane model — it says yes to 84 %
of the asked families and to only 7 % of families absent from the clip — but on the candidates the stack is unsure about it
says yes to most false ones too. **Gate failed (lower bound 0.525 < 0.70): no listener cell is scored, the held-out set stays
unread, and the detector question is closed** as written. Thesis reading: at the training-free frontier, neither a second
text-queried detector (PE-A-Frame) nor an audio LLM separates the masked true sounds from the false candidates well enough to
pay at β = 2; the shipped stack stays.

## 2026-09-28 — shipped picture model and a TEST inspection (Adam's decisions)

1. **Qwen-Image is the shipped picture model; FLUX is retired** (Adam, 28 Sept: "qwen is confirmed, use qwen, remove flux").
   `config.use_shipped()` now = the scored stack (`use_scored()`, what the amendment-21 table ran) + the frozen final
   picture setup (`use_final_pictures(2)`) + full-opacity pictures. The per-sound tables do not look at pictures, so no
   scored number changes. Disclosed: the picture claim still rests on the author-rater round 2 (Qwen-Image 26/54 vs FLUX
   14/54, one wrong object); the sealed confirmation sitting stays available as the clean check.
2. **TEST exposure 11: inspection.** Adam asked for a viewer ("inspector") that shows every clip, sound and picture of every
   set we measure on, TEST included, with pooled DEV + TEST numbers. It re-presents existing renders and scores; nothing is
   re-rendered for scoring and no setting is chosen from it. From here on, anything changed after looking at TEST mistakes
   is reported as tuned on TEST.

## 2026-09-28 — one cheap extra detector check: all three must agree (Adam's request; written before the number)

On the 280 (fit set): cell **E+3** = cell E, where every span E adds over shipped is kept only if **FlexSED ≥ 0.5 and
PE-A-Frame ≥ 5.20 (the 95th-percentile grid value of amendment 24) and the listener says yes (score > 0)**, all three for the
span's family within 1 s. Fixed values, no grid. Same rule as amendment 24: it goes on only if it lowers C-overlap and
C-onset below shipped's; then the held-out test of amendment 24. Script `benchmark/agree3.py`.
**Result:** E+3 C-overlap 3.279 vs shipped 3.071 (worse), C-onset 3.707 vs 3.886 (better); recall 58.9 % vs 50.4 %, false
spans/min 5.89 vs 4.46. The agreement keeps every true span cell E adds but still lets through too many false ones; it does
not lower C-overlap, so by the rule it stops here (`benchmark/agree3.json`).

## 2026-09-28 — external baseline: one off-the-shelf detector + a picture for every detection (written before any number)

The supervisor asked for comparisons to existing approaches, not only ablations. Baseline (b): PANNs CNN14 (AudioSet,
frame-level, the most cited open sound-event tagger) alone, every drawable detection drawn, no gate. Threshold picked on DEV
by viewer cost at β = 2 from the grid {0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}; TEST scored once at that threshold; ours vs
PANNs paired clip bootstrap on F1, cost, precision, recall. Script `benchmark/gold/baseline_panns.py`. "Blind" is renamed
**"pipeline without gate"** in reports: same detector stack, label filter, subject text and pictures; only the visibility gate
is removed.
**Result (`benchmark/gold/baseline_panns.json`).** DEV picked threshold 0.5 (the grid's top; cost fell steadily with the
threshold, towards "show nothing", 2.94). TEST, 60 clips: PANNs alone 5/43 hits, 20 wrong pictures, P 0.20, R 0.12, F1 0.147,
cost 3.20; ours 16/43, 25 wrong, P 0.39, R 0.37, F1 0.381, cost 2.63. Ours − PANNs (paired clip bootstrap, 2000, seed 0):
**F1 +0.234 [+0.085, +0.409], precision +0.190 [+0.029, +0.378], recall +0.256 [+0.095, +0.452]**; cost −0.567 [−1.267, +0.100]
(not significant). Unlike the ablation, against an existing single-detector approach the F1 gain is significant.

**Same baseline with a state-of-the-art detector (2026-09-28, before the number; Adam: PANNs is old).** PretrainedSED
(Schmid et al., CP-JKU, ICASSP 2025; BEATs backbone fine-tuned frame by frame on AudioSet-Strong, 447 classes) alone, every
drawable detection drawn, no gate; same threshold grid chosen on DEV by cost at β = 2; TEST once. `baseline_panns.py --model psed`.
**Result (`benchmark/gold/baseline_psed.json`; its keys say "panns_every_detection" because the script is shared).** DEV
picked 0.5 (grid top). TEST: PretrainedSED alone 6/43 hits, 41 wrong pictures, P 0.13, R 0.14, F1 0.133, cost 3.83. Ours −
PretrainedSED: **F1 +0.248 [+0.100, +0.421], cost −1.200 [−2.168, −0.300], precision +0.263 [+0.093, +0.472], recall +0.233
[+0.073, +0.425]** — all four significant.

## 2026-09-28 — confidence floor 0.40, chosen on DEV, tested once on the held-out 415 (Adam's yes; before the number)

On DEV a floor of 0.40 (show a picture only if its detector confidence ≥ 0.40) kept all 14 hits and cut wrong pictures 24 → 20
(cost 2.78 → 2.61; `benchmark/gold/dev_post_filters.json`). Test: on the unread held-out AudioSet-Strong set (415 clips),
shipped stack vs shipped + floor 0.40, amendment 24's cost C (overlap primary); passes iff the paired clip bootstrap ΔC-overlap
(2000, seed 0) has upper 95 % CI < 0. The 280 fit set reported beside. Script `benchmark/floor_heldout.py`.
**Result (`benchmark/floor_heldout.json`): passes.** Held-out 415: C-overlap 1.923 → 1.696, ΔC **−0.227 [−0.304, −0.154]**;
recall 49.1 % → 48.5 %; false spans/min 3.25 → 2.54. (The 280 fit set: −0.129 [−0.250, +0.036].) Adopted: `use_shipped()` sets
`PICTURE_MIN_CONF = 0.40` (display-level, as tested on DEV). The scored TEST table is unchanged (it ran without the floor).

## 2026-09-28 — detector round 3: a speech/music-aware FlexSED bar and a better veto (Adam's question; reviewer's design; before any number)

Six fixed cells (no grid), `benchmark/detector_round3.py`: C0 shipped; C1 bar 0.6 + PANNs relative veto (class 95th percentile
over the 280); C2 bar 0.6 + PretrainedSED local veto (class 95th percentile within ±1 s); C3 bar 0.5 + PretrainedSED local veto;
C4 bar 0.6 in quiet / 0.8 under speech or music (BEATs Speech or Music ≥ 0.3 in the span) + shipped PANNs veto; C5 bar 0.5 quiet /
0.7 speech-music + PretrainedSED local veto. Pick on the 280 by amendment 24's rule; the one pick tested once on the held-out 415
(ΔC-overlap upper 95 % CI < 0). Leak check: none of the 280 or 415 clips is in the AudioSet-Strong train split (0 of 103,463
train segments), so PretrainedSED has not seen them.
**Result (`benchmark/detector_round3.json`): no pick.** 280 fit set — C0 3.071 / 3.886 (C-overlap / C-onset), recall 50.4 %, 4.46
false/min; C1 4.457, 60.3 %, 9.56; C2 4.207, 60.3 %, 8.81; C3 4.979, 62.1 %, 11.29; C4 3.064 / 3.921, 52.2 %, 4.61; C5 4.186,
60.7 %, 8.79. The PretrainedSED and relative-PANNs vetoes let a lower FlexSED bar find ~10 points more events but double the false
spans; C4 lowers C-overlap by 0.007 but raises C-onset, so it fails the rule. The held-out set stays unread for this round.

## 2026-09-28 — bug fix: the stage-5 family rule had no direction (found in TEST clip mc_bridge_scene)

A visible label silenced every related label at the same moment, both ways: "cars" seen → Vehicle visible → Helicopter silenced
although the Helicopter's own gate check said "not visible". Fix (Adam): a visible label silences only the same label or a more
general one. On the scored renders (simulated, `benchmark/gold/kinship_fix_check.py`): TEST 16 → 18 hits, 25 → 28 wrong, cost
2.63 → 2.60; DEV 14 hits unchanged, 24 → 28 wrong, 2.78 → 2.94. Adopted in `use_shipped()` (`KINSHIP_DIRECTED`) as a logic fix;
the scored tables are unchanged. Disclosed: found by inspecting a TEST clip.

## 2026-09-28 — detector round 4: BEATs end-trim by FlexSED, and a BEATs self-veto in place of PANNs (before any number)

Two fixed questions from the saved caches only (BEATs, FlexSED, PANNs; no model run), `benchmark/detector_round4.py`
(steps `fit` = the 280, `heldout` = the 415) → `benchmark/detector_round4.json`. Cost C, clip sets and bootstrap exactly as
amendment 24 (paired clip bootstrap, 2000 draws, seed 0). Both tests score the stack without the display floor 0.40
(`PICTURE_MIN_CONF` is display-level), as amendments 24–26 and round 3 did. **Sanity gate first:** the script rebuilds the
shipped stack with origin tags (BEATs-origin vs FlexSED-only) and asserts, clip by clip, that its spans equal
`detector_round2.stack()`; the shipped rows must reproduce 3.071 / 3.886 / 50.4 % / 4.46 (280) and 1.923 / 49.1 % / 3.25 (415).
If not, stop.

**Test 1 — end-trim.** BEATs (0.25 s hop) blurs sound ends by ~1 s; FlexSED is frame-level (25 fps). For every final shown span
that came from BEATs (not FlexSED-only; its start may already have been pulled earlier by the twin rule), with family score
per FlexSED frame = max over FlexSED columns of the same canonical family: frames "inside" = frame time t with start ≤ t < end
(frame i covers [i/25, (i+1)/25)); P = peak family score inside; t_last = the last inside frame with score ≥ 0.5 × P; new end =
t_last + 1/25, then clamped: never later than the old end, never earlier than old end − 1.5 s, never earlier than start + 0.5 s
(if start + 0.5 ≥ old end the span is untouched). No same-family FlexSED column, no inside frame, or P ≤ 0 → untouched. No
peak floor; the number of trims that fire with P < 0.3 is reported (descriptive).
*End error:* for each consequential gold event (salient non-speech, non-music) matched by at least one span (same family and
overlap by the scorer's rule, `_overlap_ok`), take the matched span with the largest overlap (tie → earlier start); error =
|span end − gold end|. Trimming can un-match an event, so the **primary Δ = median error shipped − median error trimmed over
the events matched in both arms**; per-arm medians and matched counts reported beside.
*Other numbers:* recall (overlap, onset), false spans/min, C-overlap, C-onset — shipped vs trimmed. Onset recall cannot change
(starts untouched); a trim can only add false spans, so "false spans not higher" is a count: fp(trimmed) ≤ fp(shipped).
*Rule on the 280:* **adopt-candidate iff Δ ≥ 0.3 s AND recall-overlap(trimmed) ≥ recall-overlap(shipped) − 0.5 points AND
fp(trimmed) ≤ fp(shipped)**. Only a candidate goes to the 415; otherwise the 415 is not scored for Test 1.
*Confirmation on the 415 (only if a candidate):* **adopt iff Δ ≥ 0.3 s with its clip-bootstrap 95 % CI excluding 0 (lower
bound > 0) AND the paired ΔC-overlap (trimmed − shipped) upper 95 % CI ≤ 0.** Same numbers as the 280 reported.

**Test 2 — BEATs self-veto in place of PANNs.** Pool = the FlexSED-only spans that survive the FlexSED clip-level veto (exactly
the spans the shipped PANNs veto acts on); pool sizes on the 280 and 415 printed before b. Shipped: kept iff PANNs clip-max for
the span's canonical family ≥ 0.05. Self-veto: kept iff BEATs clip-max (`clip_peak` of the BEATs cache) for the family ≥ b.
A family absent from the model's labels is kept (the shipped convention); BEATs and PANNs share the 527 AudioSet labels, so
coverage is the same. *b:* k = the number of pool spans PANNs keeps on the 280; b = the k-th largest BEATs clip-max over the
280 pool (ties can push the kept share above PANNs'; the actual share is reported beside b, 4 decimals). b is written to the
json by `fit` before any C comparison of the self-veto and before any 415 number.
*Numbers on the 280 and on the 415:* false spans/min, recall (overlap, onset), C-overlap, C-onset, paired ΔC-overlap
(self − shipped) with bootstrap CI, and the agreement between the vetoes on the same pool spans as a 2×2 (both keep / both
drop / PANNs-only keep / BEATs-only keep) and a rate. The 415 is scored for Test 2 whatever the 280 shows (the question is a
replacement, not a pick).
*Rule on the 415:* **drop PANNs iff ΔC-overlap upper 95 % CI < +0.1 AND recall-overlap(self) ≥ recall-overlap(shipped) − 0.5
points**; otherwise PANNs stays and its measured value (the Δ in C, recall and false spans it buys over BEATs alone) is reported.
Neither rule is revised after a number is seen; no TEST, no DEV.
**Result (`benchmark/detector_round4.json`).** Sanity gate passed: the tagged stack equals `detector_round2.stack()` on every clip;
shipped reproduces 3.071 / 3.886 / 50.4 % / 4.46 (280) and 1.923 / 49.1 % / 3.25 (415).
*Test 1, the 280: not a candidate, so the 415 is not scored for it.* 104 of 1,225 BEATs-origin shown spans (all labels, before the salient filter) were trimmed (median
cut 0.57 s, mean 0.70 s; 19 of the 104 with FlexSED in-span peak < 0.3). End error on the 108 events matched in both arms: median
1.440 → 1.145 s, **Δ +0.295 s [+0.073, +0.800]** (below the 0.3 s bar); per arm 113 / 108 matched. Recall-overlap **50.4 → 48.2 %**
(−2.2 points, outside 0.5); onset-recall 25.0 → 25.0 %; false spans 208 → 208 (4.46/min both); C-overlap 3.071 → 3.143 (ΔC +0.071
[0.000, +0.186]); C-onset 3.886 → 3.886. Fails two of three parts (Δ < 0.3 s, recall −2.2 points): the trim makes ends closer
but cuts 5 events below the overlap bar. Not adopted.
*Test 2.* b = **0.1218** (280 pool 167 FlexSED-only spans; PANNs keeps 57 = 0.3413; BEATs ≥ b keeps 0.3413). The 280: C-overlap
3.071 → 3.043 (ΔC −0.029 [−0.136, +0.064]), C-onset 3.886 → 3.857, recall 50.4 → 51.3 %, onset-recall 25.0 → 25.9 %, false/min
4.46 → 4.46; agreement 71.3 % (both keep 33, both drop 86, PANNs-only keep 24, BEATs-only keep 24). **The 415** (pool 174):
C-overlap 1.923 → 1.928, **ΔC +0.005 [−0.048, +0.067]**, C-onset 2.193 → 2.207, **recall 49.1 → 49.7 %**, onset-recall 32.7 →
32.7 %, false/min 3.25 → 3.30; agreement 76.4 % (35 / 98 / 20 / 21; kept share PANNs 0.316, BEATs 0.322). **Rule met (upper CI
+0.067 < +0.1; recall ≥ shipped − 0.5 points): PANNs can be dropped** — the BEATs self-veto at b = 0.1218 does the same job
within the pre-set margin. The two vetoes disagree on about a quarter of the spans but trade equal numbers each way, so the cost
does not move. Not yet applied in `src/` (Adam's call).

## 2026-09-28 — detector round 5 (EAT, Dasheng) (Adam's question; written before any number)

**Question.** Does a newer AudioSet tagger beat BEATs iter3+ AS2M as the main (framewise) tagger in the stage-4 stack? Two
candidates, official weights only: **A = EAT-large** fine-tuned on AS2M (HF `worstchan/EAT-large_epoch20_finetune_AS2M`,
linked from github.com/cwx-worst-one/EAT) and **B = Dasheng-base** AudioSet fine-tuned (Zenodo `dasheng_audioset_mAP497.pt`,
github.com/RicherMans/Dasheng; encoder from the `dasheng` package). `benchmark/detector_round5.py` (steps `cache`, `check`,
`fit` = the 280, `heldout` = the 415) → `benchmark/detector_round5.json`; caches in new folders `eat_cache/`, `dasheng_cache/`
beside `beats/` (nothing overwritten). No TEST, no DEV.

**Baseline = the stack shipped now (commit 84de50b):** BEATs 0.175 / 0.35 + FlexSED bar 0.8 + FlexSED clip veto 0.3 + BEATs
self-veto b = 0.1218 on FlexSED-only spans, PANNs veto off. From round 4's json it scores 3.043 / 3.857 / 51.3 % / 4.46
(C-overlap / C-onset / recall / false per min) on the 280 and 1.928 / 2.207 / 49.7 % / 3.30 on the 415. Every Δ, the pick and
the 415 test are against this stack (not the PANNs-veto numbers 3.071 / 3.886 / 50.4 % / 4.46, which are only the chain check).

**Gates (stop if any fails, before `fit`):**
1. *Sanity.* Round 4's tagged stack (imported, not re-run; round 4's json is not touched) equals `detector_round2.stack()` on
   every clip; the PANNs-veto stack reproduces round 4's numbers and the self-veto baseline reproduces the numbers above
   (round 4's tolerances).
2. *Same windows.* Both candidates score exactly BEATs' windows: the `infer_beats` windowing copied line for line (2-s window,
   0.25-s hop, 1.75-s reflected lead-in, tail cover, stamp = window end − 0.5 s, keep t ≥ 0) on the same audio (ffmpeg mono
   16 kHz → `librosa.load` 16 kHz). Per clip the window count must equal the BEATs cache's and the times match (|Δ| ≤ 1e-4 s).
3. *Labels.* Index → AudioSet mid from EAT's official `inference/labels.csv` (527 rows); it must equal the `class_labels_indices`
   order in `panns_inference`, which is then used for Dasheng (its checkpoint carries no label file); mid → display name via
   `src/audioset_mid_names.json`, the route BEATs uses. The candidate's name set must equal the BEATs cache's. Order check on
   the 280 (scores only, no cost): per name, Spearman ρ across clips between candidate clip-max and BEATs clip-max (matched by
   name); the median ρ with the true order must exceed the median with the order shifted by ±1 index by ≥ 0.2.
4. *Weights.* EAT: the repo's `EAT` class built from `config.json`, `model.safetensors` loaded with `strict=True` (no
   `AutoModel`: the remote code targets transformers 4.51, the env has 5.x). Dasheng: the README's classifier; the encoder
   load must report no missing keys and only `outputlayer.*` unexpected.

**Preprocessing (fixed).** EAT, per 2-s window: subtract the window mean; Kaldi fbank (128 mel, Hanning, 10-ms shift,
htk_compat, no dither) = 198 frames, zero-padded to **208** (the official recipes scale target_length with clip length: 1024
for 10 s, 512 for ESC-50's 5 s, 128 for Speech Commands' 1 s; 208 = the smallest multiple of the 16-frame patch ≥ 198, so no
audio is cut); normalise (x + 4.268) / (2 × 4.569); CLS head logits → sigmoid (once). Dasheng: the raw 2-s waveform into the
README classifier (mean of tokens → LayerNorm → Linear → sigmoid, sigmoid already inside). fp32, no autocast, both models.
Saved as the BEATs caches are (`fw` float16 probabilities, `times` float32, `labels`).

**Cells on the 280 (four).** Each candidate replaces BEATs framewise in the baseline stack; FlexSED unchanged.
- *Primary (EAT-P, Dasheng-P):* same bars (AED 0.175, display 0.35, hysteresis 1.0, min dur 0.5). Self-veto uses the
  candidate's own clip-max for the family, bar b refitted on the 280 by round 4's rule: s0 = the baseline's kept share of its
  280 pool (57 / 167 = 0.3413); k = floor(s0 × |candidate pool| + 0.5); b = the k-th largest candidate clip-max over the
  candidate's 280 pool (ties can raise the share; actual share reported).
- *Secondary (EAT-R, Dasheng-R):* display bar d refitted on the 280 so the number of shown spans equals the baseline's. Shown
  spans = the scored set (salient non-speech, non-music spans after all vetoes, display ≥ d), summed over the 280. d on the
  grid 0.050, 0.055, …, 0.950; AED = d / 2 (hysteresis 1.0); b refitted per d by the rule above (the pool moves with the AED
  bar); d = argmin |N(d) − N_baseline|, tie → larger d. d and b are then frozen.
Onsets: the extractor's onsets in every cell and the baseline; the BEATs occlusion refinement (ONSET_CAM) is not run in any
arm (caches only, as rounds 2–4). It is model-agnostic (forward passes), so it would carry over to either candidate; its effect
is not measured here. No 0.40 display floor (display-level), as rounds 2–4.
Numbers per cell: C-overlap, C-onset, paired ΔC-overlap and ΔC-onset (cell − baseline, clip bootstrap 2000 draws, seed 0),
recall (overlap, onset), false spans and false spans/min, median end error (round 4 test 1's `end_errors`: primary = paired Δ
baseline − cell over events matched in both arms, bootstrap CI; per-arm medians and counts beside), b, d, pool, shown spans.
Clip set = clips with BEATs, FlexSED, PANNs, PE-A-Frame, EAT and Dasheng caches (expected 280 / 415; a missing candidate cache
stops the run).

**Pick on the 280.** Eligible iff C-overlap < baseline AND C-onset < baseline (point values). Pick = lowest C-overlap among
eligible (tie → lower C-onset, then primary before secondary). No eligible cell → the 415 is not scored and BEATs stays.
**Held-out (the 415, pick only, frozen d and b):** passes iff the paired clip bootstrap (2000 draws, seed 0) upper 95 % CI of
ΔC-overlap (pick − baseline) < 0. Same secondary numbers reported, plus the complex / random strata. The 415 has been used by
rounds 2–4; disclosed. No rule is revised after a number is seen.
*Clarification (written before any number):* round 4's json holds b = 0.12176513671875 (a float16 cache value); `config.py`
ships 0.1218, which is above that value, so the shipped bar may drop the one span sitting exactly on round 4's b. Gate 1
reproduces round 4's numbers with round 4's exact b; **the baseline for every Δ is the shipped bar 0.1218**, and both rows are
printed (if they differ, the difference is reported, not chosen). s0 stays 0.3413 (round 4's fitted share).
*Clarification 2 — gate 3 amended (2026-09-28, after the order check fired, before any candidate cost number; only the
baseline's gate-1 row exists).* The Spearman order check as written failed for both candidates: median ρ true order / shift
+1 / shift −1 = 0.558 / 0.454 / 0.452 (EAT) and 0.561 / 0.393 / 0.399 (Dasheng), margins 0.10 and 0.16 < 0.2. The statistic
was mis-specified: clip-max rank across clips is inflated for every column pairing by a shared per-clip activity level (busy
clips raise all 527 scores in both models), so even a shifted order scores ~0.4 and a 0.2 margin was never reachable. The
label-sanity diagnostics that isolate order (none is a cost): per-clip top-1 agreement with BEATs 0.636 true vs 0.018 / 0.007
shifted (EAT) and 0.636 vs 0.018 / 0.004 (Dasheng); top-5 overlap 0.62 / 0.59 vs ~0.13; median Pearson 0.77 / 0.75 vs
0.17–0.22; the true order wins per class for 76 % / 86 % of classes. **Replacement gate 3 (the 280):** top-1 agreement under the
true order ≥ 0.30 AND ≥ 10 × the agreement under each ±1 shifted order; ρ and Pearson reported as descriptive. This threshold
was set after seeing the diagnostics above (disclosed). Cells, bars, the pick rule and the 415 test are unchanged.
*Gate 1 note:* the shipped bar 0.1218 gives 3.036 / 3.850 / 51.3 % / 4.44 on the 280 — one false span fewer than round 4's
b = 0.12176514 row (3.043 / 3.857 / 51.3 % / 4.46, reproduced exactly) — so the baseline is 3.036 / 3.850.
**Result (`benchmark/detector_round5.json`; jobs 31329223 caches, 31329261 check + fit, 31329518 the 415).** Gates 1–4 pass
(gate 3 as amended: top-1 agreement 0.636 true vs ≤ 0.018 shifted, both models). Baseline at the shipped b 0.1218: the 280
3.036 / 3.850 / 51.3 % / 4.44; the 415 1.928 / 2.207 / 49.7 % / 3.30 (on the 415 identical to round 4's b).
*The 280 (ΔC = cell − baseline, clip bootstrap 95 % CI; end error = paired median baseline → cell):*

| cell | display / AED | b | C-overlap (ΔC) | C-onset (ΔC) | recall | onset-recall | false/min | shown | end error |
|---|---|---|---|---|---|---|---|---|---|
| baseline (BEATs) | 0.35 / 0.175 | 0.1218 | 3.036 | 3.850 | 51.3 % | 25.9 % | 4.44 | 564 | — |
| EAT-P | 0.35 / 0.175 | 0.1267 | 3.407 (+0.371 [−0.293, +0.914]) | 4.393 (+0.543 [+0.043, +1.007]) | 65.6 % | 34.8 % | 6.92 | 802 | 1.40 → 1.44 s |
| EAT-R | 0.515 / 0.2575 | 0.1659 | **2.679 (−0.357 [−0.964, +0.121])** | 3.579 (−0.271 [−0.772, +0.143]) | 59.4 % | 31.2 % | 4.14 | 565 | 1.44 → 1.39 s |
| Dasheng-P | 0.35 / 0.175 | 0.1667 | 2.971 (−0.064 [−0.557, +0.350]) | 3.871 (+0.021) | 57.6 % | 29.5 % | 4.84 | 649 | 1.44 → 1.44 s |
| Dasheng-R | 0.39 / 0.195 | 0.1775 | 2.707 (−0.329 [−0.821, +0.064]) | 3.621 (−0.229 [−0.657, +0.157]) | 56.7 % | 28.1 % | 3.96 | 569 | 1.44 → 1.50 s |

Eligible (both C below baseline): EAT-R, Dasheng-R. **Pick = EAT-R** (d = 0.515, AED 0.2575, b 0.1659), frozen.
*The 415, EAT-R vs baseline:* C-overlap 1.928 → 2.014, **ΔC-overlap +0.087 [−0.159, +0.357] → fails** (upper CI not < 0);
C-onset 2.207 → 2.294 (ΔC +0.087 [−0.212, +0.386]); recall 49.7 → 54.4 %; onset-recall 32.7 → 37.4 %; false/min 3.30 → 3.79
(228 → 262 false spans); shown spans 672 → 763; end error (76 paired events) 1.595 → 1.670 s; strata ΔC complex +0.063
[−0.262, +0.381] (n 252), random +0.123 [−0.270, +0.491] (n 163). **Not adopted: BEATs stays the stage-4 tagger.** Both newer
taggers find more of the sounds (recall +5 to +14 points at the same bars), but at BEATs' bar they also raise more false spans,
and with a bar refitted to BEATs' span count on the 280 the gain did not carry over to the 415. So
on the 280 the gain appeared only with the refitted bar (at the shipped bars neither tagger was better).

## 2026-09-28 — detector round 6 (DASM in place of or next to FlexSED): pre-registered in `docs/prereg_round6_dasm.md` (written before any DASM score or cost)

## 2026-09-28 — detector round 7 (SAM-Audio: remove speech and music, re-run BEATs + FlexSED on the residual, union with the shipped stack): pre-registered in `docs/prereg_round7_samaudio.md` (written before any SAM-Audio output or cost)

## 2026-09-28 — detector round 8 (new ideas: repetition bar, per-clip normalisation, parent emission, VLM scene prior, local-contrast veto, onset fixes, multi-scale and band-limited BEATs/FlexSED): pre-registered in `docs/prereg_round8_ideas.md` (written before any round-8 cache or cost)

## 2026-09-28 — fresh AudioSet-Strong confirmation set (500 eval ids drawn with the 415 recipe, seed 20260928, disjoint from the 280, the 415, DEV, TEST, slice B and every repo list; final check only for a candidate that passes the 415, no result read before that): pre-registered in `docs/prereg_fresh_confirm_set.md` (written before any download)
