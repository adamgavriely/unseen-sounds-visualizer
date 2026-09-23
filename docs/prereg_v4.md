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
stage-2 backend (owl.py, sam3.py, siglip.py) and is read by nothing outside stage 2 -- a grep of the
whole source finds no consumer. The gate's verdict is the VLM's alone, and an open-vocabulary object
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
