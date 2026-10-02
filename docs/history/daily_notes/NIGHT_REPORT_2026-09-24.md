# Night of 2026-09-24 — log of what was tried

Adam asked, going to sleep: keep working under reviewer supervision on (1) pictures that appear too
early, (2) pictures that are not always nice — try scene context, and try asking a model what was
drawn and whether it fits the sound. Keep a backlog of changes and results. This file is that
backlog, written as things happen. The plain-language summary for the morning is at the top once the
night is over.

---

## Running log

**Timing (continued from the evening).** Four arms on the 49 DEV clips, each with its own switch:
the fix alone (`dev_mono_v30`), the eight-second cap off alone (`dev_cap_v30`), both
(`dev_monocap_v30`), both plus the stricter merge start (`dev_monostrong_v30`).

**Judge.** The describe and judge steps finished (147 rows: 49 clips x 3 systems). The last,
grounded step failed because a job script was overwritten on the cluster while a running job was
still reading it — my mistake, not the judge's. Rerun as job 30993473. Rule for the rest of the
night: never touch a script a running job is executing; add new files instead.

**Pictures — what is actually wrong with them.** All 33 pictures of the final DEV render, looked at
one by one:

| verdict | n | examples |
|---|---|---|
| good | ~18 | bells, fire alarm, explosion, rooster, pigeon, ambulance, trains, clapping crowd |
| blank white square | 2 | both Thunder, subject "Sky rumbles" — a sky on a white background is nothing |
| wrong or misleading | 5 | "Siren spinning" drew the *mythological* Siren; "Fingers strike keyboard" drew a finger with no keyboard; an electric shaver drawn as a man with shaving foam and no razor |
| faint | 4 | pale cups on white, a white cupboard on white, faint glass shards, a thin sparkler |
| wrong kind for the scene | 3 | a subway got steam-locomotive wheels; a modern level crossing got a steam train; a crime scene got a pink bedside alarm clock |

**Why the scene never reaches the picture** (read from `src/stage5_cross_modal_analysis/reason.py`):

1. The sentence that decides *what to draw* is written by the model **without seeing the frames** —
   it gets the sound's label and a one-line description of the place, nothing else.
2. The step that asks the frames *what kind* of thing is making the sound only runs when the sound
   detector gave no sub-label. The detector almost always gives one, and it is usually generic —
   "Rail transport", "Thunderstorm" — so the frames are never asked. In the final DEV run it did not
   fire once.
3. The prompt says "no adjectives", and a later rule deletes any word that also names the place.
   Both were added for good reasons (the model wrote "Palace window cracks"; Adam asked for no
   assumptions about the place), and between them "subway train" becomes "train", which the
   generator draws as a steam locomotive.

So the scene is not *ignored* by accident; three deliberate rules squeeze it out. The distinction a
fix must keep is: the **setting** (a palace, a forest) stays out, but the **kind of the thing making
the sound** (a subway train, a burglar alarm) is exactly what a viewer needs.

**Pictures — the plan the reviewers settled** (two rounds, three reviewers;
`docs/history/preregistrations/picture_quality_prereg.md`, committed before any picture was drawn):

  * **Scene context, done the safe way.** The frames are now *always* asked which kind of thing is
    making the sound, and asked about the **setting and era** — because the gate already decided the
    source is off screen, so what the frames show is the place, and the place is what separates a
    steam locomotive from a subway train. The background stays white. All three reviewers rejected
    feeding the frames to the picture model itself: the frames contain everything *except* the thing
    to draw, and the picture model would paint the scene the viewer already has.
  * **A new instruction for what to draw**: the object that makes the sound first; never the sky or a
    body part on its own; a sound name that means two things gets the word that fixes it. No example
    sentences in the prompt (Adam's earlier rule — examples leak).
  * **A blank-picture guard**: too little ink → redraw twice → otherwise drop, counted as a failure.
  * **Qwen-Image-2512** (already on the cluster) as a separate arm, same subjects, same seeds.
  * **Adam's "ask a model what was drawn"**: kept as a *measurement*, not a redraw loop. All three
    reviewers warned that "redraw until the checker says yes" keeps exactly the pictures the checker
    is lenient on. So two independent markers (Idefics3 and CLIP-L — neither used anywhere in the
    pipeline, and not sharing a vision tower) mark every picture, and each must first agree with my
    hand verdicts on at least 26 of 32 pictures or it does not get a vote.

Jobs: 30993605 (check the markers), 30993606 (arms A0, A1), 30993607 (new subjects, arms A2, A3).

**Result 1 — neither automatic marker can be trusted as designed** (job 30993605). Checked against my
hand verdicts on 32 pictures, Idefics3 agreed on 25 and CLIP-L on 25; the bar was 26. The failure is
systematic: **both called all five pictures I judged wrong correct** — the mythological Siren, the
finger with no keyboard, the shaving man with no razor, the car dashboard, the cropped crowd. Reason:
the question was multiple choice (the right sound plus five decoys), and a wrong picture still wins if
the right sound is the least-bad option — the winged woman is closer to "Siren" than to "Quack". This
is the pre-registered outcome "no marker votes"; per the plan, the arms are judged by eye unless a
redesigned marker passes the same check. The redesign (an open question — "what is making a sound in
this picture?" — whose answer is then matched as text) went to the reviewers before being run.

**Result 2 — what the new subject rules did, by eye** (arms drawn by job 30993606/30993607; A1 = old
subjects with a new seed and the guard, A2 = new subjects):

  * **Clearly fixed by the new rules (5):** the electric shaver is now a shaver, not a man with foam;
    the "Siren" is now a red siren light, not the winged woman; the subway gets a modern subway train,
    not steam wheels; the white cupboard on white is now a wooden cupboard; the keyboard now has a
    keyboard.
  * **Blank thunder → clouds.** "Sky rumbles" drew a white square three times in a row even with the
    guard (so the guard dropped it); the new subject "Thunderclouds rumble loudly" draws a storm cloud
    every time. Better than blank, but a cloud without lightning still does not say *thunder*.
  * **Made worse (2):** the place leaked back into the picture — "Crowd cheering in palace" drew a
    crowd in front of a palace, "Firecracker exploding on street" drew a street scene. The prompt
    forbids scenery; the model added it anyway and the old strip rule kept it because removing the
    place would have left a dangling "in"/"on".
  * **Unchanged (3):** the steam train at a modern crossing (the frames were asked for the kind but
    said "unknown"); the alarm clock at the crime scene (the frames show a bedroom, so this may even
    be right); faint glass shards.
  * **A surprise worth knowing:** several "bad pictures" were just an unlucky draw. With nothing
    changed but the random seed, the cropped crowd became a full crowd and the lone finger became
    hands on a keyboard. One sample per picture is noisy.
  * **The scene step barely spoke:** asked about the setting for all 33 sounds, it named a kind only
    once (the subway) and said "unknown" 32 times.

**Result 3 — the picture arms, marked by eye with the arm hidden** (99 pictures: arms A1, A2, A3 × 33
sounds, shuffled into three sheets under codes, labelled, and only then unsealed;
`benchmark/gold/pictures/blind_labels_arms.json`). The question for each: *would a viewer glancing
at it read the right sound?*

| arm | what changed | readable (of 32) | vs the arm before |
|---|---|---|---|
| shipped | today's pictures | 24 | — |
| A1 | new random seed + blank guard | 23 | −1 (noise: fixed 2, broke 3) |
| A2 | + new "what to draw" rules | 25 | +2 (fixed 4, broke 2) |
| **A3** | **+ Qwen-Image-2512 instead of FLUX** | **30** | **+5** (fixed 6, broke 1) |

Against today's pictures, A3 fixes 7 (both blank thunders, the electric shaver, the cropped crowd, the
mythological Siren, the sparkler, the lone finger) and breaks 1 (a train drawn as a bare wheelset).

Against the gates written before the run — judged by eye, because neither automatic marker passed its
own check (the pre-registered fallback): **A3 passes** (0 blank; net +5 ≥ +4; one good picture got
worse, and it was looked at; no picture tells a viewer something false). **A2 alone does not** (+2).

Honest limits, all of which matter:
  * the labeller is me, the same person who wrote the reference, and the arm was hidden but I had seen
    the subject lists, and Qwen-Image's style is recognisable — this is *not* an independent verdict;
  * the new rules were written from these same 33 pictures, so DEV flatters them;
  * **scenery crept back in** with the new rules: two crowds in front of a palace, a firecracker in a
    street, two thunders drawn as a full dark sky rather than an object on white. None of these says
    something false, but it is exactly the drift the white background was chosen to stop, and the
    place-strip rule needs fixing before any of this is adopted;
  * Qwen-Image costs ~20 s per picture on an H200 against under 1 s for FLUX, and needs an 80 GB card.

What would make it real: Adam (or anyone who is not me) labels a sample of A3 against today's
pictures, blind, on clips the rules were not written from.

**Result 4 — the first timing runs were not a fair test (my mistake), and are being redone.**
The four timing arms (`dev_*_v30`) finished with the fix doing what it should — the Siren and the
rooster landed on time, 5 more needed sounds got a picture, none were lost, and the picture's end
moved from 1.70 s early to 0.55 s early — **but** false alarms nearly doubled (0.57 → 1.04 per clip).
Tracing the extra pictures (insects in a pet shop, cooking, coins, a cash register, a microwave)
showed they never reached the pipeline in the base run at all. The reason: the base run was started
with the two adopted false-alarm filters switched on from the command line (`VETO=0.3 PVETO=0.05`),
and I started the arms without them. So every one of those arms differs from the base by the
filters, not only by the timing fix, and none of its numbers can be used.

Redone as `dev_*_v31` with the filters on (jobs 30993797-800), plus a fourth run with **no** new
switch at all (`dev_repro_v31`), which must reproduce the base: that is the check that the
instrumentation and the switched-off code change nothing.

What survives from the confounded runs: the named onset fixes (Siren −1.50 → 0.00, Bird −1.29 →
−0.15) are pictures that were drawn in both runs, so they are the timing fix and not the filters.
The +5 hits and the false-alarm rise are not attributable until the v31 runs land.

**Result 5 — the one allowed marker redesign also failed.** Asked openly "what is drawn, and what
sound does it make", Idefics3 agreed with my verdicts on only 8 of 32. It mostly ignored the two-line
answer format and answered "nothing" for the sound. As agreed with the reviewers there is no second
redesign tonight, so the picture arms stand judged by eye (Result 3).

One thing worth keeping for tomorrow: asked what *object* is drawn, it names exactly the failures I
found by eye — "man with face mask" for the shaver picture with no shaver, "hand, two fingers" for
the keyboard picture with no keyboard, "hat" for the mythological Siren, "nothing" for the blank
thunder. An *object-only* check may be a usable marker, but it has to be calibrated on fresh pictures
labelled by someone else before it counts.

**Result 6 — the judge, checked against Adam's labels before its verdict is used**
(`benchmark/gold/judge_trust.py`, gates from `docs/history/plans/judge_plan.md`, DEV 49 clips, rubric-capped
grounded judge):

  * **Check 1 — does the judge's score fall where Adam's labels say a clip went badly? PASS.**
    Spearman −0.647, 95% CI [−0.838, −0.413]. The judge tracks the annotator well overall.
  * **Check 2 — does it separate clips with a known-wrong picture from clean ones? Just FAILS.**
    Clean 3.07 vs wrong-picture 2.45: the right direction, gap +0.62, but the CI [−0.08, +1.27]
    touches zero at 49 clips.
  * **So the ranking is not reportable yet**, by the rule fixed before the run — even though it puts
    the system first (ours 2.80, draw-everything 2.71, captions 2.49). Check 2 is a sample-size miss,
    not a wrong-direction one; the same test on all 139 gold clips would very likely settle it.

**Result 7 — a stronger judge passes both trust checks, and prefers the caption baseline.** The same
checks on all 139 gold clips of the earlier `v4b4` run (from before the false-alarm filters), where
both judges exist:

| judge | tracks Adam's labels | separates wrong-picture clips | its ranking |
|---|---|---|---|
| Mistral-7B (current) | ρ −0.53 [−0.67, −0.38] ✓ | gap +0.41 [−0.03, +0.83] ✗ | blind 2.70 ≈ ours 2.68 > caption 2.51 |
| **Gemma-4-31B** | ρ −0.61 [−0.73, −0.48] ✓ | **gap +0.86 [+0.37, +1.32] ✓** | **caption 2.84 > blind 2.48 > ours 2.43** |

So the judge we *can* trust ranks the text-caption baseline first and our pictures last — on the
older system. Reported against interest. Two structural reasons it may be unfair to pictures, both to
be checked before anyone draws a conclusion: captions are judged on their own text, while a picture is
first described by another model (information is lost before the judge sees it); and v4b4 predates the
filters that removed a third of the false alarms. The fair test — Gemma on the current system — is
running (job below).

**Result 8 — THE TIMING FIX WORKS, and it is significant** (fair runs, filters on, DEV 49 clips).

First, the check that makes the rest trustworthy: a run with every new switch off (`dev_repro_v31`)
reproduces the base **exactly** — 0 of 23 starts differ, every metric identical. So the logging and
the switched-off code change nothing.

| run | hits | P | R | F1 | wrong pictures / clip | viewer cost | early starts | picture ends early by |
|---|---|---|---|---|---|---|---|---|
| base | 11 | 0.282 | 0.306 | 0.293 | 0.57 | 3.18 | 5 | 1.70 s |
| **onset rule (`MONO`)** | **15** | **0.375** | **0.417** | **0.395** | **0.51** | **2.73** | **2** | 1.05 s |
| cap off only | 10 | 0.263 | 0.278 | 0.270 | 0.57 | 3.27 | 5 | 0.21 s |
| onset rule + cap off | 14 | 0.368 | 0.389 | 0.378 | 0.49 | 2.78 | 2 | 0.06 s |

**The onset rule alone, paired against the base (2000 bootstrap draws over clips):**
F1 **+0.101** [+0.022, +0.196] · precision **+0.093** [+0.018, +0.192] · recall **+0.111**
[+0.024, +0.225] · viewer cost saved **+0.45** [+0.08, +0.94]. All four intervals exclude zero.
**4 needed sounds recovered, 0 lost** (the Siren and the rooster as the trace predicted, plus a glass
and a laughter). This is the first time equal-weight F1 has moved significantly in this project —
because a picture that arrived early was being counted as a *wrong* picture, and is now a hit.

The log confirms the rule is really implemented: under it the refinement step moved 197 starts, **0
earlier** (median +0.14 s); without it, the same step moved 223 of 442 starts earlier, the worst by
8.98 s.

Against the go/no-go written before the run: zero regressions ✓, headline up and outside the CI ✓,
at least 2 of the 5 early cases recovered (2 — Siren, Bird — plus 2 other sounds) ✓, no start earlier
than its anchor ✓. **Passes.**

**The eight-second cap: fixes the end, not adoptable yet.** Removing it makes pictures stay until the
sound actually ends (1.70 s early → 0.06 s), which is Adam's second complaint — but it loses one
needed sound (the bell in `bell_miami` disappears entirely rather than being cut short), so it fails
"zero regressions", and on top of the onset rule it is not significant. Worth one targeted look at why
the long bell vanishes before trying again.

**Caveats that belong beside the numbers.** DEV is where the five early cases were found, so this is
not an out-of-sample result, even though the rule has no tuned parameter. The confirmation is the 60
TEST clips — which would be a **second look at TEST** (Adam and the supervisor decide). The remaining
early starts come from the step that joins the two detectors, which moves 77 starts earlier by a median
4 s; that is the next lead, not tonight's.

*Why the bell vanishes without the cap.* With the cap, the bell's 8 s span is cut into two stretches;
the visibility model says "church bell on screen" for the first and — on a 2-to-1 vote — "not on
screen" for the second, so the picture is shown and counts as a hit. Without the cap, the 14 s span is
cut into three stretches and the model says "on screen" for all three, so the picture is silenced.
Adam's label says the bell is not visible. So the "regression" is the visibility model disagreeing
with Adam *consistently* instead of *inconsistently*: the base's hit was a lucky vote. By the rule
fixed in advance it still counts as a loss, so the cap stays for now; the decision on the cap should be
taken knowing this.

**Result 9 — the trustworthy judge on the *current* system: a tie with captions, not a loss.**
Gemma-4-31B, the judge that passed both trust checks on 139 clips, run on the current system (DEV 49):

| | old system (v4b4, 139 clips) | current system (DEV 49 clips) |
|---|---|---|
| ours − captions | **−0.41** [−0.66, −0.16] — captions significantly better | **+0.08** [−0.24, +0.39] — a tie |
| ours − draw-everything | −0.05 [−0.31, +0.21] | +0.24 [−0.08, +0.57] |
| ranking | captions > blind > ours | ours 2.80 > captions 2.71 > blind 2.55 |

On these 49 clips Gemma's second trust check misses (gap +0.54, CI [−0.24, +1.25]) for the same
sample-size reason as Mistral's, but it passed both checks on 139 clips, so it is the judge worth
listening to. The honest reading: the old system was significantly worse than simply writing the
sound's name; the current one is level with it. The two columns are different clip sets, so this is
not a paired comparison. To settle it the current system would need judging on more clips — **not**
the 60 TEST clips, whose judge results must not be looked at without a decision to take a second look.

**Result 10 — the reviewers signed the onset rule off, and their doubts were checked.**
All three: **pass** against the gate written before the run. Their doubts, and the answers
(`benchmark/gold/onset_attribution.py`, base = the switches-off run):

  * *"An onset rule cannot lower false alarms; the precision gain may be the visibility vote drifting."*
    Checked: of the 4 recovered sounds, **3 are the picture's start moving into the window** (Siren
    1.1 → 2.6 s, rooster 9.1 → 10.25 s, laughter 11.9 → 13.4 s); only 1 (a glass) is a new picture.
    The false-alarm drop is **exactly those 3 pictures**: an early picture counted both as a wrong
    picture *and* as a missed sound, and moving it into the window turns both into one hit. No drift.
  * *No clip-start sound may move*: none did. *No picture may become more than 1 s late*: the largest
    later move is +1.50 s (the laughter), which lands 0.66 s after the sound — inside the window.
  * *"It rests on 4 sounds, 4 to 0: fragile. TEST decides."* Agreed; an exact sign test on 4-0 is
    p = 0.125. Reviewer C's own stricter primary (net +5 on the pre-gate detector set) is **not** met:
    that set has only 16 sounds with an early-able onset, and the rule recovers 2 and loses 0.
  * **The eight-second cap stays, 2 votes to 1.** Removing it is a separate change that touches the
    visibility vote (the bell); test it on its own, with a release rule. The dissenting reviewer's
    condition for removing it (no more than 3 of 17 pictures lingering 2 s past the sound) *is* met —
    1 of 16 — so this is a close call for Adam, not a clear no.

**What all three say the morning summary must say about TEST.** The rule was chosen on DEV, so the
DEV gain is optimistic by construction and is not the thesis number. The configuration is frozen now
(the base plus `ONSET_MONOTONE`, cap at 8 s, filters 0.3 / 0.05) and run **once** on TEST with no
further change; that is the **second look at TEST** and is recorded as such. Keep if TEST loses at most
one needed sound and F1 does not fall; if TEST does not confirm, the rule stays as a bug fix proven by
the trace (223 of 442 starts moved earlier before, 0 after), with the gain reported as DEV-only.

**Result 11 — limiting the detector union's early pull: rejected, as the rule fixed in advance says.**
Two variants on top of the onset rule (`dev_monobound_v31`: a twin may pull a start at most ~1 s
earlier; `dev_monobeats_v31`: BEATs keeps its own start). Decision rule, written before the run:
adopt only if no needed sound is lost and early starts do not rise.

| run | hits | F1 | cost | lost vs onset rule |
|---|---|---|---|---|
| onset rule (`dev_mono_v31`) | 15 | 0.395 | 2.73 | — |
| + bounded union | 14 | 0.364 | 2.90 | the approaching **helicopter** |
| + BEATs start | 13 | 0.338 | 3.02 | the helicopter and a train |

Both lose the approaching helicopter — the exact case the early pull exists for (the second detector
hears it fade in before BEATs does). So the union's median −4 s pull is mostly *right*, not a bug; the
remaining early Crowd (−1.90 s) is not worth that price. The union stays as shipped.

**Result 12 — the place fix works; the "full-frame" guard backfires on thunder.**
The place-phrase strip changed 5 subjects ("Crowd cheering in palace" → "Crowd cheering", "Firecracker
exploding on street" → "Firecracker exploding", "Car driving on hillside" → "A car drives", "People
laughing on street" → "People laughing"). By eye, all four affected pictures are now the object on a
plain background: no palace, no street, the car on a road instead of a speck on a hill.

The upper guard (redraw a picture that fills more than 85% of the frame), added after seeing A3 at a
reviewer's suggestion, fired on both thunders, redrew them twice, and **dropped** them — throwing away
what is, by eye, the best thunder picture of the night (storm clouds *with lightning*). It has no
upside on these 33 and costs two readable pictures, so it is **not recommended**. The recommended
picture setup from the night is therefore **A3c: the new subject rules + the place strip + Qwen-Image,
with only the blank guard**. Its pictures are A3b's, except the four where A3b's upper guard fired;
for three of those the subject is unchanged, so A3's seed-0 picture is exactly what A3c draws, and for
the fourth (the firecracker, whose subject changed) A3b's redraw is used — a small, recorded deviation.

---

## Morning — Adam's blind ratings (`benchmark/gold/pictures/adam_ratings_2026-09-24.json`)

189 answers, unsealed against the key after he sent them.

| version | yes | no | not sure |
|---|---|---|---|
| today's pictures (shipped) | 12 / 31 | 17 | 2 |
| **new: rules + place strip + Qwen-Image (A3c)** | **18 / 32** | 13 | 1 |
| FLUX, two more seeds | 8 / 33, 9 / 33 | | |
| Qwen-Image, two more seeds | 12 / 30, 14 / 30 | | |

Paired on 31 sounds: 11 got better (no/unsure → yes), 5 got worse. The gain holds on every seed
(FLUX ~26% yes, Qwen-Image ~43-56%), so it is not luck. Adam is far stricter than I was (43% yes on
today's pictures against my 71%; Cohen's kappa 0.46 on 28 shared yes/no). He used "not sure" for
"it depends on the scene".

A flaw of mine in the page: it showed the sound's **family** name ("Vehicle", "Siren") under each
picture — the very thing Adam then flagged. His main finding: pictures are drawn from the family tag,
not the specific sound (a bus drawn as a car, a car horn drawn as a train, thunder drawn as a sky);
7 of 33 pictures had a more specific detector label available and did not use it. And the scene must
be used to pick the right source (a car door in a car scene, not a house door).
