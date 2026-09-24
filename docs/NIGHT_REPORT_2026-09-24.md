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
`docs/picture_quality_prereg.md`, committed before any picture was drawn):

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
(`benchmark/gold/judge_trust.py`, gates from `docs/judge_plan.md`, DEV 49 clips, rubric-capped
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
