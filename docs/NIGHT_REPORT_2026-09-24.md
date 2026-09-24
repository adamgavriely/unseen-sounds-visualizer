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
