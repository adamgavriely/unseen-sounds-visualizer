# Pre-registration: a "with sound / without sound" evaluation reference

*Committed 2026-09-15 evening, before any code for it exists and before any run. Adam's
idea, developed with a three-Fable panel. The test-set headline (gated 3.17 vs blind 3.16,
human-grounded reference) does not change whatever happens here.*

## 1. What is proposed

A new way to write the **reference sentence** — the sentence the judge scores every
system against ("what a hearing viewer gets from the soundtrack that a deaf viewer would
miss", or "nothing beyond the picture"):

1. One audio-visual model looks at the clip **twice, on the same frames**: once with the
   soundtrack, once with the sound removed. Both times it fills the same fixed list of
   "things happening", each marked *seen / heard / both*.
2. The reference = the *heard* items from the with-sound run, minus any item the muted run
   already listed (same event in other words counts as the same; a text model matches).
   Speech and music items are dropped (the system never shows them).
3. An empty list becomes "nothing beyond the picture", written by a fixed template.

Why it should be better than what we have: it is not written from our detector's output
(the model-derived reference was, and rewarded whichever system repeated the detector);
it can hear a sound our detector missed (the current judge cannot see detector misses);
and the muted run is a built-in **placebo**: if "heard" items appear just as often with
the sound off, the model is guessing from the picture.

Model: **not** a Qwen model — our system's eyes are Qwen2.5-VL-7B, and a reference that
shares them would miss the same objects and agree with the gate for the wrong reason.
First choice MiniCPM-o 2.6 (open weights, video + audio, different vision encoder);
fallback Qwen2.5-Omni-7B, reported with that caveat. The reference never sees BEATs
labels, gate decisions, or any system's output; prompts never mention "gate" or
"off-screen".

## 2. What is measured, on the development split only

The **156 development clips with a video file** (103 seen, 20 unseen, 33 no-ambient; test
clips are not touched). The one decision that matters is the reference's answer to *"is
anything missing?"* (list non-empty) against Adam's label (picture due = unseen / mixed).

Because only 20 of 156 clips have a picture due, raw agreement is misleading (always
answering "nothing missing" scores 87%). So the measure is **balanced accuracy**: the
average of (a) how often it says "missing" on the 20 picture-due clips and (b) how often it
says "nothing missing" on the 136 others. Chance = 50%. Cohen's κ is reported alongside
(agreement above chance, 0 = chance, 1 = perfect), not used as a bar.

Also reported: on picture-due clips, whether the list names the labelled sound; the
placebo rate (heard items that survive with the sound off); and, if the bar below passes,
how the new reference scores the systems on dev compared with the human-grounded one.

## 3. The bar, relative to what we had

| reference | silence decision vs labels | note |
|---|---|---|
| model-derived (proposal) | balanced 57.7%, κ = 0.05 (dev, computed 2026-09-15) | says "missing" whenever the detector hears anything |
| independent (four models) | 55% raw agreement (test, 2026-09-14) | at chance |
| human-grounded | 100% by construction | uses the labels themselves |

**Adopted for one run on the test set (as a secondary result) iff** on the dev split it
(i) beats the model-derived reference's balanced accuracy by at least **10 points**
(≥ 67.7%) **and** (ii) the placebo rate is under 20%. Between 57.7% and 67.7%: reported as
"promising, not adopted". Below: reported as a negative pilot, like the six detector
attempts. Whatever the outcome, the numbers go in the report and in LIMITATIONS.md.

With 20 picture-due clips, (a) carries about ±20 points of uncertainty; the result is an
indication, not proof, and is stated as such.

## 4. What will not be done

No prompt or threshold change after seeing the dev numbers; no second run on test; no
change to the headline; no tuning against the human-grounded scores.
