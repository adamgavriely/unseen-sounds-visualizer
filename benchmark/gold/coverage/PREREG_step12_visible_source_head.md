# Pre-registration: Step 12, a trained per-second "source visibly making this sound" head (DEV only)

Written 2026-10-08 before any download, feature or model. Downloads wait for Adam's own OK. TEST untouched; adopt nothing.
Current leading DEV candidate: a/b gate + flash rule, 32 hits / 14 wrong (missed its own bar by one wrong; not adopted).

## Features per second t of a sound (family F, clip cut around t)

1. DenseAV (MIT): mean of the sound-head activation map and of the language-head map for the text "F", inside the frame;
   their agreement (cosine of the two maps); the maps' peak.
2. OWLv2 box of "F" (best box over the second's frames, if score >= 0.1): DenseAV activation inside the box / outside.
3. Synchformer synchronizability (offset-class probability of 0 s) on the 2-s cut.
4. CAV-MAE audio-image cosine (audio of the second vs the middle frame).
5. RAFT flow energy inside vs outside the OWLv2 box, at t vs t - 1 s.
6. The Step 11 flash score (max luminance jump in the second).

## Training data (only what Adam approves; benchmark YouTube ids excluded by the audioset_fresh.py name scan)

AVATAR (HF mipal/AVATAR, CC-BY-4.0): its on-screen / off-screen annotations give per-second positives (source visible
and sounding) and negatives (sounding, source off-screen). If approved later: VGGSounder audible-not-visible, LLP
audio-only segments. Model: L2 logistic regression and a 1-hidden-layer MLP (32 units), chosen by 5-fold CV on the
training data (never DEV).

## DEV checks

A. Ranking: per-sound score = mean over its seconds. Units = DEV gate records matched to gold (as Step 3 / 3b):
positive = matched gold sound ticked visible, negative = all matched sounds needed. Pass: AUROC >= 0.85 (current gate on
the same units: 0.78 on the 59 Step-3 units, 0.82 on the 109 Step-3b units).
B. Pipeline: on top of a/b + flash, (i) per-second cut: seconds with score >= s are removed from a picture (s chosen
by 5-fold CV on the training data at 90% precision for "visible"); (ii) gate-loss recovery: a sound the gate silenced is
drawn if its mean score < s. Pass: wrong <= 12 at hits >= 32.
