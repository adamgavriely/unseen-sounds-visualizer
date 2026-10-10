# Pre-registration: BEATs-strong (PretrainedSED) as an extra detector, and as a wrong-dropper (11 Oct 2026)

Written before any BEATs-strong score on the 158 benchmark clips was read.

Base: v1.7 (`python benchmark/gold/v14_score.py`): 59 hits of 124 needed sounds, 13 wrong (9 wrong family, 1 visible,
3 phantom), onset cost over 158 clips 1.810. Noise floor of a paired difference: about 0.19 (noise_floor.md).

Models: the five PretrainedSED AudioSet-Strong checkpoints (BEATs_strong_1 main; ATST-F, fpasst, M2D, ASIT as a
5-model mean), frame probabilities (40 ms, no median filter) on the same 16 kHz clip audio the pipeline uses.
A class belongs to a sound's family when score_per_sound.same_family(class name, label).

## A. Wrong-dropper on v1.7 pictures

Feature per picture: family max probability over the picture, and over [start - 0.25, start + 0.75] s; BEATs-strong
and 5-model mean (4 variants). Drop when < t; t from the feature's own quantiles; choice (incl. "off") by clip-grouped
5-fold CV (random.Random(0) over dev + test stems, folds sh[k::5]) on onset cost.
Pass: out-of-fold 0 hits lost and >= 2 wrongs removed. Expected reach: the 3 phantoms and a few wrong-family pictures;
13 wrongs are 0.165 cost in total, so no veto can reach the noise floor alone.

## B. Extra detector (proposer ceiling)

For t in {0.2, 0.3, 0.4, 0.5, 0.6}: BEATs-strong runs (class prob >= t, runs merged across gaps < 0.5 s, length >= 0.3 s)
of depictable families (not speech / music, not in the v1.4 texture ban or the v1.5 unverifiable list).
- recoverable: needed sounds v1.7 misses with a run of their family starting -0.5..+1.0 s of the onset;
- raw extra wrongs: runs starting where no gold sound of that family has its onset in the window, and no v1.7 picture of
  that family is up; split into "a gold sound of that family is playing on screen" (the gate can catch it) and the rest.
Pass (an upper bound, before the veto chain and the gate): recoverable >= 8 AND
4 x recoverable - 2 x (raw extra wrongs not on screen) > 30 at some t (i.e. clears the floor with no help from vetoes).
Only if B passes: BEATs-strong becomes a stage-4 origin only where no candidate of that family exists, the new
candidates go through the VLM gate (one GPU job), and the result is scored with the same CV.
