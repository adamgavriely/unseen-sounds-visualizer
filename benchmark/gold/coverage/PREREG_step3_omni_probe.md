# Pre-registration: Step 3 prep, two Qwen3-Omni probes on DEV (measure only, adopt nothing)

Written and committed 2026-10-06 before any answer was generated. Model: Qwen/Qwen3-Omni-30B-A3B-Instruct (Apache-2.0),
thinker only, bf16, greedy, one H200. DEV only (merged DEV, 71 clips). All answers cached in `omni_probe_dev.json`.

## (a) "Is the source visibly making this sound?" (video + audio)

Items: every gate record of the frozen system on DEV (its `_trail` arm, 72 sounds, 448 s), one question per second t
(t = start, start + 1, ... < end). Input: the clip cut [t - 1, t + 2] s (video with its audio, ffmpeg cut, 2 frames/s
by qwen-omni-utils, use_audio_in_video). Question:
"You are watching a short video with its sound. A sound of {label} is heard in it. Is the thing making this sound
visible on screen, and can you see it making the sound? Answer yes or no."
Score per second = logit(yes) - logit(no) (max over yes / no token ids, as dev_listener.py). Score per sound = mean
over its seconds (also reported: max).

Truth per gate record: the gold sounds of the same family (score_per_sound.same_family) that overlap the record.
Positive (source visible) if any of them is ticked visible; negative if all are needed (not visible, not obvious);
otherwise (obvious only, or no gold sound) left out and counted.
Reported: AUROC of the Omni score; of the current gate (share of the record's stretches judged seen, majority rule);
of the a/b rule (share seen, split -> majority); n positive / negative.

## (b) "Which of these heard families, or none?" (audio)

Items: every DEV candidate in the decision trail (`docs/decision_trail/data.js`, kept and dropped), Speech / Music
families left out, de-duplicated on (clip, canonical family, start and end rounded to 0.25 s); plus every picture the
frozen system shows on DEV. Input: the clip's audio cut [start - 2, end + 2] s (16 kHz mono). Options: the canonical
families of all candidates of that clip that overlap the cut, as letters A, B, ..., and a last letter "none of these".
Question: "Which of these sounds can you hear in this recording? Answer with every letter that applies, separated by
commas.\n(A) ...\n...\n(X) none of these". Asked twice, options in forward and in reversed order (position bias).
Score = mean over the two orders of the first-token probability of the candidate's own letter (softmax over the
option letters); also kept: the greedy answer (up to 16 tokens) and whether it names the own letter in both orders.

Truth: right if a gold sound (any: needed, visible or obvious) of the same family overlaps the candidate with 0.5 s
slack; wrong otherwise. Reported: AUROC over all candidates, over kept-only, over dropped-only, and over the frozen
system's DEV pictures (right = hit / don't-care / visible-source picture; wrong = other-sound or no-sound picture);
accuracy of "named in both orders".

## Not done here

No threshold is chosen, nothing is adopted, TEST is not touched.
