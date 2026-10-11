# Pre-registration: clip sound-inventory LLM (J-B1), as a wrong-dropper on v1.7 (Adam, 11 Oct 2026)

Written before any score was read.

Model: Qwen3.8-27B (thinking off) for both steps, instead of Qwen2.5-VL (captions) + Qwen3-32B (scoring): neither is
cached on the cluster and the home disk has 28 GB free (Qwen3-32B is about 65 GB). Qwen3.8-27B is the pipeline's own
on-screen model and is cached.

Per clip:
1. Captions: one frame per second (rate 1) and one per two seconds (rate 0.5); each frame gets one sentence, "Describe
   what is visible in this frame in one sentence: the place, the people, animals, vehicles and objects, and what they
   are doing." -> "t=3s: ..." lines.
2. Detector timeline: every sound in the clip's stage-5 plan without the on-screen check (stage5_specs.json "B"), as
   "start-end s: label (confidence)".
3. For each v1.7 picture, the text-only model reads captions + timeline and answers a yes/no question about the picture's
   family; score = logit(yes) - logit(no) at the first answer token. Three wordings:
   W1 "Based on the video description and the sound detections, is a {family} sound really present in this clip around
      {start} s? Answer yes or no."
   W2 "Given what is seen in the video and what the detectors heard, would a {family} sound plausibly be heard in this clip
      at about {start} s? Answer yes or no."
   W3 "Is it likely that the detector's '{family}' at {start} s is a mistake, given the video description? Answer yes or
      no." (sign flipped: high = present)
   6 scores per picture (3 wordings x 2 caption rates), plus their mean (each z-scored over the pictures).

Rule: drop a v1.7 picture when its score < t; t from the score's quantiles (incl. off); clip-grouped 5-fold CV
(random.Random(0) over dev + test stems, folds sh[k::5]) by onset cost, for each of the 7 scores.
Adopt (Adam's bar) only if out of fold, for the mean score, wrongs drop by >= 4 and hits drop by <= 1 vs v1.7
(59 hits, 13 wrong); the per-variant rows are reported to show how stable the effect is across wording and caption rate.
