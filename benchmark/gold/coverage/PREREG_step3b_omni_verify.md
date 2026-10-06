# Pre-registration: Step 3b, uniform Qwen3-Omni verification of every DEV candidate burst (measure only)

Written and committed 2026-10-06 before any answer was generated. Extends PREREG_step3_omni_probe.md (that run
continues unchanged). Model: Qwen/Qwen3-Omni-30B-A3B-Instruct, thinker only, bf16, one H200. DEV only. All answers
cached in `omni_verify_dev.json`. Nothing is adopted; TEST is not touched.

## Bursts

Every DEV candidate in the decision trail (BEATs, FlexSED, FlexSED band, DASM; kept and dropped), Speech / Music
families left out, grouped per clip by canonical family; same-family candidates with a gap <= 2.5 s join one burst
(start = earliest start, end = latest end). 1196 bursts.

## (1) Verifier: closed family choice (audio)

Cut: the clip's audio [start - 2, end + 2] s, 16 kHz. Options: the burst's family; its AudioSet siblings (the other
children of the parent of every member label, mapped to canonical families, Speech / Music removed, alphabetical, at
most 12); "none of these". Question: "Which ONE of these best names the sound you hear in this recording? Answer with
the letter only." Asked in forward and reversed option order. Saved: the softmax over the option letters at the first
answer token, per order. Score = mean over the two orders of P(own family).

## (2) Onset gate (audio + dense video)

Cut [onset - 1, onset + 1] s, onset = burst start; frames at 4 per second (qwen-omni-utils fps=4) with the cut's audio.
Question: "You are watching a short video with its sound. A sound of {family} starts in it. Is the thing making this
sound on screen, or is there a visible event that makes it, like a flash or a blast? Answer yes or no."
Saved: P(yes) = softmax over the max yes logit and the max no logit.

## (3) Still heard? (audio, for the hold)

For each second t from the burst start to its end + 3 s (clip end at most): cut [t, t + 1] s padded to 2 s around it
([t - 0.5, t + 1.5]). Question: "Is the sound of {family} heard in this recording? Answer yes or no." Saved P(yes).

## Truth and reports (from gold, after the run)

- Burst right: a gold sound (any) of the same family (score_per_sound.same_family) overlaps the burst with 0.5 s slack.
  Report: AUROC of (1) right vs wrong, overall, by origin mix, and on bursts that become frozen DEV pictures.
- Gate: bursts that match a gold sound of the same family; positive = that gold sound is ticked visible, negative = all
  matched sounds needed (obvious-only left out). Report AUROC of (2) on all of them and on the subset the current gate
  judged (its gate records), with the current gate's own AUROC (share of stretches seen) and the a/b rule's on the same
  subset.
- Still heard: per second, truth = inside a same-family gold sound. Report AUROC of (3).

## Then (CPU, separate pre-registration)

"Start at the earliest same-family heard evidence inside the onset window", tested on DEV after these answers exist.
