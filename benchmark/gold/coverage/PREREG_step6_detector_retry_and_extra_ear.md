# Pre-registration: Step 6, one detector retry + ATST-F as an extra ear (DEV only)

Written and committed 2026-10-08 before any new mixture, training or arm was run. TEST untouched; nothing adopted.
After Step 5: fine-tuned ATST-F sorted better (burst AUROC 0.859) but heard fewer needed sounds (29 / 58).

## A. One retry of the fine-tune

Changes against Step 5, all declared now:
- Mixtures `mix2` (20 000 + 1 000), real-video-like beds: each bed = MUSAN noise or an FSD50K crowd clip, plus with
  probability 0.5 a second layer of MUSAN speech or music 6 dB lower; events get a synthetic room reverb (exponentially
  decaying noise impulse, RT60 uniform 0.2-0.8 s, wet / dry uniform 0.1-0.5); event level 5-20 dB under the bed as
  before.
- Lower learning rate: backbone 3e-6, head 3e-5 (was 1e-5 / 1e-4).
- Stronger teacher term: loss = 0.7 x BCE(student, teacher probabilities) + 0.3 x BCE(student, teacher probabilities
  with the inserted events' frames set to 1).
- 10 rounds; the round kept is the one with the lowest loss on the 1 000 held-out mixtures.
- Detection bar chosen on DEV: the highest bar in 0.01 .. 0.30 (step 0.01) at which the ear hears at least 34 of the 58
  needed sounds (the untrained ATST-F's count at its bar). Reported: burst AUROC (bar-free), needed sounds heard, and
  spans per clip at that bar, next to the untrained model at the same rule.

## B. ATST-F as an extra ear alongside BEATs (no retraining needed)

Two uses, ears = untrained ATST-F and the Step 5 fine-tuned ATST-F (and the retry's model if A finishes first):

B1. **Union of candidates, full pipeline.** In stage 4, after every veto (where the DASM-rescue spans join), each ATST-F
span of a depictable family (frame probability >= bar, length >= 0.3 s, gaps < 0.5 s bridged) that no surviving event
of the same family overlaps joins as a new candidate (confidence = its peak mapped to 0.35 at the bar). Bars: 0.3 and
0.5. Every later step is the frozen system's: post-rules (DASM two-witness), stage 5 gate (new stretches asked live,
memoised), display. Then the a/b gate re-decision (Step 2). Arms: {untrained, fine-tuned} x {0.3, 0.5} = 4.
Pass: more DEV hits than the a/b candidate (32) at <= 16 wrong.

B2. **Feature for keeping / dropping.** Step 4's keep-score, same model and CV, plus one feature: the ear's family max
probability in the burst. Reported as Step 4 (hits at wrong <= 5 / 10 / 15; clear win = >= 3 more hits at equal wrong
or >= 3 fewer wrong at equal hits vs 32 / 16).
