# Opus H: an outside look at what is left (11 Oct)

## The arithmetic first

- v1.7 has 13 wrong pictures. Removing all of them saves 26 / 158 = 0.165, under the 0.19 floor. **Wrong-picture ideas are finished.**
- Only hits count: about 8 more of the 65 misses. 19 are never heard; 46 are dropped across 17 steps (at most 9 per step, leak_table.md); 19 are shorter than 1 s; only **7 are labelled "masked"** (under a louder sound).
- The one ceiling above the floor that has not already failed: a perfect listener with the F8 / FineLAP filters lifted, +8 DEV hits (ceiling_final.md). Gate attacks all failed.
- Power: to show a 0.10 gain you need about (0.19 / 0.10)^2 x 158 = 570 random clips.

## Ranked ideas (pass bars on DEV; TEST once at the end)

| # | idea and procedure | cost | pass bar | expected | risk |
|---|---|---|---|---|---|
| 1 | **Keep-score trained on outside human labels.** Label each stage-4 burst of the 340 fresh AudioSet-Strong clips real / not from the strong labels (example: a Dog burst at 3.1 s is real if Bark starts at 2.8-4.1 s). Train trees on the keep_score.py features. Replace the listener + F8 / FineLAP decision with it. | 1 day, under 2 GPU-h (stage-4 outputs exist, r13freshf0*); about 3k real events vs 59 on DEV | DEV at least 35 hits at at most 6 wrong (now 31 / 4), and in 4 of 5 clip folds | +2 to +5 hits | spends the fresh check; features are from D'; AudioSet is not film audio |
| 2 | **Label where systems disagree.** Add the 30 held-out clips already done (35 needed sounds). Then label only prefilled clips where v1.7 and arm 1 differ (about 40 clips, 3 h). | 0 GPU, Adam's time | a measure: who wins the disagreements | power, not a gain | Adam closed "more clips" on 5 Oct (a re-ask); not a random sample, so a second set |
| 3 | **Residual ear** (opusD #1, planned 10 Oct, not run): remove the loudest stem with AudioSep, rerun BEATs + FlexSED | 4 A100-h | opusD bar | +1 to +2 | ceiling about 7 masked misses |
| 4 | **Captioner inventory** (opusD #2, not run): Qwen3-Omni-Captioner families become FlexSED queries | 3 H200-h | opusD bar | +1 to +2 | most never-heard sounds are short |
| 5 | **Listener self-consistency:** both listeners on 5 crops (+-0.5 s, +-3 dB), averaged | 4 GPU-h | only as features for #1 | 0 to +2 | perfect listener with filters as shipped = about 2 hits |
| 6 | **What the viewer sees:** 5-8 viewers, sound off: can they name the event, is the picture too short? | 1 day | none (new result) | no cost gain | recruiting deaf viewers in 2 days is hard |

## Not worth it

- Vetoes or calibration for wrong pictures: ceiling 0.165.
- Visibility model trained on AVATAR / VGGSound: tried (0.63 vs gate 0.78).
- Another detector backbone or fine-tune: 25 rounds failed.
- Pseudo-labels on the 415 held-out clips: circular (same detectors).
- Calibrating listeners alone: it keeps their order, so no new hits.

## Honest answer

Both. The benchmark is the bottleneck for **showing** a gain (158 clips hide anything under about 8 hits). The system's pool is also thin. Only idea 1 aims at the one open ceiling; if it fails, stop system work and do ideas 2 and 6.
