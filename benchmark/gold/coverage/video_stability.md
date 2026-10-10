# Stability of the video droppers (10 Oct), all 158 clips, clip-grouped 5-fold CV (seed 0), bars from quantiles

| dense-frame variant | dense alone, out of fold (no hit lost) | dense OR off-screen, out of fold (by cost) | p vs v1.4 |
|---|---|---|---|
| 12 frames, 1/8 s, wording 1 (first run) | 58 / 27, 2.013 | 57 / 22, 1.975 | 0.107 |
| 16 frames, 1/10 s, wording 1 | 59 / 33, 2.063 | 57 / 23, 1.987 | 0.170 |
| 12 frames, 1/8 s, wording 2 | 56 / 28, 2.076 | 56 / 23, 2.013 | 0.364 |

v1.4: 59 / 34, 2.076. Hand-picked bars in video_combo.md (57 / 20, 1.949) were from a coarser grid that included the
first run's best bar; the quantile grid here is the fairer read. The dense-frame check alone is not stable (its gain
depends on frame count and wording); the combination stays below v1.4 in all three variants (about 11 fewer wrongs for
2-3 hits) but is not significant on 158 clips.

## Off-screen question (offscreen_var.md)

The same wording as screen_reason.py on 6 frames, but frames shrunk to 640 px, gives 57 / 29 out of fold instead of
58 / 25; two other wordings and 10 frames give 58-59 / 31-34; the mean of all six gives 57 / 31. None is significant
(p 0.27-1.0). Conclusion: the VLM yes/no margins move with wording, frame count and frame size more than the effect we
measure; no video dropper is adopted.
