# Clip sound-inventory LLM (Qwen3.8-27B captions + detector timeline) as a wrong-dropper, all 158 clips

pictures scored: 73 of 73; v1.7: 59 hits, 13 wrong, cost 1.810

| score | out of fold hits | wrong | cost | CV choices | pass (>= 4 wrongs out, <= 1 hit lost) |
|---|---|---|---|---|---|
| W1_r1.0 | 59 | 13 | 1.810 | [None, None, None, None, None] | no |
| W1_r0.5 | 59 | 13 | 1.810 | [None, None, None, None, None] | no |
| W2_r1.0 | 59 | 13 | 1.810 | [None, None, None, None, None] | no |
| W2_r0.5 | 59 | 13 | 1.810 | [None, None, None, None, None] | no |
| W3_r1.0 | 55 | 11 | 1.886 | [0.28, None, None, None, None] | no |
| W3_r0.5 | 55 | 11 | 1.886 | [0.16, -0.5, -0.5, -0.5, -0.5] | no |
| mean | 59 | 13 | 1.810 | [None, None, None, None, None] | no |

mean score: hits median 0.17

Score of hits vs wrongs (median), and how many of the 13 wrongs score below the lowest hit:

| score | hits median | wrongs median | wrongs below the lowest hit |
|---|---|---|---|
| W1 rate 1 | -1.00 | -2.00 | 0 of 13 |
| W2 rate 1 | 0.25 | 0.38 | 0 of 13 |
| W3 rate 1 | 1.50 | 1.00 | 1 of 13 |
| W1 rate 0.5 | -0.88 | -1.62 | 0 of 13 |
| W2 rate 0.5 | -0.25 | 0.50 | 0 of 13 |
| W3 rate 0.5 | 1.50 | 1.38 | 1 of 13 |

Bar not met: not adopted.
