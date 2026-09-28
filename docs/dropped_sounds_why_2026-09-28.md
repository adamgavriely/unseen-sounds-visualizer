# Why are "heard but dropped" sounds low? 2026-09-28

Descriptive only. CPU, the 280 and the 415, nothing on DEV/TEST. Code: `benchmark/dropped_why.py`; every event and span with
its features: `benchmark/dropped_why.json` (cluster job 31330549, `slurm/job_audit_dropped_why.sh`). The shipped stack is
rebuilt and checked against the round-5 baseline (to 1e-9) by `benchmark/detector_audit.py`.

**Dropped** = consequential events that the shipped stack misses in audit bins ii (FlexSED 0.4-0.8 near it),
iii (BEATs 0.175-0.35) or iv (removed by a veto). The 280 has 66 dropped events in 25 clips. The 415 has 52 in 25 clips.
**Band phantoms** = FlexSED spans with a peak of 0.4-0.8 that pass every shipped rule at a FlexSED bar of 0.4 (twin rule,
clip veto, self-veto b 0.1218) but overlap no gold event of their family. These are the false spans a bar of 0.4 would
add. **Masked** = BEATs Speech or Music >= 0.3 during the event (the gold "masked" flag is shown too). **Loudness** = RMS dB
of the whole mix during the event, minus the clip's median. It measures the mix, not the sound source alone.

## 1. Is it about speech/music? Partly, not mostly

| | 280 | 415 |
|---|---|---|
| hit events that are masked (BEATs) | 43 / 115 (37 %) | 53 / 85 (62 %) |
| dropped events that are masked (BEATs) | 49 / 66 (74 %) | 41 / 52 (79 %) |
| dropped events that are **not** masked | 17 (26 %) | 11 (21 %) |
| drop rate: masked vs not masked (BEATs) | 43 % vs 16 % | 34 % vs 22 % |
| drop rate: gold-masked vs not | 35 % vs 28 % | 39 % vs 22 % |
| **band phantoms that are masked (BEATs)** | 96 / 127 (76 %) | 209 / 249 (84 %) |

Speech/music raises the drop rate: about 2.7x on the 280 and 1.5x on the 415. But many hits are masked too, and one drop in
four or five is not masked. Band phantoms are masked just as often as dropped events. So "masked" explains part of the
losses, but **it cannot tell a real sound from a phantom**.

## 2. Features of the dropped events (median [q1, q3])

| | hit 280 | dropped 280 | hit 415 | dropped 415 |
|---|---|---|---|---|
| duration, s | 0.80 [0.41, 2.15] | **0.72 [0.24, 1.69]** | 1.00 [0.49, 2.85] | **0.43 [0.26, 1.21]** |
| bin ii only: duration, s / shorter than 0.5 s | | 0.44 / 25 of 40 | | 0.36 / 25 of 36 |
| loudness vs clip median, dB | 9.9 [3.9, 13.1] | 7.3 [2.4, 12.1] | 4.3 [0.7, 13.3] | 2.6 [0.4, 6.8] |
| other gold events overlapping it | 4 [2, 5] | 2 [1, 2.75] | 2 [1, 3] | 2.5 [2, 4] |
| BEATs score for the family (+-1 s) | 0.79 [0.54, 0.91] | **0.29 [0.20, 0.40]** | 0.82 [0.64, 0.89] | **0.18 [0.08, 0.30]** |
| FlexSED peak for the family (+-1 s) | 0.50 [0.40, 0.83] | 0.66 [0.63, 0.76] | 0.61 [0.43, 0.85] | 0.57 [0.50, 0.74] |
| FlexSED peak inside the event | 42 % | 61 % | 54 % | 75 % |
| FlexSED shift when outside, s | -0.01 [-0.58, 0.50] | 0.38 [-0.48, 0.75] | 0.49 [-0.02, 0.76] | 0.56 [0.11, 0.73] |
| top families | Dog 70, Alarm 15, Train 12 | Dog 17, Telephone 13, Alarm 10, Gunshot 9, Glass 5, Baby cry 3 | Dog 27, Alarm 12, Cat 10 | Alarm 8, Dog 7, Telephone 7, Explosion 6, Train 6, Baby cry 5 |

By bin (280 / 415): ii is short (median 0.44 / 0.36 s) and BEATs barely hears it (0.27 / 0.11). iv is long (1.7 / 1.2 s)
and BEATs hears it well on the 280 (0.74). It is lost only to the exact-name veto (see `detector_audit_2026-09-28.md`).
iii has only 7 events per set.

**What happens to bin-ii events at a FlexSED bar of 0.4:**

| | 280 (40) | 415 (36) |
|---|---|---|
| no 0.4-0.8 FlexSED span covers the event | 29 (22 shorter than 0.5 s) | 20 (19 shorter than 0.5 s) |
| covered, but lost to the BEATs self-veto | 4 | 7 |
| covered, but swallowed by a weak BEATs twin | 4 | 7 |
| covered and shown | 3 | 2 |

Most bin-ii events are short, single sounds (barks, gunshots, beeps). A lower FlexSED bar does not recover them, because no
FlexSED span of at least 0.5 s covers them. My guess is that the 0.5-s minimum span length and FlexSED's peak shift are the
cause, but I did not check this.

## 3. The trade at a FlexSED bar of 0.4

| | 280 | 415 |
|---|---|---|
| new phantoms shown (band phantoms, admitted) | **127** (62 clips) | **249** (110 clips) |
| missed consequential events recovered | **5** (3 clips) | **2** (2 clips) |
| band phantoms before the self-veto removes them | 931 | 1674 |

## 4. Which features separate real from phantom

**A. As asked: dropped events (gold times) vs admitted band phantoms.** Each rule is fitted on the 280 (best Youden J) and
applied unchanged to the 415.

| feature | AUROC 280 / 415 | rule | 280 kept: true / phantom | 415 kept: true / phantom |
|---|---|---|---|---|
| **BEATs family score** | 0.74 / 0.63 | >= 0.225 | 47/66 / 27/127 | 22/52 / 56/249 |
| FlexSED peak | 0.72 / 0.45 | >= 0.63 | 51/66 / 34/127 | 22/52 / 123/249 |
| loudness vs clip | 0.64 / 0.51 | >= 4.9 dB | 42/66 / 37/127 | 14/52 / 90/249 |
| BEATs Speech (masking) | 0.50 / 0.55 | <= 0.525 | 52/66 / 74/127 | 26/52 / 153/249 |

The duration rule (<= 0.51 s keeps 28/66 true, 0 phantoms) is an artifact and is left out: phantoms are spans, which are at
least 0.5 s long by construction, while events have gold durations.

**B. Span level (a rule could use this): 0.4-0.8 FlexSED spans on a missed event vs band phantoms, any fate** (true n = 20
/ 13, phantom n = 931 / 1674).

| feature | AUROC 280 / 415 | rule | 280 kept: true / phantom | 415 kept: true / phantom |
|---|---|---|---|---|
| **BEATs family score** | **0.86 / 0.84** | >= 0.27 | 16/20 / 155/931 | 6/13 / 233/1674 |
| FlexSED peak | 0.77 / 0.60 | >= 0.62 | 18/20 / 368/931 | 8/13 / 730/1674 |
| other gold events overlapping (descriptive, not usable at run time) | 0.67 / 0.67 | >= 3 | 16/20 / 439/931 | 10/13 / 1024/1674 |
| BEATs Music | 0.66 / 0.65 | >= 0.13 | 18/20 / 530/931 | 11/13 / 1035/1674 |
| loudness | 0.61 / 0.66 | >= 4.9 dB | 14/20 / 301/931 | 5/13 / 510/1674 |

Only one feature holds on both sets: **BEATs' own score for the family**. Real band sounds have BEATs >= 0.27. Phantoms
mostly do not. But even the best rule keeps about 10 phantoms (280) and about 40 (415) for each real event. FlexSED peak
and loudness do not hold on the 415. Masking does not separate at all.

## Short answer for Adam

- Speech/music is part of the story: masked events are dropped 1.5-2.7x more often. It is not most of it: one drop in
  four or five is unmasked, many hits are masked, and phantoms are masked just as often.
- The dropped sounds that stand out are **short single sounds (< 0.5 s) that BEATs barely hears** (family score about
  0.2-0.3 vs 0.8 for hits). A FlexSED bar of 0.4 does not recover them: it adds 127 / 249 phantoms for 5 / 2 recovered
  events.
- The veto losses (bin iv) are a different case. They are long, heard sounds, lost to the exact-name veto.

Limits: the counts are small and clustered in a few clips (bin ii: 13 / 14 clips). Loudness is the whole mix. The
separation tables are descriptive, with no significance test.
