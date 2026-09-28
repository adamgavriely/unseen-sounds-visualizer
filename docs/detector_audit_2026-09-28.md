# Detector audit, 2026-09-28: where the false spans and misses come from

Descriptive only. No model run, no selection, nothing on DEV/TEST. Shipped stage-4 stack (BEATs 0.35 + FlexSED 0.8, twin
rule, FlexSED clip veto 0.3, BEATs self-veto b 0.1218, PANNs off), rebuilt from the saved caches on the 280 (fit set) and
the 415 (held-out). Code: `benchmark/detector_audit.py`; all numbers and every item: `benchmark/detector_audit.json`
(cluster job 31329865, CPU partition, `slurm/job_audit_detector.sh`).

The script checks itself: its spans equal `detector_round5.run` span for span on every clip, and its numbers equal the
round-5 gate-1 baseline to 1e-9.

| baseline | 280 (fit) | 415 (held-out) |
|---|---|---|
| C-overlap / C-onset | 3.036 / 3.850 | 1.928 / 2.207 |
| recall (overlap) | 51.3 % | 49.7 % |
| false spans | 207 (4.44 / min) | 228 (3.30 / min) |
| consequential events | 224 | 171 |

**Caveat, read first.** The harness scores with `config.LABEL_FILTER = "lists"` (the default). The shipped profile uses
`"depictable"`. **42 / 207 (20 %) and 47 / 228 (21 %) false spans have labels that the shipped system never draws**, for
example Hum, Mains hum, Sine wave, Radio, Distortion and Rub. Most of them (39 per set) are phantoms (bin c). Only 5 and 1
misses (all Rumble) are affected. Nothing below is re-scored with that filter.

## 1. False spans

Bins, in this order: **a** = the span overlaps a gold event of a *related* family, meaning a sibling or cousin. Parent and
child labels already count as a match. "Related" means the two labels share an ancestor in the AudioSet ontology that is
not one of the 7 top branches. **b** = the same family has a gold event somewhere else in the clip, but it does not overlap
the span (a timing problem). **c** = no related gold event at all (a phantom).

| bin | 280 | 415 |
|---|---|---|
| a related family overlaps | 59 (28.5 %) | 46 (20.2 %) |
| b same family elsewhere (timing) | 11 (5.3 %) | 26 (11.4 %) |
| c phantom | **137 (66.2 %)** | **156 (68.4 %)** |
| (both a and b true) | 9 | 5 |

Top families per bin (count):

| bin | 280 | 415 |
|---|---|---|
| a | Train 10, Vehicle 7, Siren 7, Helicopter 4, Bird 4, Bee 3, Groan 3, Screaming 2, Water 2, Babbling 2 | Bicycle 4, Baby cry 3, Telephone 3, Sawing 2, Screaming 2, Mallet perc. 2, Chant 2, Yodeling 2, Busy signal 2, Helicopter 2 |
| a, main pairs | Train over Truck 10 (one clip), Siren over Ice-cream truck 5, Bee over Buzz 3 | Bicycle over Boat 4, Telephone over Beep 3, Busy signal over Beep 2 |
| b | Crowd 3, Dog 2, then singles | Water 4, Dog 3, Chink 2, Whoosh 2 |
| c | Vehicle 10, Hum 9, Mains hum 7, Explosion 7, Sine wave 5, Blender 5, Steam 4, Bell 3, Rub 3, Ratchet 3 | Gunshot 7, Hum 6, Glass 6, Effects unit 6, Water 6, Scissors 5, Radio 5, Steam 4, Distortion 4, Rub 4 |

Bin b gap to the nearest same-family event: median 0.96 s (280) and 0.81 s (415). About half are closer than 1 s, so they
are near misses in time, not wrong sounds.

Phantoms (bin c), in more detail:

| | 280 (n 137) | 415 (n 156) |
|---|---|---|
| raised by BEATs only | 110 (80 %) | 131 (84 %) |
| raised by both (BEATs span with a FlexSED twin) | 12 (9 %) | 13 (8 %) |
| raised by FlexSED only | 15 (11 %) | 12 (8 %) |
| BEATs Speech >= 0.3 in the span | 70 (51 %) | 95 (61 %) |
| BEATs Music >= 0.3 in the span | 41 (30 %) | 61 (39 %) |
| **BEATs Speech or Music >= 0.3** | **90 (66 %)** | **123 (79 %)** |
| BEATs-origin, confidence 0.35-0.5 | 55 / 122 (45 %) | 65 / 144 (45 %) |
| BEATs-origin, FlexSED not asked for the family (so the clip veto always passes) | 44 / 122 (36 %) | 44 / 144 (31 %) |
| BEATs-origin, FlexSED in-span 0.5-0.8 / >= 0.8 | 25 / 21 | 46 / 17 |
| gold under the phantom (top) | Mechanisms 28, Male speech 27, Music 18 | Male speech 59, Music 40, Mechanisms 30 |

## 2. Missed consequential events (C-overlap)

Each miss gets one bin, checked in this order: **iv** (heard above the bar, then removed by a veto) > **v** (a shown span
of a related family overlaps it) > **t** (a same-family span is shown within 1 s, but the overlap is too small; t is part of the task's bin vi, split
out because it is the most common "other") > **ii**
(FlexSED 0.4-0.8 within 1 s) > **iii** (BEATs 0.175-0.35) > **i** (no score >= 0.2 from either detector) > **vi** (other).
"Masked" = the gold event is at least half under gold Speech/Music.

| bin | 280 | 415 |
|---|---|---|
| iv removed by a veto | 19 (17.4 %), masked 0 | 9 (10.5 %), masked 6 |
| v heard under a related name | 0 | 5 (5.8 %), masked 2 |
| t shown, overlap too small | 14 (12.8 %), masked 2 | 15 (17.4 %), masked 5 |
| **ii FlexSED 0.4-0.8** | **40 (36.7 %)**, masked 8 | **36 (41.9 %)**, masked 24 |
| iii BEATs 0.175-0.35 | 7 (6.4 %), masked 3 | 7 (8.1 %), masked 4 |
| i unheard (< 0.2) | 23 (21.1 %), masked 5 | 4 (4.7 %), masked 3 |
| vi other | 6 (5.5 %) | 10 (11.6 %) (8 = FlexSED 0.2-0.4 only) |
| total | 109, masked 19 (17 %) | 86, masked 51 (59 %) |
| BEATs Speech/Music >= 0.3 during the event | 72 (66 %) | 69 (80 %) |

Flags can overlap. Without the order: ii 61 / 51, iii 31 / 26, both ii and iii 19 / 15. FlexSED was asked for every
missed family.

- **ii**: the FlexSED peak median is 0.65 (280) and 0.57 (415). Misses cluster in a few clips: 13 clips on the 280 (one
  clip has 16), 14 clips on the 415. Families: Dog, Gunshot, Alarm (280); Alarm, Baby cry, Dog, Telephone (415). Round 2
  already tried a plain FlexSED bar of 0.5 (cell E): recall went up to 59 %, but false spans went from 4.5 to 6.9 per
  minute and C went up.
- **iv**: 280 = FlexSED clip veto 13, weak BEATs twin absorbed a FlexSED >= 0.8 span 4, "twin rule + weak twin" 2 (10 clips, one clip
  has 6). 415 = clip veto 4, self-veto 3, twin rule 2. **One cause repeats: both vetoes compare the exact family name,
  not the ontology family.** Example 1: BEATs said "Telephone bell ringing" 0.78. FlexSED scored 0.76 on a label that
  matches Telephone in the ontology, but that label is not in the exact "Telephone" family (the JSON does not store which
  label). FlexSED's clip-max for the exact family was 0.115, so the clip veto removed a real phone (280, DCFrCX4HPO8,
  6 events). Example 2: FlexSED said "Alarm" 0.86, but BEATs heard the child "Ringtone". BEATs' clip-max for
  "Alarm" was below b, so the self-veto removed it (415, 1UXOJ3IACj4).
- **i** on the 280: Alarm 8, Thunder 8, Door 5, in 8 clips. No bar change can recover these.

## 3. Timing: hit under C-overlap, missed under C-onset

| | 280 | 415 |
|---|---|---|
| events | 63 (15 clips; one clip has 19) | 37 (14 clips) |
| signed onset error, median [q1, q3] | **-3.69 s [-5.37, -2.37]** | **-2.29 s [-4.10, -1.06]** |
| too early (< -0.5 s) / too late (> +1.0 s) | 58 / 5 | 34 / 3 |
| onset set by BEATs | 57 (90 %) | 34 (92 %) |
| onset set by the FlexSED twin rule | 6 (10 %), all early | 2 (5 %) |
| onset set by a FlexSED-only span | 0 | 1 |
| **merge: span starts on an earlier same-family event** | **57 (90 %)** | **32 (86 %)** |
| families | Dog 46, Alarm 9, Train 5 | Dog 12, Cat 6, Alarm 4, Crying 4, Baby cry 4 |

Almost every onset miss follows one pattern. A single long BEATs span covers a run of repeated sounds (barks, beeps,
meows): the BEATs score (2-s windows, 0.175 low bar) does not fall between the sounds. Which of the two keeps the span open
is not checked here. The span's onset is the onset of the first sound, so every later
sound in the run misses the onset window. This is not a lag problem: late onsets are only 5 and 3 events.

## What this says for the next detector idea

1. **Before any new idea, re-score the baseline with the shipped label filter** (`depictable`). About 20 % of today's
   false spans are labels that the shipped system never draws. This costs nothing and changes the target.
2. **Phantoms are the biggest cost.** They make up 2/3 of false spans. They are mostly BEATs-only (80-84 %) and mostly
   under speech or music (66 % / 79 % have BEATs Speech/Music >= 0.3). About a third are families that FlexSED is never
   asked about, so no veto touches them. Candidate: a stricter rule for BEATs-only spans when BEATs Speech/Music >= 0.3
   (a higher bar or FlexSED agreement), and ask FlexSED about the phantom families (or drop them if they are not
   depictable). Warning: FlexSED already scores >= 0.5 inside 38 % (280) and 44 % (415) of the BEATs-origin phantoms,
   so "FlexSED agrees" alone would keep most of them. Only the < 0.3 group (15 / 10) is clearly rejected.
3. **Ontology-aware vetoes** (parent/child counts as the same family in the clip veto and the self-veto). This is small
   and cheap, and it targets part of bin iv: 11 events in 3 clips on the 280 (all telephones) and 2 events in 1 clip on
   the 415. So the gain is small and depends on a few clips.
4. **Span splitting at a new rise inside a span** would fix most C-onset misses (the merge pattern). First decide if it
   matters to a viewer: the picture is already on during the run of sounds.
5. **FlexSED 0.4-0.8** is the largest miss bin, but a plain lower bar failed in round 2. It needs a precise second check,
   not a lower bar.

Missing number: the audit lists only false spans and misses. True (matched) spans are not stored, so this audit does not
say how many true spans a new rule would remove. That count is needed before choosing between ideas 2 and 5.

Definitions and limits: "near" = [start - 1 s, end + 1 s]. The detector's family = labels that match the gold label under
`E._same`, which includes ancestors, so FlexSED "near" scores can come from a parent label. The ontology file keeps one
parent per label, so a few relations are missed. Counts are small and clustered in a few clips, so read the percentages as
rough.
