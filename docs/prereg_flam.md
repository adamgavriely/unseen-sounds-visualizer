# Pre-registration: FLAM as the sound detector (scope v2, point 1, first swap)

*Committed 2026-09-17 evening, before the run. Smoke test on the ambulance clip: FLAM
loads in the `sota` env (0.7 GB), scores a clip in 15 s, gives the siren 0.99 and nothing
else above 0.2 except "alarm" 0.75 (a near-synonym) and one padding artefact.*

## What is swapped

Stage 4's detector. BEATs (2022, clip-level tagger, 527 fixed classes, 2-s windows) is the
measured bottleneck: 7 of the 11 wrongly withheld test pictures are sounds that never reached
its bar under speech or music, and ~80% of unnecessary pictures are its phantoms. **FLAM**
(Adobe, ICML 2025; `openflam` v1-base) is a frame-wise language-audio model: it scores any
text query per audio frame. It is used here with the **same 527 AudioSet label names as
queries**, so everything downstream (family merge, gate, dedup, depiction) is unchanged and
the comparison isolates the detector.

Scoring: 10-second windows at 48 kHz (FLAM's input), padding masked; per query the sigmoid
of the local similarity per frame; the shipping span rule on top (reach the bar, extend
through half the bar, minimum 0.5 s), identical to BEATs'.

## The bar is chosen on DCASE gold, not on our clips

FLAM's scores are not on BEATs' scale, so a bar must be chosen. It is the bar at which
FLAM's **false-positive rate on the 255 DCASE gold events equals BEATs' (5.2 spans/min)**,
or the nearest bar below that rate. Nothing is tuned on the development or test clips.

## Pass bars (declared now; the same measures as the seven earlier attempts)

| measure | BEATs (current) | FLAM must reach |
|---|---|---|
| DCASE recall on the 126 events overlapping speech or music (the masked case) | 9.5% | **≥ 24.5%** (+15 points) |
| DCASE recall on the 129 clear events | 32.6% | **≥ 32.6%** (not worse) |
| DCASE false positives per minute | 5.2 | ≤ 5.2 (by construction of the bar) |
| dev: labelled real detections still fired (of 23) | 23 | **≥ 21** |
| dev: labelled phantoms no longer fired (of 77) | 0 | **≥ 40** |

**PASS iff all five.** On pass, FLAM becomes stage 4 (`AED_MODEL = "flam"`) and the 100-clip
protocol is re-run once as **v4**, reported next to v3 with the same judge and references.
On fail, FLAM joins the attempts table with its numbers.

## Not done

No prompt engineering of the queries after seeing the numbers; no per-class bars; no
re-tuning of the gate. The reason for a fail is diagnosed and written up like the others.

## Outcome (added 2026-09-17 night, after the run)

`benchmark/flam_setting.json`. (A bookkeeping error is disclosed first: the cache applied a
sigmoid on top of FLAM's "unbiased" similarity, which is already a probability; the cached
scores were converted back with the logit before evaluation — an exact inverse up to
float16 rounding. The code no longer applies the sigmoid.)

| bar | masked-event recall | clear-event recall | false positives / min |
|---|---|---|---|
| BEATs 0.35 (current) | 9.5% | 32.6% | 5.2 |
| FLAM 0.35 | 80.2% | 91.5% | 1957 |
| FLAM 0.75 | 73.0% | 90.7% | 1287 |
| FLAM 0.95 | 69.8% | 88.4% | 693 |

**FAILED as pre-registered:** no bar brings the false-positive rate within reach of BEATs'
5.2 spans per minute, so the pass bars cannot even be applied at a matched rate (dev
numbers at the fallback bar 0.95 are not meaningful and are not reported as results).

**What the numbers say anyway.** Recall is the striking part: FLAM finds **7 to 8 of every 10
sounds buried under speech or music** where BEATs finds 1 — the masked-miss problem is
solvable by a frame-level language-audio model. The failure is precision with *this query
set*: with the 527 bare AudioSet names as prompts, FLAM fires 60+ labels per 5-second clip
even at 0.95 ("Hands", "Hammer", "Owl", "Lullaby" in ordinary indoor scenes). FLAM was
trained on descriptive captions; single-word class names are poor prompts, and 527 of them
overlap heavily, so every real sound also lights up dozens of neighbours.

**Not done here** (per §Not done): no query rewriting after seeing the numbers. A follow-up
is a *separate* pre-registration: a small descriptive query vocabulary (the pipeline's ~28
families phrased as "the sound of …"), per-query calibration on held-out audio, and the same
five bars. Recorded as attempt eight in the attempts table, with the recall finding.
