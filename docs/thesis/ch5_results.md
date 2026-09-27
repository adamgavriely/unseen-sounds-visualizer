# Chapter 5 — Results (draft, 27 Sept 2026)

*Order fixed before the final numbers were read (`docs/panel_2026-09-26_plan.md` §A, signed 5/5) and confirmed by the
evaluation panel (`docs/panel3_topic3_rounds.md`, rounds 2–3 and "Topic 3 — outcome", signed by all three): the
pre-registered primary metric first, then the declared secondary family with a multiple-testing correction, then the
cost curve as an interval claim, then the analyses that explain the result. Every number below is read from a committed
file named beside it. Every p-value is two-sided, from the paired bootstrap draws; no one-sided share of draws (such as
"P(Δ > 0)") is printed as significance anywhere. Items marked **[pending]** are filled when their run or sitting
finishes.*

## 5.1 Set-up in one paragraph

Three systems share every stage up to the visibility gate: the same detector stack (BEATs ∪ FlexSED with two
cross-model vetoes), the same onset rule, the same label filter and the same renderer. **Ours** draws a sound only if the
gate finds its source off screen; **blind** draws every detected sound; **silence** shows nothing. A fourth arm, text
tags, is discussed in §5.10. Gold: one annotator, per-sound labels (label, onset, visible, obvious, importance). A needed
sound (neither visible nor obvious; scored when rated 2–3, Chapter 4 §4.6) is *hit* when a picture of its family starts within [−0.5, +1.0] s of
its onset. **Modelled viewer cost** per clip (β = 2 assumed; weights pre-specified, promoted to a reported outcome after
F1 was null) = 4 × missed needed sounds + β × wrong pictures. All intervals are paired clip bootstraps (2000 draws,
seed 0). **TEST** = 60 clips, *held out, with disclosed exposure*: the final table (amendment 21) is the tenth exposure of
these clips, all dated in §5.15, where the overlap with the detector's selection split is also listed. **DEV** = 49
clips used during development, shown beside TEST for transparency.

## 5.2 Three claims and how strong each is

1. **Confirmatory (pre-registered, Holm-corrected, TEST).** The primary metric, F1 against drawing every sound, shows no
   significant difference (§5.3). In the declared secondary family the gate cuts wrong pictures per clip (−0.52,
   replicated on DEV) and raises clean-clip accuracy (+0.22, replicated on DEV) at no detectable recall price; precision
   (+0.137) survives on TEST only and is not replicated on DEV (§5.4).
2. **Post hoc, with a pre-declared subgroup.** The modelled viewer cost was promoted to a reported outcome after F1 came
   out null (amendment 9; its weights predate the gold). Its Holm row (§5.4), its β curve (§5.5) and the off-screen
   subgroup row against silence (§5.6; subgroup declared in amendment 5) carry that label. Holm corrects for the number
   of rows; it does not repair a post-hoc entry.
3. **Descriptive (no test, CIs only).** The per-category ours-vs-blind rows (§5.6), the detector on the 280 AudioSet-Strong
   clips and the detector ceiling (§5.7–5.8), the gate's balanced accuracy (§5.9), the picture glance test (§5.11) and
   the timing numbers (§5.12).

## 5.3 Primary metric: F1 against drawing every sound — no significant difference

| | ours | blind | Δ (ours − blind) | 95 % CI | p |
|---|---|---|---|---|---|
| TEST, 60 clips, 43 needed sounds | 0.381 | 0.322 | **+0.059** | [−0.030, +0.144] | 0.18 |
| DEV, 49 clips, 36 needed sounds | 0.378 | 0.330 | +0.048 | [−0.054, +0.131] | 0.30 |

*Source: `benchmark/gold/holm_test_final_v33_test_bench.json`, `holm_dev_monocap_v31_dev.json`.*

The pre-registered primary is null: +0.059 [−0.030, +0.144] on TEST, positive on both halves and significant on neither.
The design could not have detected an effect of this size: with 60 clips the smallest effect detectable at 80 % power is
about +0.13 (from the interval width), and about 320–570 clips would be needed for a true +0.03–0.05. The oracle
diagnostic points at the cause (`docs/GOLD_RERUN_2026-09-22.md` §9a, 22 Sep renders, 109 benchmark clips, which include
the TEST clips — exposure row 1 in §5.15): when the detector is replaced by the annotator's own sound list, the same gate's
F1 gain is significant (+0.067 [+0.012, +0.117]); giving back only the missed sounds, while keeping every false alarm the
detector made, already makes it significant (+0.075 [+0.029, +0.119]). So misses, not false alarms, cause the null. F1
also weighs a removed wrong picture and a lost right one equally, so it cannot see the trade the gate makes (§5.7).

## 5.4 Declared secondary family (ours − blind), Holm-corrected, per table

| row | TEST Δ [95 % CI] | p | Holm | DEV Δ [95 % CI] | DEV Holm |
|---|---|---|---|---|---|
| **Wrong pictures / clip** | **−0.517** [−0.800, −0.283] | < 0.001 | **< 0.001 ✓** | −0.531 [−0.776, −0.306] | < 0.001 ✓ |
| Precision — *TEST-only, not replicated on DEV* | +0.137 [+0.043, +0.256] | 0.002 | 0.010 ✓ | +0.115 [+0.020, +0.204] | 0.080 |
| Modelled viewer cost / clip (β = 2 assumed; post hoc, §5.2) | −0.833 [−1.434, −0.300] | 0.003 | 0.012 ✓ | −0.816 [−1.388, −0.204] | 0.040 ✓ |
| Clean-clip accuracy | +0.219 [+0.086, +0.364] | 0.001 | 0.006 ✓ | +0.231 [+0.074, +0.391] | 0.030 ✓ |
| F0.5 | +0.110 [+0.017, +0.210] | 0.023 | 0.069 | +0.093 [−0.003, +0.170] | 0.186 |
| Recall | −0.070 [−0.154, +0.000] | 0.103 | 0.206 | −0.083 [−0.200, +0.000] | 0.186 |
| Weighted F1 | +0.022 [−0.074, +0.103] | 0.580 | 0.580 | +0.016 [−0.094, +0.102] | 0.705 |

*Source: `family1` in the two files of §5.3.* ✓ = survives Holm at 0.05 within its table (seven rows). p is two-sided
from the bootstrap draws (ties counted on both sides; resolution 1/2000, so "< 0.001" means no draw on the other side).
Negative wrong-picture and cost differences are gains.

**Lead result.** The gate cuts wrong pictures per clip by −0.52 [−0.80, −0.28] (Holm p < 0.001), about half of blind's
0.93, **at no detectable recall price (−0.07 [−0.15, 0.00])**. "No detectable price" is not "free": the recall interval
touches zero and the point estimate is a loss of three needed sounds. The same row replicates on DEV (−0.53, Holm
< 0.001). Precision (+0.137) is significant on TEST but not on DEV (Holm 0.080), so it is stated as a TEST-only result.

**What the gate removed (TEST, from the existing counts).** Ours draws a subset of blind's pictures (gated ⊂ blind), so
every difference is a picture blind showed and ours did not:

| | blind | ours | removed by the gate |
|---|---|---|---|
| wrong pictures (TEST, 60 clips) | 56 | 25 | **31 wrong pictures removed** |
| needed sounds hit (TEST, of 43) | 19 | 16 | **3 needed sounds removed** |
| wrong pictures (DEV, 49 clips) | 50 | 24 | 26 wrong pictures removed |
| needed sounds hit (DEV, of 36) | 17 | 14 | 3 needed sounds removed |

About ten wrong pictures removed for each needed sound lost, on both halves; the break-even at β = 2 is two. *The
integers are recovered from committed files, not from a per-clip file (none is committed for `test_final_v33`): misses
and wrong pictures from the intercept and slope of each line in `cost_curve_test_final_v33.json` /
`cost_curve_dev_monocap_v31.json` (cost = 4 × misses / clips + β × wrong / clips), checked against ΔR = −3/43 and
Δwrong/clip = −31/60 in the Holm file and against amendment 23's DEV blind row (17 hits, 50 wrong pictures).*

## 5.5 The whole trade-off, not one weight: modelled viewer cost across β

*Figure: `benchmark/gold/cost_curve_test_final_v33.png` (DEV twin `cost_curve_dev_monocap_v31.png`).*
The price of a wrong picture for a deaf or hard-of-hearing viewer (β; a missed sound = 4) is not known, so the result is
stated as an interval, not at one weight. **On TEST, ours is the cheapest of the three systems for every β between 0.39
and 2.56** (it costs less than blind above β = 0.39 and less than silence below β = 2.56). On DEV the same interval is
0.46 to 2.33. The assumed β = 2 lies inside both. Blind itself costs less than silence below β ≈ 1.36 on both halves, so
beating silence at a low β is not the gate's achievement; the gate's own contribution is the lower end of the interval.
The same gate given the annotator's own sound list would cost 1.23 per clip at β = 2 on TEST (DEV 1.63) — the headroom
lies in detection.

| crossing | β | tag | split | date | status |
|---|---|---|---|---|---|
| ours = blind (ours cheaper above) | **0.39** | `test_final_v33` | TEST, 60 | 27 Sep | final |
| ours = silence (ours cheaper below) | **2.56** | `test_final_v33` | TEST, 60 | 27 Sep | final |
| ours = blind | **0.46** | `dev_monocap_v31` | DEV, 49 | 27 Sep | final — CI **[pending: bootstrap on `dev_monocap_v31`]** |
| ours = silence | **2.33** | `dev_monocap_v31` | DEV, 49 | 27 Sep | final — CI **[pending: bootstrap on `dev_monocap_v31`]** |
| blind = silence | 1.36 | both final tags | TEST / DEV | 27 Sep | final, descriptive |
| oracle (gold sound list + our gate) = silence | not crossed within β ≤ 4 | both final tags | TEST / DEV | 27 Sep | the curves stop at β = 4; not verified beyond |
| ours = blind / ours = silence | 0.40 / 1.27 | `v4b4` | benchmark, 109 (DEV + TEST) | curve committed 23 Sep | **superseded** |
| ours = blind / ours = silence | 0.35 / 1.37 [0.71, 2.19] | v4b6-era cell, `use_v4("59")` (8-s cap on, before amendment 21) | DEV, 49 | 23 Sep (`beta_specification.md` §2) | **superseded** |
| oracle = silence | 4.30 | as above | DEV, 49 | 23 Sep | **superseded** — not crossed within β ≤ 4; not verified from a committed file |

*Sources: `benchmark/gold/cost_curve_test_final_v33.json`, `cost_curve_dev_monocap_v31.json`, `cost_curve_v4b4.json`
(grid β = 0–4, step 0.1; crossings from the straight lines). The final curves reproduce the scored tables exactly
(amendment 21 addendum "Cost curve of the final tables": `--maxspan none --match-scorer`).* No confidence interval is
computed for the TEST crossings: it would be a new TEST number, and the evaluation panel's coordinator recommends not
computing it (the DEV interval beside the TEST point is enough). The oracle line reuses the cached gate votes of
`gate_gold` (frames sampled around the gold sounds).

## 5.6 Against showing nothing (family 2, Holm over three rows, kept as declared)

| row | TEST Δ [95 % CI] | p | Holm | DEV Δ [95 % CI] | DEV Holm |
|---|---|---|---|---|---|
| F1 vs silence, all clipsᵃ | +0.381 [+0.217, +0.557] | < 0.001 | < 0.001 | +0.378 [+0.242, +0.500] | < 0.001 |
| Modelled viewer cost vs silence, all clips | −0.233 [−1.000, +0.467] | 0.585 | 0.585 | −0.163 [−0.735, +0.408] | 0.642 |
| Modelled viewer cost vs silence, clips whose sound is off screen (TEST 13, DEV 14)ᵇ | −3.077 [−4.769, −1.385] | < 0.001 | < 0.001 | −0.857 [−1.857, 0.000] | 0.182 |

*Source: `family2` in the two files of §5.3.* The three rows are kept exactly as declared (deleting a row after seeing it
would be selective); only their labels change.
ᵃ Silence F1 ≡ 0; true by definition, carries no claim.
ᵇ Detection + picture result; gate-neutral (ours = blind = 4.00, Δ 0.00 [−0.62, +0.77]). One pre-declared subgroup
(amendment 5) of a post-hoc metric (amendment 9). Silence's cost here is 4 × its misses; ours and blind cost the same
(4.00), so the row measures detection and pictures, not the gate. It is not a tautology: ours could have cost more than
silence with about 1.5 wrong pictures per clip.

Neither row ᵃ nor row ᵇ is a headline or goes into the abstract. **Over all clips the system does not beat silence on
cost** (−0.23 [−1.00, +0.47], p 0.585; DEV −0.16, p 0.642): clips with nothing to draw can only cost, and busy "mixed"
clips are its weak category (ours 6.27 vs silence 5.33 per clip).

**Where the gate earns its keep (descriptive, CIs only; ours − blind, cost per clip at β = 2).**

| category (TEST) | clips | ours | blind | silence | ours − blind [95 % CI] | DEV twin |
|---|---|---|---|---|---|---|
| source on screen | 12 | 0.67 | 2.50 | 0.00 | **−1.83 [−2.83, −1.00]** | −1.44 [−2.23, −0.78] (18 clips) |
| mixed | 15 | 6.27 | 8.00 | 5.33 | **−1.73 [−3.73, −0.13]** | −1.33 [−2.67, −0.22] (9) |
| nothing to draw | 20 | 0.20 | 0.30 | 0.00 | −0.10 [−0.30, 0.00] | −0.50 [−1.50, 0.00] (8) |
| off screen (gate-neutral) | 13 | 4.00 | 4.00 | 7.08 | 0.00 [−0.62, +0.77] | +0.14 [−1.14, +1.43] (14) |

*Source: `categories` in the two files of §5.3.* The gate's evidence lies where a visible source can be silenced: on
screen and mixed clips.

## 5.7 Why F1 does not move: the detector, not the gate

- **Oracle diagnostics** (§5.3; `docs/GOLD_RERUN_2026-09-22.md` §9a): with the annotator's sound list the gate's F1 gain is
  significant; with the missed sounds alone given back it already is. Misses, not false alarms, cause the null.
- **Where needed sounds are lost** (DEV miss autopsy of the `v4b6` row of 23 Sep, an earlier pipeline, 21 misses;
  `docs/GOLD_RERUN_2026-09-22.md` §15): 11 never detected, 5 timing, 3 gate, 2 label filter. The final
  DEV row (§5.4) has 22 misses (14 of 36 hit); it has not been re-autopsied.
- **Detector on 280 human-labelled AudioSet-Strong clips** (descriptive, shipped bars, out of sample; `benchmark/audioset_stage4_report.json`):
  the shipped stack keeps BEATs' recall of consequential events within about 3 points (51.3 % vs 54.5 %) and cuts false
  spans per minute by 29 % (4.56 vs 6.41). (The detector-round scripts of §5.8 recompute the shipped stack at 4.46 false
  spans per minute on the same clips; the 4.56 of this report omitted the weak-twin absorption.) FlexSED at its shipped
  bar recovers none of the 31 masked consequential events there — and on the clean DEV-49 it recovers only 1 of 7 at that
  bar. Amendment 8's recovery count used scores below the adopted bar, so the union's recall gain at 0.8 comes from other
  sounds, not from the masked ones that motivated it.
- **Detector threshold and TEST overlap.** FlexSED's bar was chosen on a split overlapping 35 of the 60 TEST clips; on
  the clean DEV-49 it passes two of its three adoption rules and misses the third by 0.011 (`flexsed_recheck_dev49.json`).

## 5.8 The detector ceiling (amendments 22–25)

After the final table, three pre-registered detector rounds asked whether more needed sounds can be found without
paying for them in wrong pictures. None touched TEST; each rule was written before its numbers.

**Where the missing recall sits (DEV-49, diagnostic only — an oracle-bar read on the very sounds it would be scored on,
so no bar is chosen from it).** Of 36 needed sounds, the caches reach 21 at the shipped bars and 30 at looser bars; 4 are
below every detector (amendment 22). The detection panel splits the same ceiling into three tiers (reviewer cache read on
the 33 depictable needed sounds, `docs/panel3_topic1_rounds.md`, T1-a round 1; not a committed result file):

| tier | DEV sounds | what it means |
|---|---|---|
| reachable at the shipped bars | 22 of 33 | found now, or lost later (timing, gate, filter) |
| only FlexSED hears — masked under speech or music | 8 | FlexSED 0.5–0.8, BEATs below its bar; 4 of the 8 get no PANNs support, so the PANNs veto removes them at any FlexSED bar |
| below every detector | 3 | no public model tried scores them (a hammer, a bird, a civil-defence siren) |

**Amendment 22 (Stage 0, 280 AudioSet-Strong clips, a fit set).** No cell raised onset-recall while staying at or below
the shipped false-span rate (4.46 per minute); nothing was picked. Lower FlexSED bars and time-aligned corroboration of
FlexSED bought at most 0.4 points of onset-recall. Only the cascade (cell E: weak BEATs spans promoted when FlexSED or
PANNs agrees) raised recall a lot, at a false-alarm price (onset-recall 25.0 → 46.4 %, false spans 4.46 → 6.90 per
minute) (`benchmark/detector_round_stage0.json`).

**Amendment 23 (cascade E on DEV-49, blind arms decide).** E found **no new needed sound (17 hits vs shipped 17; ≥ 19
required)** and drew 30 more wrong pictures (80 vs 50), so its modelled viewer cost at β = 2 rose (**4.82 vs 3.59**). It
failed both parts of its rule. The 280's recall gain was mostly earlier starts on sounds already found.

**Amendment 24 (a new corroborator and a cost rule, the 280 as fit set).** Cost per clip C = 4 × missed consequential
events + 2 × false spans (C-overlap: a span must overlap the event; C-onset: it must start within [−0.5, +1.0] s).

| cell | C-overlap | C-onset | recall | onset-recall | false spans / min |
|---|---|---|---|---|---|
| shipped | **3.071** | 3.886 | 50.4 % | 25.0 % | 4.46 |
| F (twin fix) | 3.079 | 3.893 | 51.3 % | 25.9 % | 4.56 |
| E (cascade) | 3.614 | 4.014 | 58.9 % | 46.4 % | 6.90 |
| E-F 0.3 | 3.529 | 3.971 | 58.9 % | 45.1 % | 6.64 |
| E-F 0.5 | 3.321 | 4.107 | 58.9 % | 34.4 % | 6.02 |
| E-AND | 3.286 | 3.757 | 58.5 % | 43.8 % | 5.87 |
| D+PE | 3.136 | 4.050 | 53.6 % | 25.0 % | 4.95 |
| E+PE | 3.329 | 4.257 | 58.5 % | 29.5 % | 6.00 |
| U+PE | 4.750 | 5.536 | 52.7 % | 28.1 % | 9.71 |

*Source: `benchmark/detector_round2.json`.* **No cell beats shipped on cost C-overlap**, so nothing was picked and the new
held-out set (complex scenes, downloaded for this round) was not scored; its caches are kept unread. The exchange rate
explains why: cell E buys its recall at about **2.4 false spans per gained onset** (114 more false spans for 48 more
consequential onsets on the 280) and about **6 per gained event** (for 19 more events). At β = 2 a gained sound pays
for itself only at two false spans or fewer. The new text-queried detector with an independent encoder, PE-A-Frame, is
**at chance on the candidate pool (AUROC 0.53 [0.45, 0.61])**, below PANNs' 0.64 on the same spans; it passed its ratio
screen only because the pool was already near the bar before filtering.

**Amendment 25 (an audio "listener", Qwen3-Omni, as a per-candidate verifier; hard AUROC gate: lower 95 % bound ≥ 0.70
and above PANNs).** This was the last detector question. An audio LLM (Qwen3-Omni-30B-A3B-Instruct) was asked "is the
sound of X present?" about each candidate span. It is a sane listener (yes to 84 % of asked families, 7 % of families
absent from the clip), but on the 280's uncertain candidates it separates real needed sounds from false ones only
weakly: **AUROC 0.661 [0.525, 0.784]** (14 hit / 96 false spans; PANNs 0.574). The pre-written gate (lower bound
≥ 0.70) failed, so nothing was scored on the held-out set and the shipped detector stays.

**Reading.** The shipped stack sits at the cost optimum of every training-free stack tried. Extra recall costs more false
spans than β = 2 allows, and the missing sounds are those masked under speech or music, which closed-set taggers cannot
hear by construction. A detector trained to hear under speech and music is the fix; none with public weights was found.

## 5.9 The gate itself

| visibility judge (DEV gold sounds) | balanced accuracy |
|---|---|
| OWLv2 (object detector) | 0.50 |
| Qwen2.5-VL-7B, 3 votes | 0.61 |
| Qwen3.8-27B, 3 votes (shipped) | 0.62 |

Each vote alone: open naming 0.64, a/b in both orders 0.60 (keeps 94 % of needed sounds), description 0.56 — all within
the ±0.11 interval of the shipped majority (`gate_vote_table.py`). Three further mechanisms (any-stretch rule, OWLv2 and
SAM 3 per-stretch votes) each removed exactly two wrong pictures per needed sound lost — the break-even at β = 2
(amendments 14, 18). The remaining errors are *presence without source*: a visible bell tower silences an off-screen bell;
a visible tank silences a helicopter through the "kind of vehicle" rule (3 such kinship silences on DEV).
**Stability:** re-running the gate on the same frames changes no verdict; moving every frame by 0.5 s changes 6 of 79
(7.6 %, below the 10 % bar set before the run), five of them towards drawing (`gate_shift_compare.py`).

## 5.10 Pictures versus text

With the same gate decisions and the same spans (derived arm, identical on 49/49 DEV clips), the automatic judge
(Gemma-4-31B, trust checks passed) scores pictures and text tags alike: −0.04 [−0.22, +0.14]. The judge reads a text tag
as the label itself, so this measures no advantage for pictures; whether a picture is *understood* is measured by people
(§5.11). The ungated text baseline stays in the tables with its confound stated.

## 5.11 Pictures: author-rater glance test

*Until independent raters exist, this is an **author-rater glance test**: the rater is the author (Adam), blind to the
version but not independent of the project. 3–5 independent hearing raters on the same protocol: **[pending]**.*
54 sounds from 50 never-annotated clips, each picture shown 1.5 s at 384 px, the rater typing what makes the sound,
three versions interleaved and hidden: today's generator (FLUX.1-schnell) 14 of 54 recognised; Qwen-Image-2512 with the
same text 26 (+0.22 [+0.07, +0.37]); with the V3 text 32 (+0.33 [+0.22, +0.46]); 9 of 10 repeated pictures answered the
same. V3 was rejected for one invented object (a thud drawn as a door), and the new generator alone shows one wrong
object (a ringing phone drawn as a desk bell), so neither is adopted as clean. Six automatic picture checkers failed
calibration against the human rater (§5.13). The frozen final setup is Qwen-Image-2512 with the V3.1 text
(`docs/freeze_picture_setup_2026-09-25.md`); it replaces FLUX.1-schnell only if the sealed confirmation sitting
confirms it (rules in Chapter 6 §6.3). **Final frozen setup: [pending — blind confirmation sitting]**.

## 5.12 Timing

A traced bug moved 223 of 442 picture starts earlier than their sound (worst 8.98 s). The fix (a start may never move
before its anchor) raised DEV F1 by +0.101 [+0.022, +0.196]; on TEST the change was inconclusive by its pre-written rule
(−0.019 [−0.127, +0.082]). It is kept as a bug fix; the gain is reported as DEV only. The 8-second picture cap was removed
on principle (a picture should last as long as its sound): 4 of 15 TEST pictures now outlast their sound by > 2 s.

## 5.13 Failed instruments

Each was pre-registered or piloted with a bar written first, and each failed it. They are reported because a reader
should know which measurements this project could not make.

| instrument | what it tried to measure | result | source |
|---|---|---|---|
| model-made references (tags `v2`, `v3`) | what the viewer should have been shown | circular: written by the system's own VLM from the detector's own events, so they reward whichever system repeats the detector | `docs/supervisor_meeting/04_LIMITATIONS.md` |
| independent four-model reference (Qwen2-Audio, CLAP, Idefics3, Phi-3.5) | "nothing beyond the picture" (the gate decision) | agrees with the human tag on 55 of 100 clips — chance on the silence decision | same |
| audio-visual "with vs without sound" reference (Qwen2.5-Omni-7B) | "is anything missing?" | balanced accuracy 49.5 % (bar 67.7 %), κ 0.00, placebo 37 % | `docs/prereg_av_reference.md` |
| gap-closing score (same model, four fixed questions) | does sound change what a viewer can answer | AUROC 0.53 (0.529; bar 0.75); shifted frames move the answers as much as the soundtrack | `docs/prereg_gap_closing.md` |
| six automatic picture checkers (Idefics3 marker, CLIP-L marker, GLM per-picture judge, Gemma verifier, PP-1, pairwise GLM) | is the picture recognisable | all failed calibration against the human rater (e.g. Idefics3 / CLIP-L markers 25/32, bar 26; GLM κ 0.45; Gemma verifier κ 0.28; PP-1 "can't tell" on 502 of 567; pairwise chose the first picture 187 of 190 times) | `docs/LEDGER_2026-09-26.md`; `docs/panel_2026-09-26b_round1.md` |

Two instruments survived: the human-grounded per-sound scorer (one annotator's gold) and the Gemma direct judge (a
secondary, "validated for ranking, not absolute quality"; it cannot tell a picture from a text tag, §5.10).

## 5.14 Disclosures and limitations

- **One annotator** (a 30-clip second pass planned: κ on needed / visible / obvious, weighted κ on importance, median
  onset difference — **[pending]**). The benchmark is a pilot benchmark (v1, 139 clips); the protocol is the
  contribution, the numbers are provisional.
- **DHH helpfulness untested.** No deaf or hard-of-hearing viewer has used the system; the viewer study that would measure
  β was designed and not run (`docs/beta_specification.md` §5). A hearing sound-off pilot, if run, is a proxy on DEV clips
  only, never a DHH study.
- **Modelled viewer cost** is post hoc (§5.2) and β is assumed; hence the interval of §5.5.
- **Picture recognition** is an author-rater glance test until independent raters exist (§5.11).
- **TEST is held out, with disclosed exposure:** ten exposures, all dated in §5.15 with the split overlap; this table
  replaces the 23 Sep table by a rule written before it was rendered. No TEST number has been read since.
- **Split composition.** 49 DEV clips were used in development. TEST holds more clips with nothing to draw than DEV (20
  vs 8). Card class moves about one picture in 49 clips.
- **Scorer disclosures** (appendix): missing importance defaults to 2 (0 of 298 sounds today); clean-clip accuracy counts
  importance-1 needed sounds as needed while the headline treats them as don't-care; the cross-trigger rule uses any gold
  sound; precision is 0 when nothing is shown. Fixes, if any, are checked on DEV only (a TEST re-score would be another
  read).

## 5.15 TEST exposure log (appendix)

*TEST (60 clips, `test_bench`) is held out, with disclosed exposure. Sources: `docs/LEDGER_2026-09-26.md` audit finding 1
(ten exposures), `docs/prereg_v4.md`, `docs/WEEK_PLAN_2026-09-26.md` A2. The ledger counts 5 deliberate reads and 10
exposures. The sources do not use one list of which five are deliberate (`docs/panel_2026-09-26_plan.md` §7 lists the
23 Sep accidental print among the reads; `GOLD_RERUN` §14c calls the whitelist check "read once on TEST"), so the "kind"
column uses each source's own words.*

| # | date | what was read or seen | kind | what could depend on it |
|---|---|---|---|---|
| 1 | 22 Sep | Gold re-run on the 109 benchmark clips (DEV + TEST): v4b4 / v4ab4 tables, oracle gate, judge (`GOLD_RERUN_2026-09-22.md` §4, §9) | planned | the direction of later work ("fix the detector, not the gate", amendments 6–8); the oracle diagnostic of §5.3 |
| 2 | 22 Sep | PSED dry-run grid, a pooled print that included TEST-60 | print | PSED not adopted (a rejection; nothing adopted from it) |
| 3 | 22 Sep | v4b6 tables and amendment 9's viewer cost, computed on tables that include TEST | indirect | the decision to report viewer cost after F1 came out null |
| 4 | 23 Sep | Family whitelist: rule fixed on DEV, read once on TEST (DEV 61 % → TEST 30 %; `GOLD_RERUN` §14c) | planned read of a rule | the whitelist was rejected |
| 5 | 23 Sep | Accidental print of the veto / family-gate rows on TEST (`prereg_v4.md`, "Disclosure: an unplanned look at TEST") | unplanned print | the veto's TEST result was known before the remaining grid cells; amendment 10's second clarification was written after it; amendment 16's TEST threshold (≥ 15 hits) was set from the printed 17 hits |
| 6 | 23 Sep | Amendment 16's PANNs-veto grid endpoints {0.02, 0.05, 0.10}, chosen after reading the 109-clip table (0.20 left out) | indirect | the range of the veto grid (the cell inside it was chosen on DEV) |
| 7 | 23 Sep | The single look, `test_final_v30` (`prereg_v4.md`, "THE SINGLE TEST LOOK") | planned | adoption of both vetoes by the go/no-go written before it |
| 8 | 24 Sep | Second look, timing: `test_monocap_v31` vs `test_final_v30` (`docs/test_second_look.md`) | planned | the onset rule kept as a bug fix (verdict inconclusive), its gain reported as DEV-only; the cap removal was decided before the number |
| 9 | 24–25 Sep | Direct judge (Gemma-4-31B) trust checks on `v4b4` renders of all 139 clips, TEST included (`NIGHT_REPORT_2026-09-25.md` §7) | indirect | the choice of Gemma as the secondary judge |
| 10 | 25–26 Sep | Amendment 21 final table, `test_final_v33` (rule committed before the render; Adam's yes 25 Sep 21:50; scored 26 Sep) | planned | nothing after it: it replaced the 23 Sep table by the committed rule, and no TEST number has been read since |
| S | found 26 Sep | **Split overlap.** Two DEV/TEST definitions coexist: 35 of the scorer's 60 TEST clips are in `split.json`'s DEV-79, where FlexSED's bar 0.8 (amendment 8) and the detector-level rejections (amendment 12, `GOLD_RERUN` §13) were selected; 20 of the scorer's DEV-49 are `split.json` TEST (`prereg_v4.md`, correction after amendment 21) | design, not a read | FlexSED's bar and the detector rejections. Mitigation: the DEV-49 end-to-end grid also selects 0.8; on the clean DEV-49 the bar passes two of its three adoption rules and misses the third by 0.011 (§5.7) |

*After row 10, only derived statistics of the closed table were computed (27 Sep: the β crossings 0.39 / 2.56 of §5.5
from the committed `cost_curve_test_final_v33.json`; no render, no re-score, no decision). No confidence interval on the
TEST crossing is computed. The second-annotator packet shows annotators some TEST clips (not system output or scores),
which is not a read.*

## 5.16 Draft abstract paragraph (for the front matter)

Deaf and hard-of-hearing viewers miss sounds whose source is off screen. We present a training-free pipeline that
detects non-speech sounds, asks a vision-language model whether each source is visible, and shows a generated picture
beside the video only for sounds that are heard but not seen. We test it on a pilot benchmark of 139 clips labelled by
one annotator (a 30-clip second pass: κ **[pending]**). On 60 held-out clips (with disclosed exposure), the pre-registered primary metric, F1
against drawing every detected sound, shows no significant difference (+0.059 [−0.030, +0.144]; the smallest detectable
effect was about 0.13). In the Holm-corrected secondary family, the visibility gate halves wrong pictures per clip
(−0.52 [−0.80, −0.28]) at no detectable recall price (−0.07 [−0.15, 0.00]). Under a modelled viewer cost where a missed
sound costs 4 and a wrong picture β (β = 2 assumed), the gated system is the cheapest option for β between 0.39 and 2.56.
A newer picture generator was recognised 22 points more often in an author-rater glance test. The detector, not the gate,
limits recall. DHH helpfulness is untested.

## References

This chapter cites no literature; the shared reference list is `docs/thesis/references.md`.
