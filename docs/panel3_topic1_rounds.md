# Topic 1 — Sound detection: panel rounds (27 Sept 2026)

Brief: docs/panel3_topic1_*.md · context: docs/panel3_common_context.md

## T1-c — round 1 (skeptical statistician)

**Arithmetic in counts** (detector_round_stage0.json: 224 consequential events, 31 masked, 280 × 10 s = 46.7 min).
Shipped → cascade E: onsets 56 → 104 (+48); events by overlap 113 → 132 (+19); masked 12 → 14 of 31; false spans 208 → 322
(+114). **E buys one onset for ~2.4 false spans; break-even at β = 2 is 2.0** (p > 1/3). E pays only for β < 1.7.
Detector-level cost (4 × missed onsets + 2 × false spans, per clip) moves ~+0.13 — slightly worse than shipped. +48 onsets
vs +19 events → ~29 of the gains are sounds already heard now caught inside the window (timing, not new sounds). Cell F:
+2 onsets, +5 false spans (a correctness fix, moves nothing).

**Q1.** Nothing in reach can be shown to help the viewer this week; every route to the cache recall pays ~2.4:1. E on DEV:
the 11 never-detected misses are masked sounds with BEATs/PANNs 0.000–0.026, E needs BEATs ≥ 0.175 → E can reach some of
the 5 timing misses, not the 11. **Cheapest real experiment: E + a verifier, scored on the E-minus-shipped delta pool of
the 280 clips** (~48 true onsets, ~114 false spans, AudioSet-Strong labels): AUROC with CI in minutes; break-even needs
kept-false/kept-true ≤ 0.84; bar ≤ 0.6 on the fit set, then held-out confirmation. Different from the Qwen2-Audio negative
(ranking weak candidates, not re-detecting). Training: amendment 11 showed instability (27 per-family bars all moved);
a fusion head on 224 events repeats it. Speech-aware FlexSED bars: no.

**Q2 Rule.** Stage 0's pick rule is lexicographic (can never accept a trade that pays at β = 2). Replace, disclosed as
post hoc this round. Fit set = the 280 (not out-of-sample). Pick = lowest detector-level cost C = 4 × (consequential events
with no onset in window) + 2 × false spans, per clip. Confirm on a new AudioSet-Strong **eval-split** download (FlexSED
trained on train split), disjoint from the 280 and gold's 111 ids, N + seed written first, Adam's yes; paired clip bootstrap
of ΔC, 2000 draws, seed 0, upper 95 % CI < 0; one cell only. N from per-clip SD (read-only re-run of Stage 0 dumping
per-clip diffs); default 300. E's margin (+0.13, wrong sign) cannot be rescued at any N. DEV-49 amendment 23 = **sign
check, not a test**: shipped blind 17 hits, cost 176 total; "+2 hits, cost ≤ 3.59" allows ≤ 4 added wrong pictures; a
paired sign test needs ≥ 5 gained / 0 lost; call the outcome "DEV point estimate". Synthetic mixes: recall-vs-SNR curve for
the thesis only.

**Q3 Goal.** The 23 % is stale; final TEST: ours P 0.39, blind 0.25. E's additions are right 30 % by onset, 14 % by event —
at or below 1/3. On TEST the gate removed ~55 % of blind's false pictures and 16 % of its hits; post-gate an average
addition might pay, but E's additions are weaker than average and choosing on the gated gain is forbidden — report beside.
Thesis: two ceilings — (a) recall in caches 30/36 on DEV, 4 unreachable; (b) affordable ceiling at β = 2, exchange rate
~2.4 false spans per gained sound → shipped point is near the cost-optimum of the training-free frontier. No gold set can
resolve a detector change this week (TEST ΔR CI ±0.08 on 43 sounds).

**Q4 Don't.** If amendment 23 fails narrowly: no amendment 24 with looser floors on the same 49 clips; no TEST read for a
detector change; nothing chosen from the oracle-bar table.

**Actions it proposes:** (1) approve a new AudioSet-Strong eval-split download (default 300, seed first); (2) adopt the
detector-level cost as the pick rule (post hoc this round); (3) run the verifier-on-E-delta test before/instead of waiting
on amendment 23.

## T1-b — round 1 (pragmatic systems engineer)

Read-only scratchpad script over local gold caches: counts **33** needed DEV sounds with all caches (brief says 36; 3
probably lack a cache).

**Q1.** **Rank 1 — cascade E, read correctly, ship a tighter variant.** On the 280: onset-recall 25.0 → 46.4 % but recall
50.4 → 58.9 % — over half the gain is earlier starts on events already found. Price ≈ 114 extra false spans for ≈ 48
onset hits (≈ 19 events) → added detections ~30 % right by onset, ~14 % by overlap vs 33 % break-even at β = 2.
Prediction: E fails amendment 23's cost rule on DEV, narrowly. On DEV caches E reaches 27 of 33 (shipped 19): 6 by tier 3
(Laughter .208/F .919, Gunshot .211/F .549, Explosion .242/F .557, Bird .211/F .708, Telephone .332/F .904, Cricket
.318/F .918) + 2 by tier 2 at bar 0.5 (Vehicle B .10, Crowd/applause P .075). **All six tier-3 adds are corroborated by
FlexSED ≥ 0.3; none needs the PANNs ≥ 0.05 branch** — that branch is permissive, likeliest source of the 114 false spans.
**Variant E-F:** tier 3 with FlexSED only (`BEATS_LOWBAND_CORROB = (0.3, 1.01, 1.0)`, a-priori grid {0.3, 0.5}) = the
time-aligned form of the FlexSED veto τ 0.3. One CELLS entry, CPU minutes. **Commit before E's DEV render is read.**
**Rank 2 — audio-LLM verifier on the FlexSED-only band, one day.** Per candidate (FlexSED-only spans at 0.5 failing tier
2), cut [start − 1, end + 1] s, yes/no "do you hear {family}?", P(yes) from first-token logits, bar at matched false/min.
Qwen2.5-Omni-7B (loads under transformers 5.16); Qwen2-Audio fallback. Target on DEV: the 4 masked FlexSED-only sounds
(Hammer 13.7 s F .70, Whistle .585, Footsteps .658, Crowd/protest .882; BEATs/PANNs ≤ 0.032). Pool on the 280 thin
(~30–50 candidates) → needs held-out download pooled. Go/no-go: AUROC ≥ 0.75 on ≥ 60 candidates.
**Not this week:** text-queried separation (new env; Demucs negatives; FlexSED already is the text-queried view); training
(fusion head eats the only pick set; a head on AudioSet-Strong train re-derives PSED/FlexSED); ensembles (on ledger).

**Q2 Rule — three levels, all must pass; report as "detector upgrade beside the frozen system".**
1. Pick on the 280 (cached, CPU): a-priori cells; highest onset-recall at false/min ≤ shipped 4.46 + paired bootstrap CI;
   masked recall reported, never used (±2 events = noise).
2. Confirm on a new held-out AudioSet-Strong set: eval_strong.tsv has 16,997 segments, ~470 used; N = 400 new seed (≠ 11),
   disjoint from calib + slice B, via audioset_calib.py (~330 survive yt-dlp). Rule: Δ onset-recall lower CI > 0 **and**
   false/min ≤ shipped **and** masked recall not lower. **Code gaps to fix first:** `audioset_detector_eval.use_set()`
   knows only calib/else-slice-B (`--set heldout` silently scores slice B); `FLEX` hard-coded to flexsed_calib.
3. DEV-49 blind render = sanity, not selector (am. 23 rule).
No synthetic mixes.

**Q3 Goal.** Right goal = more sounds at ≥ 33 % precision per added span. DEV residual: 6 where two detectors agree
(BEATs weak + FlexSED ≥ 0.3) — the only bucket worth a build; 4 FlexSED-only — verifier or nothing; 3 below every detector
(Hammer 8.1 s, Bird .435, Civil-defense siren .42). Chasing buckets 2–3 at 14–30 % would reverse the TEST wins. Thesis:
(i) ceiling = speech/music masking of closed-set taggers (11/21 never detected; 0/31 masked by FlexSED at 0.8);
(ii) recall exists in the text-queried detector (27/33 at 0.5) but can't be separated from its false alarms — every
corroborator shares BEATs' deafness; (iii) with the annotator's list ΔF1 significant (+0.067), cost 2.63 → 1.23 → the
detector is the binding constraint; (iv) fix = a detector trained to hear under speech; no public weights as of 27 Sept.

**Q4 Don't.** Start anything needing a new env or a FlexSED re-cache (Audio Flamingo 3, AudioSep, new prompts/vocab,
fine-tuned head); SAM-Audio ate two days. Existing caches + at most one small model pass. No bar from DEV scores; no TEST.

**Actions it proposes:** (1) commit E-F as a cell now; (2) approve held-out download (N = 400, new seed); (3) yes/no on a
one-day verifier smoke.

## T1-a — round 1 (SED SOTA researcher)

Disclosure: three diagnostic reads of the DEV-49 caches (same kind as F1's disclosed dry sweep), used only to size pools
and record one forecast; no threshold chosen. Counts 33 needed DEV sounds (definition difference vs 36).

**Where reachable recall sits (DEV-49).** 22/33 reachable at shipped bars by oracle. **8** more have FlexSED in [0.5, 0.8)
with BEATs < 0.35 (Vehicle/horn 0.71, Bird forest 0.71, Hammer 13.7 s 0.70, Footsteps 0.66, Explosion 2.8 s 0.62, Whistle
0.59, Explosion 5.6 s 0.56, Gunshot 0.55); only 4 have PANNs ≥ 0.05 → PANNs veto kills the other 4 at any FlexSED bar.
**3** below everything (Hammer 8.1 s, Bird rainforest_21, civil-defence siren). Weak-BEATs band with FlexSED < 0.5 holds
**0** → cascade route and lower-FlexSED route are the same 8 sounds from two sides.

**Q1 ranked.**
- **Rank 1 — PE-A-Frame (Meta, Dec 2025) as independent corroborator for FlexSED-only spans.** facebook/pe-a-frame-large:
  Apache-2.0, 1.4 B, not gated, 48 kHz, text-conditioned score every 40 ms (same 25 fps as FlexSED cache), in transformers
  as `PeAudioFrameLevelModel`; paper PE-AV arXiv 2512.19687. Missed by the 27 Sept web check. Every corroborator tried is
  either a closed-set AudioSet tagger (deaf under speech/music) or shares FlexSED's Dasheng/CLAP lineage; PE-A-Frame is
  text-queried but a different encoder (PE-AV, ~100 M audio-video pairs) → genuinely new evidence. Cost: < 1 GPU-hour to
  cache 139 + 280 clips, half a day engineering (separate env if transformers 5.16 lacks it). Recommended: PE-A-Frame ≥ θ
  **replaces** the PANNs veto on FlexSED-only spans (ceiling 8 on DEV) rather than adds to it (ceiling 4). Price: DEV
  FlexSED-only pool in [0.5, 0.8) = 219 spans, 7 true (5 sounds); after PANNs veto 54 spans, 5 true (~9 %). To pay at β = 2
  keeping 4/5 true it must reject ~85 % of 49 false spans. Decidable only on held-out. Also read its score at the 3
  unreachable onsets (thesis line).
- **Rank 2 — cascade E: let it finish, adopt nothing else from it.** Forecast on record: on DEV E promotes ≈ 247 weak BEATs
  spans; 16 cover a needed sound but only **2 are new** (Cricket rainforest_7629, Bird birds_forest); stricter FlexSED
  corroboration (≥ 0.5: 162 spans; ≥ 0.7: 114) — same 2 new. If the render shows ≥ 2 new blind hits at cost ≤ 3.59 it
  passes and the forecast was wrong.
- **Rank 3 — window-level audio-LLM yes/no, logit scoring (2 days, riskier):** Qwen3-Omni-30B-A3B-Instruct (Apache-2.0,
  ~79 GB, H200/A100-80); MiDashengLM-7B (Dasheng encoder = correlated with FlexSED); Audio Flamingo 3 (non-commercial).
  4-s window, "Is there a {family} sound? yes/no", p(yes) first token, θ on the 280. ~1 GPU-hour; yes-bias warning stands.
- **Not this week:** SAM-Audio separation (env broke 20 Sept; separation AUROC +0.003; needs null-prompt control; 3+ days);
  fine-tuning (31 masked events); ≤ 5-param logistic late fusion allowed as calibration but ≤ +2–4 pts, invisible on DEV;
  temporal context / closed-set ensembles don't address masking; 3-prompt FlexSED failed its bar. Boundary-aware SED,
  COSED, OW-SED/WOOT (2605.03934), DASM: no usable checkpoints — confirmed.

**Q2 Rule — two stages, TEST never read.** Pre-check on the 280 caches: score the corroborator on FlexSED-only boxes at bar
0.5; go iff AUROC ≥ 0.80 and ≥ PANNs' AUROC + 0.05; θ = loosest grid value keeping stack false/min ≤ 4.46. **Stage A —
new held-out AudioSet-Strong eval-split, N and seed before download, only clips with Speech or Music ≥ 50 % and ≥ 1
drawable non-speech event labelled, N = 500 (~150–300 masked events):** primary consequential recall (overlap) at fixed θ;
pass iff Δrecall lower 95 % CI > 0 **and** false/min ≤ shipped **and** masked recall not lower. Onset-recall secondary
(the 280's 25 % is a span-start artefact). **Stage B — DEV-49 blind arm, amendment 23 as written.** Synthetic mixes only as
diagnostic. No TEST.

**Q3 Goal.** Right only if added detections are > 1/3 right; every cell so far adds at 9–25 % → report the β at which a
cell crosses shipped. Thesis ceiling in three tiers: reachable at shipped bars; reachable only where FlexSED alone hears
(masking, not vocabulary — sounds under Speech 0.6–0.8 / Music 0.5); unreachable by any public model (3/33). Sentence: *a
masked sound can only be corroborated by a second model asked about it by name, or by a listener; closed-set taggers
cannot, by construction.* The 280 hold only 31 masked events — too few, hence held-out sized on masked events.

**Q4 Don't.** Adopt cell E on the 280's onset-recall 25 → 46 % (consequential recall only 50 → 59 % for +55 % false spans;
timing artefact; forecast ≤ 2 new DEV sounds). No bar from DEV oracle counts, no TEST read, no fine-tune.

**Actions it proposes:** (1) cache PE-A-Frame on 139 + 280 clips (≤ 1 GPU-h) + Step-0 AUROC pre-check; (2) approve held-out
download (masked filter, N = 500); (3) let E finish.

