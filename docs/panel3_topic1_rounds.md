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

---
# Round 2

## T1-a — round 2

(One more DEV-49 cache read, disclosed: span lists for the six disputed sounds. No E render output read.)

**(1) Counts reconciled.** T1-b's 27/33 = oracle frame-peak presence in window; T1-a's 2 = span level after AED_MIN_DUR
0.5 s, FlexSED veto, twin rule. Of the six tier-3 adds: Cricket (weak span 0.00–0.75 at 0.25, F 0.92) and Bird
birds_forest (weak 2.0–2.75 at 0.22, onset 1.3, F 0.71) are real; Telephone is already a shipped hit; Laughter 8.1 s,
Gunshot 0.0 s, Explosion 2.8/5.6 s have no BEATs span ≥ 0.175 at the onset (single 0.25-s windows die at min_dur) —
reachable only as FlexSED-only spans at bar 0.5 (tier 2; Explosion 2.8 passes PANNs 0.12, Laughter fails 0.001). **On DEV:
E = 2 new by tier 3, ≈ 2–4 more by tier 2, none by the PANNs branch.** Forecast stands: E fails amendment 23 on cost.

**(2) E-F: commit now**, before E's render is read: `E-F = D + tier 3 (0.3, 1.01, 1.0)`, a-priori grid {0.3, 0.5}. DEV
read: promoted spans 247 → ≈ 160, same 2 tier-3 adds; PANNs branch buys nothing on DEV. Its 280 false/min decides it; do
not expect 4.46.

**(3) Pick rule: converges on T1-c's cost**, post hoc this round: C = 4 × missed + 2 × false spans per clip on the 280;
pick = lowest C; confirm on held-out, ΔC upper 95 % CI < 0. Amendment: miss term = consequential events with **no
same-family span overlapping** (not "no onset in window") — repeated short events make the onset jump an artefact; report
onset-recall + onset MAE beside. E loses under either (+0.13 or ≈ +0.5 per clip). Drops the lexicographic rule.

**(4) One corroborator: PE-A-Frame** if `from transformers import PeAudioFrameLevelModel` works on the login node
(1-minute check); else Qwen2.5-Omni-7B logit yes/no, no new env. Pools: E-delta on the 280 (~48 true / ~114 false) and
FlexSED-only [0.5, 0.8). **Go iff at the point keeping ≥ 80 % of true items, kept-false / kept-true ≤ 0.84** (p > 1/3),
bootstrap CI written first; AUROC reported, not the gate. Then corroborator-replaces-PANNs-veto = one more CELLS entry.

**(5) Held-out spec.** AudioSet-Strong eval split, new seed ≠ 11, disjoint from 280 + slice B + gold's 111; **N = 400
sampled (~330 survive), half from clips with Speech/Music ≥ 50 % and a drawable consequential event** (≥ 100 masked events);
N + seed before download; per-clip SD states power, doesn't raise N. Fix use_set() heldout branch + FLEX path first.
**Primary ΔC (overlap miss) upper CI < 0**, 2000 draws seed 0, one cell; secondary recall, onset-recall, masked recall,
false/min. DEV-49 = amendment 23, "DEV point estimate". No TEST.

**Agreed:** E fails on the 280 under any cost; E-F committed now; cost C; PE-A-Frame first, gate 0.84; held-out N = 400
stratified, ΔC primary; no synthetic mixes for selection. **Open:** PE-A-Frame import under 5.16; Adam's download yes;
overlap vs onset miss term.

## T1-b — round 2

**(1) Concedes to T1-a.** Named re-run: six tier-3 adds were Cricket, Laughter, Gunshot, Explosion 5.6 s, Bird
birds_forest, Siren mv_tornado (B 0.34998 — float16 rounding at the bar, not real). Telephone was a transcription error
(shipped-reachable). Only Cricket and Bird are new by any bar; Laughter is killed by the shipped stack (PANNs clip peak .001;
also weak-twin bug); Gunshot/Explosion are in T1-a's FlexSED [0.5, 0.8) bucket. E's real DEV gain ≤ 5 needed sounds (2
new-by-bar) at ≈ 247 promoted spans (≈ 4.7/clip). **E fails amendment 23 by a wide margin**; "27/33" was oracle reach.

**(2) One corroborator: PE-A-Frame, not the Omni verifier** — same shape as the FlexSED cache (plugs into near/veto code),
cached once, no prompt, no yes-bias, threshold by AUROC on exhaustive AudioSet labels; Omni needs prompt, logit plumbing,
thin pool, yes-bias control. Risk: local transformers 4.56.2 lacks it; cluster 5.16 may have it. Go/no-go (2 h): import in
msproj; if absent, clone msproj + upgrade transformers only; if one clip doesn't score in 2 h, drop and report. Then T1-a's
pre-check (AUROC ≥ 0.80, ≥ PANNs + 0.05); it replaces the PANNs veto on FlexSED-only spans as one cell.

**(3) E-F withdrawn — no CELLS entry.** Stricter FlexSED keeps ~2+ false promotions/clip for 2 sounds; at ≥ 0.8 tier 3
degenerates to cell F. Only defensible cascade = E + corroborator on the E-delta pool = (2).

**(4) Adopts cost C** (post hoc, disclosed; lexicographic beside). C on overlap-matched events primary, C on onsets
secondary; **a pick must lower both**. Paired bootstrap 2000, seed 0.

**(5) Held-out spec.** eval_strong.tsv (16,997 segments), disjoint from the 280 (+40 missing), slice B (111 + 39 missing),
every gold AudioSet id; T1-a's filter (Speech/Music ≥ 50 % + ≥ 1 drawable non-speech event); **N = 500, seed 23**, written
before download (~400 survive). Primary ΔC (overlap) upper 95 % CI < 0 for the one picked cell + false/min ≤ shipped on the
same clips; masked recall reported. Code first: (a) use_set() heldout branch; (b) FLEX path hard-coded; (c) a ~20-line
selector implementing the filter; (d) cache paths keyed off use_set. Commit all four, then download.

**Agreed:** E fails on cost, ≤ 2 new-by-bar on DEV; no E-F; PE-A-Frame the one corroborator, 2-h load gate + 280 AUROC
pre-check; cost C; held-out N = 500 masked-filtered, seed 23; DEV blind = sign check; no TEST, no oracle-DEV bars, no new
env beyond a transformers upgrade in a cloned env. **Open:** PeAudioFrameLevelModel under 5.16; C vs recall wording (takes
C-overlap); Adam's download yes.

## T1-c — round 2

1. **Verifier: PE-A-Frame** (audio-LLM only if it fails). The 280's FlexSED-only-at-0.5 pool holds ~7 true in ~38 spans →
   AUROC SE ≈ 0.15, T1-a's pre-check undecidable; E-delta pool ~48 true / ~114 false (SE ≈ 0.045). One candidate
   definition: every span cell E raises that shipped does not show, on the 280 (expect ~55 true / ~145 false; count
   positives before writing the bar). Screen: AUROC lower 95 % CI ≥ 0.70 and point ≥ PANNs' (drop +0.05). θ = argmin C
   on the 280; adoption only by held-out. Bar must be a cost, not an AUROC (required rejection is pool-dependent).
2. **E-F: yes, one CELLS entry before E's DEV render is read** (disclose its motivation is a DEV cache read). Add **E-AND**
   a priori (promote iff FlexSED ≥ 0.3 AND PANNs ≥ 0.05): tier 3 launders FlexSED-only spans past the PANNs veto. Grid
   {E-F 0.3, E-F 0.5, E-AND}, CPU minutes, judged by (3).
3. **Pick rule C** (lexicographic is dominated). pick = min C = 4 × unmatched consequential events + 2 × false spans per
   clip on the 280; label "strict pass" (also ≤ shipped false/min) or "trade pass". C by ≥ 0.5 s overlap primary, by onset
   window secondary; **if the two disagree in sign, no pick.** E ≈ 6 false per gained event (overlap) / 2.4 (onset).
4. **Held-out:** eval split, N = 500 (~330 survive), seed ≠ 11, disjoint, Adam's yes; **stratified 250 masked + 250
   unfiltered random** (masked-only can't see false alarms on quiet clips). Primary ΔC upper 95 % CI < 0 over all clips;
   secondary Δrecall (overlap) in masked stratum, Δfalse/min per stratum, masked recall not lower. Code gaps fixed first.
5. **Reconcile:** of T1-b's six, three have FlexSED ≥ 0.8 (Laughter, Telephone, Cricket) — lost to twin absorption or PANNs
   veto, recovered by tier 3 bypassing the veto; three are genuine weak-BEATs (Gunshot, Explosion, Bird). E's plausible DEV
   yield 2–3 new + up to 3 rescued. Only the render is evidence, as a point estimate.

**Agreed:** E as-is not an upgrade; PE-A-Frame the one corroborator; E-F (+ E-AND) before E render; stratified held-out
N = 500; no TEST, no DEV-chosen bar, no fine-tune. **Open:** C vs lexicographic (others now accept C); verifier pool
(E-delta vs FlexSED-only; union proposed); PE-A-Frame availability.

*(Coordinator check 27 Sept: cluster env msproj has transformers 5.16.1 and `from transformers import PeAudioFrameLevelModel`
succeeds. E-F: T1-a and T1-c yes, T1-b withdrew — not committed yet; E DEV render still unread and not yet run.)*

---
# Round 3 (final)

## T1-a — round 3

(a) **E-F: yes, commit now** — E-F 0.3, E-F 0.5, E-AND as three CELLS entries, motivation disclosed as a DEV cache read,
judged only by C; expectation written: all three lose under C-overlap on the 280. No further cascade cells after these.
(b) **Pool = T1-c's union** (every span cell E raises that shipped does not show, on the 280; ~55 true / ~145 false; count
positives before the bar). Screen: AUROC lower 95 % CI ≥ 0.70 and point ≥ PANNs'. **Bar = cost:** θ = argmin C on the 280
for the stack "PE-A-Frame ≥ θ replaces the PANNs veto on FlexSED-only spans and the PANNs branch of tier 3"; adoption only by
held-out ΔC. Drops the 0.84 gate. (c) **Held-out 250 masked + 250 random**, N = 500, seed 23, eval split, disjoint; ~400
survive. (d) C by ≥ 0.5 s overlap primary, onset secondary, no pick if signs disagree; "strict pass" / "trade pass".

Plan (GPU / Adam):
1. Today, CPU, Adam 0: amendment 24 (C rule, three cascade cells, union pool, held-out spec + seed 23, PE-A-Frame cell)
   committed **before E's DEV render is read**; re-run detector_round_stage0.py with C.
2. Day 1, ≤ 1 GPU-h, Adam 0: PE-A-Frame cache on 139 gold + 280 calib, 215 queries, 25 fps, flexsed_cache npz shape; 2-h
   load gate.
3. Day 1–2, CPU: union-pool screen + θ on the 280. Fail → stop, report negative, thesis ceiling text.
4. Day 2, Adam yes + ~10 min: code fixes first (use_set heldout, FLEX path, stratified selector, cache keys), then download
   (~3 h wall) + caches ≈ 2 GPU-h.
5. Day 3, CPU: held-out ΔC upper 95 % CI < 0, one cell; stratum secondaries; sign check.
6. Day 3–4, only if 5 passes: DEV-49 render of that cell, both arms, H200, 4–6 GPU-h, amendment 23 rule, "DEV point
   estimate". E's own render: read when it lands, reported, selects nothing.
7. Day 4–6, Adam ~2 h reading: thesis sections either way (three-tier ceiling 22/33 · 8 · 3; cost frontier ≈ 2.4–6 false
   spans per gained sound; 280 vs held-out; PE-A-Frame result); TEST untouched.

**Signs. Dissents** on T1-b's withdrawal of E-F and on "false/min ≤ shipped" as more than a label.

## T1-b — round 3

(a) **E-F: yes, signs** (withdrawal was a forecast, not a process objection). Grid {E-F 0.3, E-F 0.5, E-AND} committed before
E's render runs, disclosure as T1-c. Expectation: none reaches ≤ 4.46 false/min; E-AND closes tier 3's veto bypass (on DEV
keeps Bird P .065, loses Cricket P .001). (b) Union pool + cost bar; screen AUROC lower CI ≥ 0.70 and ≥ PANNs'; single
global θ = argmin C (no per-family bars). One cell **E-PE**: PE-A-Frame ≥ θ within 1 s replaces the PANNs veto on
FlexSED-only spans and the PANNs branch of tier 3. Cache mirrors flexsed_run.py (215 queries from depictable_vocab.json,
25 fps, npz [n_labels, T] + fps), 48 kHz input, env msproj. (c) Held-out 250 masked + 250 random, seed 23, disjoint (~330
survive); primary ΔC (overlap) upper CI < 0; false/min ≤ shipped on same clips; masked recall reported. (d) Code order,
all before download: 1 use_set("heldout") (SLICE/VIDEOS/WIN/OUT); 2 FLEX/PANNS keyed off the set; 3 selector in
audioset_calib.py (strata, disjointness, seed, `--dry` prints ids); 4 detector_round_stage0.py per-clip C rows, `--set`,
new CELLS (E-F ×2, E-AND, E-PE placeholder).

Plan (GPU / Adam): 1 CELLS + C rows, Stage 0 re-run (CPU 10 min). 2 code gaps + selector dry run (CPU 2 h / Adam's yes).
3 download N = 500 (login 1–3 h). 4 PE-A-Frame cache 139 + 280 (< 1 GPU-h). 5 pre-check on union pool, θ, E-PE. 6 held-out
caches (BEATs + PANNs 0.5 GPU-h, FlexSED ×4 ~3 GPU-h wall, PE 0.5). 7 Stage 0 with C → one pick. 8 held-out ΔC. 9 DEV-49
render of the pick, blind arm point estimate (H200 3–4 h); E's render after it as the amendment-23 negative. 10 thesis
ceiling paragraph + round as upgrade or negative. Adam: two approvals.

**Signs the whole plan.** Reservation: H200 goes to the picked cell's render before E's; if no pick, E's render is the only
DEV number, reported as the forecast negative.

## T1-c — round 3

Settled: (a) cost C by all; a pick must lower both C-overlap (primary) and C-onset. (b) **Dissents on T1-b's held-out
"false/min ≤ shipped" gate** (re-imports the lexicographic rule) — report false/min and the β at which the cell crosses
shipped instead. (c) E-F kept, **on the 280 only**; no DEV render for any cascade cell unless it lowers both C's; E-AND
alongside. (d) Held-out N = 500, seed 23, half masked / half random; primary ΔC over all clips. (e) Verifier PE-A-Frame
only; screen on the E-delta pool; **one cell in total reaches held-out** (multiplicity).

Plan (GPU / Adam): 1 commit amendment 24 (Adam read + yes, 5 min). 2 code before any number (use_set heldout, FLEX path,
masked selector, cache paths). 3 Adam's yes → download N = 500, caches BEATs/FlexSED/PANNs (~1 GPU-h). 4 PE-A-Frame 2-h
load gate, cache 139 + 280 + held-out (≤ 2 GPU-h). 5 screen PE on the E-delta pool (CPU). 6 score all a-priori cells on the
280: shipped, F, E, E-F 0.3/0.5, E-AND, D+PE, E+PE. 7 pick min C → one held-out test; fail = round ends, negative written.
8 pass only: DEV-49 render both arms, blind sign check (4–6 GPU-h). 9 E's amendment-23 render last on a free card.

**Amendment 24 text (commit before any number):** "Detector round 2. Cells fixed a priori: shipped; F (twin fix); E; E-F
(0.3, 1.01, 1.0) and (0.5, 1.01, 1.0); E-AND (promote iff FlexSED ≥ 0.3 and PANNs ≥ 0.05); D+PE and E+PE with θ from the
screen. Screen: on the 280's E-delta pool (spans E raises that shipped does not show; positives counted and written before
scoring), at the loosest grid θ keeping ≥ 80 % of true items, kept-false/kept-true ≤ 0.84 with a 2000-draw bootstrap CI
reported; AUROC reported, not a gate; fail → no PE cell. Pick (fit set = the 280, not out-of-sample): per clip C = 4 ×
consequential events with no same-family span overlapping ≥ 0.5 s + 2 × false spans; C-onset the same with the [−0.5,
+1.0] s onset window; the pick is the cell with the lowest C-overlap that also lowers C-onset below shipped's; none → round
ends. Held-out: AudioSet-Strong eval split, N = 500, seed 23, 250 with Speech/Music ≥ 50 % and ≥ 1 drawable non-speech
event + 250 unfiltered, disjoint from the 280, slice B and all gold ids, downloaded after this text is committed and Adam's
yes; the one picked cell passes iff paired clip bootstrap ΔC-overlap (cell − shipped), 2000 draws, seed 0, has upper 95 %
CI < 0; reported beside: ΔC-onset, Δrecall, Δonset-recall, Δmasked recall, Δfalse/min per stratum, the β at which the cell
crosses shipped. Pass → DEV-49 render both arms, amendment-23 rule as a sign check, reported as 'detector upgrade beside the
frozen system, DEV point estimate, not confirmed on TEST'; the amendment-21 table stands. No TEST number; no bar from DEV
scores; the rule is not revised after a number is seen."

**Signs.** Dissents: (1) any false/min gate on held-out; (2) calling a DEV +2-hit outcome an "improvement"; (3) a
masked-only held-out set. **Forecast:** no cell lowers both C's on the 280 except possibly D+PE; most likely product is a
precise negative (≈ 2.4:1 by onset, ≈ 6:1 by event) for the thesis ceiling section.

---
# Topic 1 — outcome (coordinator)

**Signed by all three.** E as-is is not an upgrade. One amendment (24, T1-c's text) to commit before any number: cells
shipped / F / E / E-F ×2 / E-AND / D+PE / E+PE on the 280, pick by cost C (overlap primary, must also lower onset-C), one
cell to a new held-out AudioSet-Strong set (N = 500, seed 23, half masked / half random), pass iff ΔC upper CI < 0, then a
DEV render as a sign check. PE-A-Frame is the one new model (imports on the cluster). Only split: T1-b wanted a held-out
"false/min ≤ shipped" gate, T1-c against (T1-a: label only) — coordinator takes the majority: report false/min, no gate.
Needs Adam: yes to the download (~500 clips, AudioSet-Strong eval split, YouTube via yt-dlp as before).

