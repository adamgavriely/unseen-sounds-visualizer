# Topic 3 — Evaluation: panel rounds (27 Sept 2026)

Brief: docs/panel3_topic3_*.md · context: docs/panel3_common_context.md

## T3-c — round 1 (thesis examiner, related work)

**Q1 Framework.** Four-stage cascade + one end-to-end utility; say which stage limits the rest (make ch5 §5.6 the spine).
Detection: event F1 with onset collar (DCASE/sed_eval) + operating curve — cite PSDS (Bilen et al. 2020); the β curve is the
same idea in cost units. Gate: balanced accuracy on gold visible ticks (0.62 vs 0.50); nearest field = sound-source
localisation with negative audio (Juanola 2025; "Localizing Visual Sounds the Hard Way", CVPR 2021) — those assume the source
is visible; ours inverts that (a gap to claim). Timing: median lateness + share within window, own row. Picture: human glance
recognition (Clotho-Eval/FENSE/CLAIR-A precedent); Gemma = CLAIR-A-style judge, "validated for ranking, not absolute quality",
blind spot text tags. End-to-end: cost utility (4×miss + β×wrong) as a curve with crossings vs blind and silence + off-screen
subgroup. Headline: keep F1 as pre-registered primary (null + MDE); examiner's headline = restraint: gate halves wrong
pictures (−0.52/clip), precision +0.137, Holm-sig on held-out 60, at non-sig recall price (−0.07).

**Q2 Human eval in 6 days (priority order).** (1) Second annotator, 30 clips (packet ready): κ per field (family, visible,
obvious, importance ≥ 2) + re-score ours/blind under their ticks ("gold-robustness" row); κ(visible) < 0.4 is a finding.
(2) 3–5 independent hearing raters, same 54 pictures, same 1.5 s protocol (Adam's own rate isn't blind); one evening.
(3) Hearing-proxy comprehension pilot, n = 8–12, sound off, within-subject silence/blind/ours, Latin square over 24 DEV-49
clips (8 off-screen / 8 mixed-on-screen / 8 nothing); after each clip: list unseen sounds (recall + false beliefs) and the
3-point helped/neither/hurt; ~25 min; paired Wilcoxon + mixed-effects logistic; n = 10 gives > 80 % power for 20 → 60 %
naming; β fit exploratory. Label as proxy pilot (muted hearing ≠ deaf, Frontiers 2020 EEG); 2–3 DHH qualitative if
reachable; check BIU ethics, else "informal pilot".

**Q3 Presentation.** Keep pre-registered order; add "three claims and how strong each is": confirmatory (precision, wrong
pictures, clean-clip acc.), post-hoc with pre-declared subgroup (viewer cost, off-screen −3.08), descriptive (gate acc.,
AudioSet-280, timing). State MDE 0.13 and n 320–570 in the same sentence as the null, oracle diagnostic next. Relabel:
TEST → "held-out (60; exposed 10 times, dated log)"; blind recognition → "author-rater glance test" until independent
raters; viewer cost → "post-hoc cost (weights predate gold)"; precision → "TEST-only (not on DEV)". One-annotator statement
into abstract + limitations. Add exposure log, DEV-49 FlexSED re-check beside overlap disclosure, per-error-type table with
examples, "what the judge cannot see" box, proposal RQs → sections map, note that the proposal's audio-to-image baseline
was dropped and why. Do not soften negative rows.

**Q4 Benchmark as contribution.** Yes, probably the most citable part: no public benchmark labels source-on-screen + viewer
need. Ship: clip IDs + timestamps + per-sound table (ontology id, on/offset, visible, obvious, importance); annotation
guideline incl. "rate as if the screen were black"; κ on a subset; scorer package (window, family match, visible-picture =
false alarm, bootstrap, Holm, README reproducing thesis table); reference system outputs (ours/blind/silence/text);
data card (category counts, licence); version tag + task name "off-screen sound visualisation". Pilot gives first
viewer-cost ↔ helped/hurt correlation (validity check, as FENSE/CLAIR-A did).

**Q5 Don't.** Read TEST again for anything (cascade, pilot, "final confirmation"). Second: don't call the hearing-proxy
pilot a DHH user study.

Sources: PSDS arXiv 1910.08440; DCASE 2020 metric insights arXiv 2010.13648; CLAIR-A arXiv 2409.12962; Hard Way arXiv
2104.02691; Juanola 2025 (sightsound.org); Beyond Subtitles ASSETS 2022; SoundWatch ASSETS 2020; DHH preferences CHI 2019;
Frontiers Integr. Neurosci. 2020 EEG captions.

## T3-a — round 1 (HCI / accessibility user-study methodologist)

**Q1 Framework.** One question: does the viewer learn that a sound happened, what, roughly when — without distraction when
nothing is missing? Detection: per-sound P/R/F1. Gate: balanced accuracy on the 79 ticked sounds, "seen silenced" and
"needed kept" separately. Timing: median lateness + window + 5-s sensitivity row; split lateness by importance 2 vs 3
(SoundWatch: tolerance depends on urgency). Picture: blind 1.5-s glance recognition — keep. **Headline: the β cost curve
with landmarks 0.35 / 1.37 / 4.30, as an interval claim (beta_specification §6).** F1 is a component metric (weights miss =
wrong picture; no DHH study supports that — SoundWatch, Beyond Subtitles, Caption Royale report asymmetry). Missing: a
human-measured β; say so.

**Q2 Human eval, three tiers.**
(a) Do: second annotator, 30 stratified clips, κ on needed + visible with bootstrap CI. Don't redesign.
(b) Labelled pilot **only if the supervisor says an internal hearing pilot needs no ethics filing** (ask): N = 10–12
hearing lab members, sound off, within-subject, ~25 min. Stimuli: DEV clips with ≥ 1 needed sound (~25–30, ≤ 28 s).
Conditions gated / blind / none, Latin square. Primary: 4-option forced choice "which sound happened off screen?" (true
family, same-scene distractor, other distractor, none). Secondary: 1–5 distraction, helped/neither/hurt. Mixed logistic
(participant, clip random). Power: ≈ 0.55 vs 0.30 recognition → ~75–80 paired observations; ~112/condition, ~75 after
design effect 1.5 — enough for gated-vs-none, **not** gated-vs-blind. Label "pilot; instrument validation for a future
DHH study". Assumption: in-video recognition ≈ glance recognition.
(c) Not feasible: any DHH participation (weeks of recruitment + ethics; comparable studies N = 8–39 took months).
Proxy limits: hearing sound-off viewers validate picture + timing; they cannot price miss vs wrong picture (β).
Conflict: β-spec §5 pre-registers TEST renders as stimuli; TEST is closed → pilot on DEV only.

**Q3 Presentation.** Lead with cost curve + interval sentence, then Holm-significant components, then null ΔF1 with
reason (oracle vocab still +0.03; MDE 0.13; ~335–570 clips). Keep amendment 9's "cost declared post hoc" and print the
pre-gold COST_MISS=4, COST_FA=2 provenance. DEV's disagreement on silence in the same table. One annotator in abstract,
limitations, gold description, with κ from the second pass + intra-rater 0.60. Dated 10-row exposure log in appendix.
Relabel "viewer cost" → "modelled viewer cost (β = 2 assumed)"; "helps DHH viewers" → "reduces modelled viewer cost; DHH
helpfulness untested" (in abstract). Add a one-page "what a DHH study would settle" section.

**Q4 Benchmark.** Yes — no released set pairs per-sound onset with visibility, obviousness, importance. Needs: two-annotator
κ; annotation tool (build_tool.py) + frozen tick definitions; clip IDs, timestamps, licence (ATTRIBUTIONS.md); scorer +
cost_curve + β landmarks with frozen split; exposure log; the pre-registered DHH study design. Call it v1 (139 clips); v2
needs ≥ 300 clips, ≥ 2 annotators, DHH-measured β.

**Q5 Don't.** Run a quick un-pre-registered hearing sound-off study and cite it as evidence the system helps deaf viewers
(underpowered for gated vs blind, wrong population for β, a second null, eats the time for annotator + write-up).

**Actions it proposes:** ask supervisor about ethics for an internal hearing pilot; skip the pilot unless "no filing" by
Monday; put "DHH helpfulness untested" in the abstract.

## T3-b — round 1 (ML evaluation, metrics, benchmark design)

**Q1 Framework.** Four components + one end-to-end headline. Detection: onset-collar event F1 (sed_eval) + PSDS on the
280 AudioSet-Strong clips; the [−0.5, +1.0] window *is* a DCASE onset collar; add F1-vs-lateness curve (0.5/1/2/5 s).
Gate: 2×2 vs gold ticks, balanced accuracy + MCC with CI; not folded into F1. Sharpest gate number with no new TEST read:
gated ⊂ blind, so "pictures removed" splits into wrong-removed (FA_blind − FA_ours) vs needed-removed (hits_blind −
hits_ours), both already in TEST counts — that pair is what precision +0.137 means. Timing: lateness histogram p50/p90 +
long-sound coverage. Picture: blind glance test; Gemma = triage, not outcome. **Headline: β cost curve with crossings**
(Drummond & Holte 2006 cost curves; IR utility), interval where ours is cheapest. F1 stays pre-registered primary, first,
null. DHH grounding for β: Jain CHI 2015, Findlater 2019, Goodman 2020 (check years).

**Q2 Human eval.** (1) Second annotator 30 clips: κ on needed/visible/importance + re-score every system under annotator-2
gold. (2) Glance test with 3–5 naive raters on the 54 sounds, same protocol; Fleiss κ + per-rater rate. (3) Hearing sound-off
proxy, within-subject panel / caption / nothing, Latin square over ~20 DEV or slice-B clips; "which sounds, where was the
source?"; recall, false-belief rate, 1–5 confidence; n ≈ 18–24 for d ≈ 0.6; mixed model; ethics is the blocker.
(4) DHH 2–3 qualitative pilot, not a result. (5) Not on TEST.

**Q3 Presentation.**
1. F1 first, null, CI, MDE 0.13, 320–570 clips; null by construction (gated ⊂ blind; equal weights).
2. Viewer cost: "weights pre-specified, promoted after F1 null"; full β curve; Holm doesn't repair post-hoc entry — say so.
3. **Drop/relabel "F1 vs silence" (+0.381)**: silence F1 = 0 by definition — a tautology.
4. **Off-screen subgroup (13 clips, −3.08 vs silence) is the same trap**: silence must miss every needed sound there. On that
   stratum ours = blind = 4.00/clip — the gate adds nothing there. Honest reading: gate earns its keep on "source on screen"
   (0.67 vs 2.50) and "mixed" (6.27 vs 8.00); over all clips cost vs silence is null (−0.23 [−1.00, +0.47]).
5. P(d>0) is not a p-value; print only the two-sided bootstrap p.
6. TEST exposure table (10 rows, what read, which decision could depend on it); "held out, with disclosed exposure";
   split overlap in the same table.
7. Reconcile β crossings: beta_specification says ours beats silence below β = 1.37 [0.71, 2.19] (DEV, older tag); ch5
   says 2.56 TEST / 2.33 DEV — state the pipeline tag per row.
8. Failed instruments as a section.
9. Code disclosures: load_gold defaults importance to 2 when missing (0 of 298 today); clean-clip accuracy counts level-1
   needed sounds as needed while the headline treats them as don't-care; cross-trigger vs phantom uses any gold sound incl.
   visible level-1; precision 0 when nothing shown.

**Q4 Benchmark.** Yes, as a **pilot benchmark** (n = 139, one annotator) — claim the protocol, not the numbers. Release clip
ids (check licences), annotation protocol (importance, visible/obvious, exhaustiveness, "dog at 45 s"), scorer with ALIASES
and ontology-depth rule, four baselines (blind, caption, silence, oracle-gate), agreement, metric card (β range, collar).
Add a small sealed held-out split + a script scoring a submission folder.

**Q5 Don't.** Swap the headline to whatever came out significant. F1 first; cost as curve not β = 2 point; the off-screen-
vs-silence row must not carry the abstract.

---
# Round 2

## T3-c — round 2

**(1) Headline ordering.** Keep the pre-registered order: F1 first (null, CI, MDE 0.13, 320–570 clips, oracle diagnostic
same paragraph) → Holm-significant secondary family (precision +0.137, wrong pictures −0.52, clean-clip acc. +0.22; recall
−0.07 printed as the price) → β cost curve with interval sentence (cheaper than blind above β 0.39, than silence below
2.56, TEST; DEV twin beside). Curve-first contradicts an order signed 5/5 before the numbers. "F1 first" and "restraint" are
the same thing: the confirmatory rows are the restraint result. Agreed with T3-b: tag per row (beta_spec 1.37 = DEV, older
tag; ch5 2.56/2.33 = final tag); print only two-sided bootstrap p.

**(2) Silence rows.** Drop "F1 vs silence" (+0.381) from tables, one sentence only. Off-screen −3.08: ours = blind = 4.00
there, so not a gate result — keep as pre-declared subgroup, relabel "pipeline recovers off-screen sounds (gate-neutral
stratum)", move to the per-category table, out of §5.4 headline and abstract. Gate's own evidence: on-screen (0.67 vs 2.50)
and mixed (6.27 vs 8.00). Label: "descriptive, by construction".

**(3) Hearing pilot.** Monday: ask supervisor whether an internal hearing sound-off pilot needs an ethics filing. If yes:
skip; thesis gets T3-a's design as "designed, not run" beside the DHH design. If no: run T3-a's design (n 10–12, DEV clips
with ≥ 1 needed sound, gated/blind/none Latin square, 4-option forced choice primary — no blinded scorer needed;
helped/neither/hurt secondary), powered for gated-vs-none only, "instrument validation for a future DHH study". n 18–24
unreachable; free-recall variant withdrawn. Adam's hours: second annotator > 3–5 glance raters (1 h; same ethics question)
> pilot.

**(4) Abstract.** Task; pipeline; pilot benchmark (139 clips, one annotator, second on 30 with κ); F1 null with MDE; gate
halves wrong pictures at non-sig recall price (held-out, Holm); modelled viewer cost as interval over β, "β = 2 assumed";
picture recognition +22 pts (independent raters if done, else "author-rater"); "DHH helpfulness untested". Not: F1 vs
silence, −3.08, "unseen" for TEST.

**(5) Benchmark.** "Pilot benchmark v1 — 139 clips, one annotator (+ 30-clip second pass); the protocol is the
contribution, the numbers are provisional". Minimal release: clip IDs + timestamps + licence; per-sound table; guideline
with frozen tick definitions; κ table; scorer + cost_curve + README reproducing the thesis table; blind/silence/text/
oracle-gate outputs; exposure log; metric card. Sealed split + submission script: v2.

**Agreed (all three):** F1 primary and first; cost as curve; second annotator top priority; no TEST read; pilot DEV-only,
labelled proxy; "DHH helpfulness untested" + one annotator in abstract; exposure log + split overlap in one table;
benchmark as pilot v1.
**Open:** (a) ethics answer decides pilot and maybe glance raters; (b) T3-b's scorer disclosures — disclose in ch5, fix only
on DEV (TEST re-score = another read); (c) pilot primary: forced choice (T3-a, T3-c) vs helped/hurt (T3-b silent).

## T3-a — round 2

**(1) Headline — concedes.** Order = pre-registered: ΔF1 first (null, CI, MDE 0.13, 320–570 clips, oracle-vocab +0.03)
→ Holm-confirmed restraint (precision +0.137, wrong pictures −0.52/clip, clean-clip +0.22) → β cost curve as end-to-end
interpretation. Abstract claim sentence = restraint, then the interval sentence with final-table numbers.

**(2) Silence rows — agrees.** Drop F1 vs silence. Off-screen: silence's 7.08 = 4 × misses by construction; ours = blind =
4.00 → −3.08 is a *detector* result, descriptive per-category row only. Replace with: (i) strata where the gate acts —
on-screen 0.67 vs 2.50 (−1.83 [−2.83, −1.00]), mixed 6.27 vs 8.00 (−1.73 [−3.73, −0.13]); (ii) wrong-removed vs
needed-removed split from existing TEST counts; (iii) all-clips cost vs silence null (−0.23 [−1.00, +0.47]).

**(3) Pilot — one design.** n = 12 hearing lab members (15–18 if recruitable); 24 DEV clips (8 off-screen / 8 mixed / 8
nothing); nothing / blind / ours Latin square, **no caption arm**. **Primary: free recall** ("list the sounds you could
not see", scored by the glance-test family matcher) — withdraws forced choice (distractors leak the answer; the 0.48 power
anchor came from typed answers). Secondary: false-belief count (only endpoint that might separate ours from blind), then
helped/neither/hurt. Mixed logistic, two pre-declared contrasts. Ethics: no supervisor answer by Mon 29 Sept → skip, keep
design as pre-registered appendix. Never TEST; never "DHH study".

**(4) Priority: annotator > raters > pilot.** Annotator: κ on needed/visible/obvious, weighted κ on importance, median
|Δstart|; gold-robustness re-score on the packet's DEV clips only — TEST clips in the packet: κ only. Raters: 3–5 hearing,
same protocol, ~15 min each; per-rater recognition per generator, pairwise/Fleiss κ, Adam relabelled "author-rater"; same
ethics question.

**(5) β crossings — resolved.** 1.37 [0.71, 2.19] = DEV, 2026-09-23, cost_curve under use_v4("59") (8-s cap on,
pre-amendment 21); 2.56 TEST / 2.33 DEV = final tables, 2026-09-27, cap off. Later supersede. β-spec §2/§6 get a dated
"superseded — see ch5 §5.5" banner; every crossing prints tag + split + date; DEV interval recomputed on dev_monocap_v31;
TEST interval only if ruled a re-presentation of the closed table, not a new read.

**Agreed:** F1 → restraint → curve; drop F1-vs-silence; −3.08 descriptive, replaced by gate strata + removed split; pilot
12+, 24 DEV clips, 3 arms, free recall + false beliefs, ethics-gated; annotator > raters > pilot; annotator-2 re-score DEV
only; 1.37 superseded; "DHH helpfulness untested" in abstract; exposure log + failed-instruments section; code disclosures
in appendix.
**Open:** TEST CI on the crossing = new read?; ethics answer in time; n 12 vs 18.
*(Note: T3-a now prefers free recall, T3-c now prefers forced choice — they swapped; settle in round 3.)*

## T3-b — round 2

**(1) Headline.** F1 first (+0.059 [−0.030, +0.144], p 0.18, MDE 0.13). Then Holm family-1 as the examiner's headline,
**led by wrong pictures/clip −0.52 [−0.80, −0.28] (p_holm < 0.001)** — not precision (did not replicate on DEV); recall
−0.07 [−0.15, 0.00] = "no detectable price", not "free". Then β curve as interpretive frame (interval claim). Curve-first
leads with the post-hoc metric = the Q5 error. Source: benchmark/gold/holm_test_final_v33_test_bench.json.

**(2) Tautology — refined.** (a) F1 vs silence is a true tautology → drop from family 2 (flips nothing: other rows keep
p 0.585 null and p < 0.001). (b) Off-screen −3.08 is *not* true by construction (ours could have exceeded 7.08 with ~1.5
wrong pictures/clip) — it is a **detection + picture result, gate-independent** (ours = blind = 4.00; ours − blind 0.00
[−0.62, +0.77]). Keep, relabel "the gate plays no part here". Verified (JSON `categories`): unseen 13: 4.00/4.00/7.08;
mixed 15: 6.27/8.00/5.33, ours − blind −1.73 [−3.73, −0.13]; seen 12: 0.67/2.50/0.00, −1.83 [−2.83, −1.00]; no_ambient
20: 0.20/0.30/0.00, −0.10 [−0.30, 0.00]; all clips ours − silence −0.233 [−1.000, +0.467], p 0.585. Replacement §5.4:
family 2 keeps "cost vs silence, all clips (null)" + "cost vs silence, off-screen (detection + picture; gate-free)"; add the
three per-category ours-vs-blind rows as **descriptive, CI only** under "where the gate earns its keep". Blind also beats
silence below β = 1.36 (v31, v33) → "beats silence" is never the gate's achievement.

**(3) Pilot — concedes n.** n = 10–12, sound off, gated / blind / none (caption arm dropped), 24 DEV-49 clips (8/8/8), Latin
square, ~25 min. Primary: recognition of the off-screen sound (4-option forced choice), gated vs none. Secondary: false
beliefs + 1–5 distraction, gated vs blind on on-screen + mixed clips, exploratory. Mixed logistic. Only if supervisor says
by Monday no ethics filing; β-spec §5 must be amended from TEST to DEV stimuli.

**(4) β crossings, one table** (recomputed from committed curve files, grid 0–4): test_final_v33 / test_bench / 60: 0.39,
2.56. dev_monocap_v31 / dev / 49: 0.46, 2.33. v4b4 / bench / 109: 0.40, 1.27. beta_spec §2 (DEV, 23 Sept, v4b6 era): 0.35,
1.37 [0.71, 2.19]. **T3-a's 0.35 / 1.37 / 4.30 is the stale set.** `docs/FOR_SUPERVISOR.md:68` still quotes 1.37 — update.
The 4.30 oracle ceiling is not reproducible from any committed curve file → "consistent with > 4, not verified".

**Agreed:** F1 first, null with MDE; annotator 2 + independent glance raters; no TEST reads; 10-row exposure log +
relabels; benchmark v1/pilot; pilot DEV-only, no ethics filing.
**Open:** T3-a accepts F1 → restraint → curve? (yes, in T3-a r2); T3-c accepts "gate-free" + descriptive per-category rows?;
ethics; retire 1.37 / 4.30 everywhere.

---
# Round 3 (final)

## T3-a — round 3

(a) Forced choice wins (2 vs 1; no blinded scorer needed). 4 options per clip fixed before any participant: true family,
one same-category family from another clip's gold, one other family, "none"; optional free-text line after (exploratory).
(b) Wrong pictures/clip −0.52 leads the Holm family (replicated on DEV; user-facing quantity). (c) T3-b's label: "detection
+ picture result, gate-free (ours = blind = 4.00)"; keep as pre-declared family-2 row (dropping it is selective), out of
abstract and §5.4 headline. (d) A TEST CI on the crossing is a new TEST number → skip; print TEST point crossings with the
DEV CI beside. (e) Retire 0.35 / 1.37 / 4.30 everywhere; final: blind 0.39 TEST / 0.46 DEV, silence 2.56 TEST / 2.33 DEV;
4.30 → "consistent with > 4, not verified from a committed file".

Recommendations (Adam-hours):
1. Send the second-annotator packet Monday; κ (needed, visible, obvious, weighted importance) + median |Δstart|;
   gold-robustness re-score on the packet's DEV clips only — 0.5 h send, 1 h score.
2. Ask supervisor Monday: does an internal hearing sound-off pilot (and naming by 3–5 lab members) need ethics filing? — 0.2 h.
3. Independent glance raters, 3–5 hearing, same 1.5 s protocol + answer sheet; per-rater rate per generator + κ; Adam =
   "author-rater" — 1 h setup, 15 min per rater.
4. Pilot only if "no filing": n = 12, 24 DEV clips (8/8/8), nothing/blind/ours Latin square, forced-choice primary (ours vs
   none), false beliefs + distraction secondary (ours vs blind, exploratory), mixed logistic — 3 h materials + 5 h sessions
   + 2 h analysis. If "filing": one page "designed, not run" — 1 h.
5. Amend β-spec §5 stimuli TEST → DEV for the hearing pilot; DHH design keeps TEST for a future run — 0.5 h.
6. Thesis edits: F1 → Holm family led by −0.52 → curve; F1-vs-silence to one sentence; relabel −3.08; "superseded"
   banner on β-spec §2/§6, fix FOR_SUPERVISOR.md:68; crossings with tag + split + date; DEV crossing CI on dev_monocap_v31;
   exposure log + split overlap in one table; "DHH helpfulness untested" + "one annotator (+30-clip second pass)" in
   abstract — 3 h.
7. No TEST read for anything — 0 h.

**I sign; no dissent.**

## T3-b — round 3

1. Headline: F1 → restraint → curve. Lead sentence: "wrong pictures per clip −0.52 [−0.80, −0.28], Holm p < 0.001, at no
   detectable recall price (−0.07 [−0.15, 0.00])". Precision +0.137 in the same table, "TEST-only, not replicated on DEV",
   not in abstract. — 1 h.
2. Silence rows: keep family 2 exactly as pre-declared in holm_table.py; change labels/text, not rows (deleting after
   seeing is post hoc). Row 1 footnote "silence F1 ≡ 0; true by definition, no claim". Row 3: "off-screen (13): detection +
   picture result; gate-neutral (ours = blind = 4.00, Δ 0.00 [−0.62, +0.77])". Neither in abstract or §5.4 heading. Add
   per-category ours-vs-blind rows (on-screen −1.83, mixed −1.73, nothing-to-draw −0.10) descriptive, CI only. — 1.5 h.
3. Pilot primary: **free recall** (moves to T3-a's round-2 view): a forced-choice list tells the viewer a sound happened —
   the very thing the panel conveys; power anchor came from typed answers scored by answer_sheet.py, reusable. n = 12,
   24 DEV clips 8/8/8, none/blind/ours Latin square; primary ours vs none; secondary false beliefs ours vs blind
   (on-screen + mixed). Only if no ethics filing by Mon 29 Sept; else appendix + β-spec §5 TEST → DEV. — 6–8 h / 1 h.
4. TEST CI on the β crossing = **re-presentation, allowed** if logged as exposure row 11 "derived statistic on the closed
   table; no decision"; no re-render or parameter change. — 0.5 h.
5. Scorer disclosures: appendix; DEV-only flag for clean_n counting only scored needed sounds (print DEV both ways); TEST
   not re-scored; importance default and cross-trigger rule one sentence each. — 1 h.
6. β table with tag/split/date per row (0.39/2.56 TEST v33; 0.46/2.33 DEV v31; 0.40/1.27 bench v4b4; 0.35/1.37 DEV 23 Sept
   superseded); banner on β-spec §2/§6; fix FOR_SUPERVISOR.md:68; 4.30 → "not crossed within β ≤ 4; not verified". — 1 h.
7. Priority: second annotator > 3–5 glance raters > pilot. — 2 h + 1 h.

**Signs 1, 3, 4, 6, 7. Dissents on 2 in form only** (keep declared table intact and relabel, vs moving −3.08 out) —
numbers and abstract identical either way. *(Note: T3-a's round 3 also keeps it in family 2, so this dissent is moot.)*
*(Note: pilot primary swapped again — round 3: T3-a forced choice, T3-b free recall; T3-c pending.)*

## T3-c — round 3 (examiner)

1. Pilot primary = **free recall** (concedes): score only with the committed glance-test family matcher; unmatched = miss
   or adjudicated blind to arm — never hand-fixed by Adam. Secondary: false beliefs, then helped/neither/hurt.
2. Lead row = wrong pictures −0.52 [−0.80, −0.28]; precision second, "TEST-only (Holm 0.080 on DEV)"; print wrong-removed vs
   needed-removed pair; recall −0.07 = "no detectable price". Adam 0.5 h review.
3. Off-screen −3.08: accept T3-b's label; keep in family 2, "detection + picture, gate-free (ours = blind = 4.00)";
   per-category rows descriptive; not in abstract.
4. TEST CI on the β crossing = re-presentation **if** computed from committed per-clip counts of test_final_v33, same 2000
   draws, declared in the prereg before computing, no re-render/re-score, logged as "10b, derived, no decision".
5. Retire 1.37 / 4.30; fix FOR_SUPERVISOR.md:68 and β-spec §2/§6 banners; recompute DEV oracle crossing from committed
   oracle votes (≤ 1 h), else "above roughly 4, not verified".
6. Second annotator first (κ + weighted κ + median |Δonset|; re-score DEV clips only). Adam 1 h; annotator ~3 h.
7. 3–5 independent glance raters; Adam = "author-rater". Adam 1 h. Same ethics question.
8. Ethics question to supervisor Monday. No filing → pilot n 10–12, 24 DEV clips, 3 arms, Latin square (Adam ~6 h).
   Filing or no answer → "designed, not run" appendix, β-spec §5 TEST → DEV.
9. Abstract: F1 null + MDE; gate halves wrong pictures at no detectable recall price (held-out, Holm); modelled cost as
   interval over β ("β = 2 assumed"); picture +22 pts (rater label); one annotator + κ; "DHH helpfulness untested".
10. "Pilot benchmark v1"; licence check 0.5 h; scorer disclosures, fix on DEV only.

Total Adam: ~4 h without pilot, ~10 h with. **Signs; no material dissent.** Reservation: if free recall needs human
adjudication and no blind adjudicator exists, primary reverts to helped/neither/hurt, not Adam scoring.

---
# Topic 3 — outcome (coordinator)

**Signed by all three.** Final votes on the last open points: pilot primary = free recall (T3-b, T3-c) over forced choice
(T3-a) — matcher-scored, never hand-fixed; TEST CI on the β crossing = allowed as a derived re-presentation (T3-b, T3-c)
if declared in the prereg first and logged as exposure 10b, vs "skip it" (T3-a) — coordinator recommends **skip** (costs
nothing; DEV CI beside the TEST point is enough). Off-screen row stays in family 2, relabelled gate-free (all three).

