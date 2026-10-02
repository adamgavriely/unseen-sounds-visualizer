# Panel 2026-09-26 — round 2 answers

Brief: docs/history/panels/panel_2026-09-26_round2_brief.md

## P1 — round 2
(verbatim text kept in the session transcript; key points:)
- Withdraws "~40 more unseen clips": the 23 Sep TEST look already has unseen cost vs silence +3.08 [+1.08, +5.38] sig.
- D1: (c) is already scored from cache (majority+obvious balanced 0.60 vs 0.62; §10d oracle ΔF1 +0.090 < §10e bar +0.12) — not live. (a) P3's bar mixes two tables and sits inside noise (balanced-accuracy half-width ≈ 0.11 on 79 sounds); a three-option a/b/c needs its own position-bias control. (b): first count DEV specs whose `reason` starts with "a kind of" in dev_monocap_v31/*/augmentations.json; under ~5 → diagnostic paragraph, not a test. If (a) runs anyway, only the §10e bar.
- D2 (a): the thesis describes one system in one table; render all three arms on one card class (H200/A100); commit "replaces the 23 Sep table whatever it shows; 23 Sep to the appendix" before any score; score only with Adam's yes. Risk of reversal low (onset fix lifted blind too, deltas unchanged in kind on DEV).
- D3 no. D4 yes: recruit only; existing tool (build_tool.py, tool_template.html), 25–30 stratified clips, same ticks; κ on needed and visible, plus inter-annotator onset spread (median |Δstart|) — the hit window is 1.5 s wide.
- D5 yes: near-identical spans, not identical (captions have no row limit, l.175–176); confound removal, expect null. D6 yes: shipped bars were set on DEV, per-family calib bars not adopted → calib is out-of-sample; union+vetoes row needs a PANNs framewise pass on the 280 clips. D7 no. D8 yes (reads subject texts the assistant wrote, not answers/arm key).
- One honest thing: reporting change + a timing subsection (onset rule = trace-proven bug fix, 223/442 starts moved earlier → 0; DEV-selected gain; TEST inconclusive; no cap by principle; median lateness 0.04–0.14 s).
- Tonight: (1) D1(b) count (2) D8 (3) D6 PANNs pass + report (4) D5 (5) D2 renders, nothing scored (6) Holm (7) only if time D1(a) under §10e alone.
D1=b→d D2=a D3=no D4=yes D5=yes D6=yes D7=no D8=yes

## P2 — round 2

**D1 Gate treatment — vote (d).** (a) P3's bar is inside noise: needed_kept 0.86 → 0.92 on 36 DEV needed sounds is +2 sounds, and the binomial SE of 0.86 at n=36 is ≈0.06; those two sounds are the DEV share of the six named losses, so the bar is "recover the known cases" in disguise. P3's forecast (+0.05–0.07 on the pooled 109, "the only route to a non-null F1") sits at the pooled MDE (≈0.07, ~50 % power at best) on a pool that contains DEV and a TEST read four times. (c) is already decided: §10d's oracle sweep gives the obvious-vote rule +0.090 [+0.023, +0.160], under the §10e bar of +0.12, so running it end to end cannot adopt it. (b) is the only cheap one (CPU from `gate_votes.json`) and its denominator is honest, but a 2:1 rule on counts of 1–3 is a coin flip and three mechanisms already landed at exactly 2.00. If anything: (b) as a *reported diagnostic*, adoption impossible on this gold — which is (d).

**D2 TEST reporting — vote (a), with P5's discipline.** P5 is right that two TEST tables discount both; the fix is the replace rule, not abstaining. Headlining the 23 Sep table would headline a configuration the thesis itself rejects (8-s cap, old timing) while the ours arm at the shipped timing already exists on TEST (`test_monocap_v31`) — the asymmetry is worse than a fifth read. Write, before any score: one TEST table in the thesis, the shipped config, replacing 23 Sep whatever it shows; 23 Sep goes to the history section with the read count. Risk stated now: TEST ΔP SE ≈ 0.06, so +0.143 may fall out of significance at the new timing (DEV shows +0.115 [+0.020, +0.204]); that is exactly why the rule must be written first.

**D3 More gold — not necessary.** The claim P1 wants to power already exists on TEST: unseen 13 clips, cost vs silence +3.08 [+1.08, +5.38]. DEV's +0.86 [0.00, +1.86] on 14 clips is the same effect, wider. Forty more clips would be a third sample whose purpose is to move a CI after seeing it — my Q6. If Adam wants it anyway: N fixed at 40 unseen/mixed, stratified as `split.py`, annotated before any render, one declared analysis (cost vs silence on unseen, β=2, paired), reported as a replication set, never pooled. Annotator: Adam, so it fails his own "only what is necessary".

**D4 Second annotator — yes, ask.** Who: anyone Adam can recruit (a lab peer); not the assistant. What: Adam's sound list with times pre-filled, the second person ticks visible / obvious / importance only — so κ measures the *definition*, not sound detection. How many: 30 clips stratified by category, 15 DEV + 15 TEST, ≈65 sounds rated ≥2; SE(κ) ≈ 0.1 at that n, enough to separate ≥0.6 from <0.4 (`FOR_SUPERVISOR.md` §3a bars). Report: Cohen's κ on `needed`, on `visible`, weighted κ on importance, plus one sensitivity row: the primary re-scored with the second person's ticks on those clips. P4's second rater on the 168 cards is the picture-side twin; both cost Adam one email.

**D5 Gated caption on DEV — yes, as a confound fix, no win claim.** F1/P/cost are span-only (`load_pictures`), so the arm scores identically to ours; only the judge moves. Power: judge vs caption SE ≈ 0.18, MDE ≈ 0.5 — a tie is the expected outcome. What it can claim: "under the same gate, pictures and text tags are judged alike (±0.4)", replacing a number that currently confounds modality with gating. Declare decision-free, B1/B2 first, minutes on an L4.

**D6 AudioSet-Strong detector report — yes.** The shipped stack's choices (FlexSED 0.8, vetoes 0.3/0.05) were made on DEV gold, not on the 280 calibration clips, so the set is out-of-sample for it; thousands of events, zero Adam time, the SED table an examiner expects. Write "report only" first; caveats already recorded (incomplete labels, 10-s clips, no "needed").

**D7 Synthetic mixes — no.** They measure the gate's false-silence rate where no source is on screen, which is the easy half (needed_kept already 0.86–0.88 on gold); the real false silences are same-family collisions, which a mix cannot construct without a visibility annotation. It adds a flattering number on an easier population, days before the thesis.

**D8 Qualifier count — yes.** A JSON count, no decision hangs on it, it bounds the RQ2 claim before the sitting rather than after.

**The one thing that most raises the significance claims honestly:** a writing change. One pre-specified results hierarchy, committed tonight: primary ΔF1 null first (with the oracle-label reason), then the declared secondary family with Holm (bootstrap two-sided p ≈ 2·min(P(d>0), 1−P(d>0)) from `paired_ci`, resolution 1/2000 — say so), then the β curve with the break-even p > β/(4+β), then the TEST read count and the primary's history (judge → F1 → cost). The corrected TEST numbers (ΔP +0.143, unseen vs silence +3.08) are already the strongest evidence; they only need to sit under that hierarchy instead of beside a null F1 without one.

**What I can do tonight, in order:** (1) commit the hierarchy doc and the D2 replace rule; (2) render blind + caption on TEST at the frozen config on the base card class, no scoring; (3) D5 gated caption on DEV + `judge_trust.py` B1/B2; (4) D6 calibration-set detector table; (5) Holm over the existing DEV and 23 Sep secondaries; (6) D8 count; (7) draft the D4 packet (30 clips, tool page) so Adam only forwards it.

D1=d D2=a D3=no D4=yes D5=yes D6=yes D7=no D8=yes

## P3 — round 2

**On P4/P5's correction (l.589–595 already asks "visibly making that sound").** True, and I should have said so. But it is one vote of three. The other two — `by_name` (l.578–584) and `by_desc` (l.606–611) — both resolve through `MAKES_SOUND_PROMPT` l.511–517, "*Could* that thing be what is making that sound?", a possibility question; under `VISIBILITY_RULE = "majority"` two possibility votes outvote the one attribution vote. And even the a/b offers no rival hypothesis: its "no" option is "{label} is not visibly happening", so with macaws in frame the model has no way to say "another bird, out of view". That is the defect I mean — but I concede P2's two points: DEV holds ~3 of the 6, name~ab agree 85 % (§10e), and 40/47 sensitivity misses are perception, so an attribution vote touches specificity only: ≈ +0.03–0.06 ΔF1 at best, below the 0.13 MDE. It can only ever be a labelled secondary, and it must run under the already-declared §10e bar (oracle ΔF1 > +0.12, ≤ 1 hit lost), not a bar I write now — P4 is right that a new bar is a new degree of freedom.

**D1.** Ordered pair **(b) then (a)**, and (c) is not separate — it is the same cached sweep. (b) is not a perception mechanism, so P2's 2.00 prior does not bind it: it is the same class of bug as amendment 3 (kinship silence was time-blind; now it is *source*-blind — "a tank is visible" silences a helicopter because `same_source ∧ _overlap`, l.1335, never checks the `named` thing). Everything is on disk: `VOTE_LOG` writes `named` per stretch into `gate_votes.json` (l.1299–1301) and the reason string "a kind of X, whose source is visible" is in `augmentations.json` (l.1339; `gate_redecide.py` already parses it, l.53–57). Rule, written before the run, denominator = every kinship-silenced sound on DEV: keep the silence only if `_about_the_sound(named, member)` or `MAKES_SOUND(named, member)` says yes; adopt iff 2 × needed recovered > visible pictures re-admitted (strict). (a) afterwards, GPU, as a 4th cached vote in `gate_gold.run_vlm` (the `obvious` pattern, l.82–89), scored on all 79 DEV sounds and swept with `gate_sweep.py` under the §10e bar; expect it to fail and say so.

**D2 = (a).** The shipped system has the onset rule; a thesis needs its *final* configuration against its baselines on TEST once, with the same detector, timing and card class. The 23 Sep table stays as the disclosed history of reads (P5's list of four), the replacement statement is written before scoring, scoring waits for Adam's yes. Two TEST tables is P5's risk — the fix is the replacement statement, not skipping the like-for-like.

**D3 = no.** Not necessary: 40 unseen clips halves a 0.93 half-width to ≈ 0.5 against d = 0.86 — borderline, ~3 h of Adam's annotation, and it moves no gate claim. The unseen-13 cost-vs-silence +3.08 [+1.08, +5.38] already exists on TEST.

**D4 = yes**, worth asking Adam to recruit one person (~1.5 h). 25–30 stratified clips (`split.json` strata, with the mixed stratum over-represented so the bird/bell collision cases fall in), ticks `needed`/importance **and** `visible`/`obvious` — the latter is the gate's ground truth and the "same-family collision" is a definition question only κ can answer. Report κ; no decision hangs on it.

**D5 = yes**, cheap and it deconfounds. Claim it can make: only "the picture-vs-text judge tie is not caused by the gate"; F1/P/cost are identical by construction (`load_pictures` reads spans). B1/B2 first; judge MDE ≈ 0.5 means no win can be claimed.

**D6 = yes** (shared stage, report-only, zero Adam; out of my lens).

**D7 = no.** For a frame-only gate, an injected sound with no same-family thing on screen is the easy case the gate already handles (needed-kept 0.86–0.88); the losses are collisions, and a synthetic collision has no ground truth either — the annotator would tick "visible". A labelled mechanism check at most; not necessary.

**D8 = yes** — CPU count of the 81 frozen subjects, reveals no sealed answer, bounds RQ2 at zero cost.

**The one thing that most raises the significance claims honestly:** a reporting change — state the two claims *stratified by the gate's own condition*, both already significant on TEST: on seen-source clips the gate's value is precision/false alarms/cost vs blind (ΔP +0.143, Δcost +0.73); on unseen clips the system's value is cost vs silence (+3.08, precision 0.846). Beside them: the gate as a visibility classifier (sens 0.43 / spec 0.88, `oracle_gate.py`), the oracle and half-oracle rows as labelled diagnostics, Holm over the declared secondaries, F1 null with the equal-weight explanation. That is one coherent story instead of one null primary.

**Tonight, in order, zero Adam:** (1) D1(b) on CPU from `gate_votes.json`/`augmentations.json`, rule committed in a doc first; (2) D8 count; (3) D5 gated caption DEV render + `judge_direct.py` + `judge_trust.py`; (4) D6 detector report on the 280-clip set; (5) D2 TEST render of all three arms at the current timing, unscored, replacement statement written; (6) D1(a) attribution vote cached on DEV `gate_gold` (H200, ~1 h), swept under §10e.

**D1=b→a D2=a D3=no D4=yes D5=yes D6=yes D7=no D8=yes**

## P4 — round 2

**Critique (from the evaluation lens).** P3's attribution vote is the best-argued gate idea, but its bar (needed_kept ≥ 0.92, seen_silenced ≥ 0.40) is set at the shipped numbers +0.06/−0.03 and ends with a fifth TEST read; even if it hits the bar, the oracle ceiling (+0.067) and P2's MDE (0.13) mean F1 stays null, so its only measurable output is a classifier row in `gate_gold.py`. P1's kinship restriction is honest in denominator but post hoc in motive, and NIGHT_REPORT §8 shows a gate-arm render moves ±1 picture with card class, so a ratio bar on ~3 DEV cases is noise. P2 and P5 are right that the judge cannot resolve pictures-vs-caption (MDE ≈ 0.5 on 49 clips); my D5 must therefore be claimed as a correction, not a win. P5's strongest point is the unseen-TEST row (+3.08 [+1.08, +5.38] vs silence, precision 0.846) — already significant, already read, needs no build.

**D1.** (c), as a secondary row only, no TEST read. The "obvious" vote is the one mechanism pre-declared before gold (amendment 6); run it end to end on DEV, score with `gate_gold.py` on all 79 sounds and `oracle_gate.py`, adopt only under §10e (oracle ΔF1 > +0.12, ≤ 1 hit lost). P3's attribution vote may be cached as a fifth vote in the same DEV pass and reported as a classifier row (sens/spec/balanced), never adopted this week: the bar was written after the baseline was seen and the six losses are the motive. Reject (b): three mechanisms at ratio 2.00 says the fourth will be too.

**D2.** (a). The judge rows (`judge_direct_dev_monocap_v31.json`), the picture freeze and the DEV table are all at the new timing; a thesis with DEV at one timing and TEST at another invites the "which table" question P5 fears. Write before scoring: "this table replaces the 23 Sep table whatever it shows; the 23 Sep table moves to the read-history appendix"; render all three arms on TEST now on the base card class, score only with Adam's yes, and judge TEST in the same reporting run (`prereg_per_picture_judge.md` already says TEST is judged only there). The read-history table (four reads) goes beside it.

**D3.** Not necessary. All-clips cost vs silence needs ~1,300 clips (P2); unseen-TEST is already significant; 40 more clips annotated by Adam is exactly the work he asked us not to make, and with N unfixed it is optional stopping. If Adam volunteers: N fixed at 40 unseen/mixed, one analysis (cost vs silence on unseen, β curve), declared before the first clip is opened.

**D4.** Yes, and it is the one item worth asking Adam for — one hour of someone else's time. A lab-mate/friend, not Adam; 25–30 stratified clips from `split.json`; ticks visible / obvious / importance only (the `needed` definition is derived); report Cohen's κ per tick, declared bars ≥ 0.6 substantial, < 0.4 unreliable (FOR_SUPERVISOR §3a). Never merge the two annotations into a new gold. From my lens add the same person on the 168 confirmation cards after Adam's answers are sealed: κ on picture recognition, reported, never a gate.

**D5.** Yes, report-only, minutes, zero Adam. Claim it can make: "at identical spans, pictures vs text tags is a pure modality comparison" — and the expected reading is a tie or caption win, because `judge_direct.py` line 136 hands the judge the label as text. Write before running: no decision hangs on it; the existing (ungated) caption row stays in the table with its confound stated; B1/B2 from `judge_trust.py` rerun on the new render before the delta is read. Its value is removing a wrong description from every table, not a number.

**D6.** Yes. Zero Adam, standard SED evaluation an examiner expects, shipped bars only, "report never select" written first; it is the honest sensitivity report for the shared stages and cannot touch the gate claim.

**D7.** No. The gate's false-silence failure is the collision case (same family on screen), which a source-free mix cannot produce; the specificity half is already measured (needed_kept 0.86–0.88 on DEV and TEST). It would be a labelled diagnostic that answers a question nobody is asking.

**D8.** Yes, after the confirmation is unsealed, not before: it reads the 81 sealed subject texts and the freeze says nobody looks first. It costs minutes, needs no rating, and bounds RQ2 honestly: pictures can beat a caption only on the sounds where the subject carries a qualifier beyond the family word.

**The one thing that most raises the significance claims honestly.** A reporting change: one like-for-like table, one configuration, with the hierarchy written above it — primary ΔF1 null with the oracle-label reason; the declared secondaries as a Holm family (`paired_ci` already returns `p_gt0`); the β curve, not a point; the judge stratified seen/unseen with its three asymmetries stated (text tag = label, 512 px, cap of 4, caption ungated); the unseen-TEST cost row vs silence as the headline sentence for the project's target category; the picture confirmation as the only picture claim. And correct "caption" to "ungated text tags" everywhere it appears (`docs/report/report.tex` col. 471 onward).

**Tonight, in order, no Adam.** (1) Write the D2 replace-rule paragraph, then render proposed/blind/caption on TEST at the current timing, base card class, scoring locked. (2) Gated caption on DEV (`GEN=placeholder SMALL_OK=1`, gate on), `judge_direct.py`, `judge_trust.py` B1/B2. (3) AudioSet-Strong 280-clip detector report at shipped bars. (4) Holm over the declared secondary family from the existing per-sound JSONs (CPU). (5) The "obvious" vote end to end on DEV as a secondary row under §10e, attribution vote cached alongside as a classifier row. (6) Prepare the second-annotator page (25–30 stratified clips) and the second-rater copy of the confirmation page, both unsent until Adam names a person.

D1=c D2=a D3=no D4=yes D5=yes(report-only) D6=yes D7=no D8=yes(after unseal)

## P5 — round 2

**Critique of the others (examiner's lens).**
- P3's diagnosis is correct and I under-read it in round 1: `MAKES_SOUND_PROMPT` (reason.py l.511–517) asks "Could that thing be what is making that sound?" — possibility, not attribution — and the kinship silence (l.1335, `same_source ∧ _overlap`) never checks that the *named* thing can make the member's sound. But the fix P3 proposes is a new treatment question with a bar written after the six losses were listed, and its adoption needs a fifth TEST read. A VLM on six stills cannot know whether the visible macaw is the one calling; option (b) "comes from out of view" will collect agreement, which is the exact failure the file's own comment (l.55–57) warns about. P2's prior (three mechanisms at ratio 2.00) applies.
- P1's variant is the only one that is CPU-only, declared as one rule at the β break-even, and evaluated on every kinship-silenced sound rather than the six. Its weakness: the denominator on DEV is probably 1–3 cases (the helicopter/tank case is TEST), so it cannot be adopted on that evidence, only reported.
- P2 is right about the MDE arithmetic and about optional stopping; P2's D2 proposal is the one I disagree with (below).
- P4's caption correction changes what the thesis may say: every "pictures vs caption" number is gated pixels vs ungated text. That must be fixed in the write-up whether or not D5 runs.

**D1.** (d) for the headline; P1's (b) run as a CPU sensitivity row from `gate_votes.json` and *reported*, not adopted, no TEST read. Exact rule for the row: denominator = every kinship-silenced sound on DEV; print recovered needed sounds and re-admitted visible pictures; the text states whether 2 × recovered > re-admitted. P3's attribution vote goes into future work with the possibility-vs-attribution diagnosis as the stated mechanism — that diagnosis is a contribution; a fifth TEST read for it in the last days is not.

**D2.** (b). The 23 Sep table was the single look under a frozen rule; the 24 Sep timing look was inconclusive by its own rule. A new-timing TEST table that "replaces" it is a fifth read whose result becomes the headline — an examiner counts reads, and five with a replacement is worse than four with a footnote. Allowed: P2's render at the frozen config with a written line "supplementary, reported whatever it shows, replaces nothing", scored only with Adam's yes.

**D3.** Not necessary. The only claim more gold would power — cost vs silence on unseen clips — is already significant on the held-out half (+3.08 [+1.08, +5.38], 13 clips) and directionally consistent on DEV; F1 cannot be powered at any reachable n. Forty more clips would cost Adam hours and invite the optional-stopping charge. If ever done: N fixed in writing, one analysis, DEV-only.

**D4.** Yes, ask Adam for one name. One lab member, ~1 hour, 25–30 clips stratified by `split.json` category × population; give them Adam's sound list with times and ask only the ticks `visible`, `obvious`, `importance` (κ on `needed` is what the metric hinges on; label detection agreement is a separate, harder task). Report Cohen's κ on `needed` and on `visible` with a bootstrap CI, thresholds ≥ 0.6 / < 0.4 as in FOR_SUPERVISOR §3(a), as a limitation quantifier, never as a gate on any number.

**D5.** Yes, DEV only, minutes on an L4, zero Adam time. Declare before reading: the gated-caption judge row replaces the confounded ungated row in the judge table; B1/B2 rerun first. What it can claim: pictures vs text *at equal gating* on the judge — expected tie or text wins, and the thesis states the judge's text-tag asymmetry beside it. It cannot claim anything on F1/P/cost (spans identical to `proposed` by construction, `load_pictures`).

**D6.** Yes, low priority. The shipped stack's bars (FlexSED 0.8, τ 0.3, τ2 0.05) were chosen on DEV gold, so the 280-clip set is untouched by their selection; a report-only table (onset-recall, false spans/min, onset MAE, masked recall) is the SED evaluation an examiner expects and costs nothing. Label it "descriptive; no selection".

**D7.** No. Off-screen by construction makes the gate trivial except in the collision case, and a deliberately built collision (bird call over visible birds) is scored against a definition the gold itself disputes. New experiment, new caveats, days left.

**D8.** Yes. CPU, one script over `confirm_set_frozen.json` subjects: count subjects whose head phrase adds a kind beyond the family name. It bounds the share of sounds where a picture can carry more than a caption and answers the proposal's RQ2 without touching the sitting.

**The one thing that most raises the significance claims honestly.** A writing change: headline the corrected 23 Sep TEST numbers (ΔP +0.143 sig, Δcost +0.73 sig, unseen vs silence +3.08 sig, ΔF1 null with MDE ≈ 0.13 stated) together with a primary-history table (judge → per-sound F1 → cost), Holm over the seven declared secondaries using `p_gt0`, and a table of every TEST read. The brief itself undersold the result by quoting an old row; the thesis must not.

**What the assistant can do tonight, zero Adam work, in order.** (1) The reporting tables above (Holm, history, read count, DEV-new-timing beside TEST-old-timing). (2) D8 count. (3) D5 gated caption on DEV, declaration written first, B1/B2, then judge. (4) D6 calibration-set detector table. (5) P1's kinship sensitivity row from `gate_votes.json`, reported only. (6) Prepare the second-annotator page and a two-line instruction so Adam only forwards a link. Nothing that reads TEST.

D1=d (b reported only) D2=b D3=no D4=yes D5=yes D6=yes D7=no D8=yes

