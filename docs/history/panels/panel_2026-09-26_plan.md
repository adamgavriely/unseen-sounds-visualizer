# Panel 2026-09-26 — the merged plan, v2 (round 3 amendments merged; round 4: final sign)

Rounds: `docs/history/panels/panel_2026-09-26_round1.md`, `_round2.md`, `_round3.md`. Round 3: P5 signed (dissent on D2
recorded), P1/P2/P3/P4 signed with amendments, all merged below. Where two amendments conflicted, the choice is
marked **[merge]** with its reason.

| | P1 | P2 | P3 | P4 | P5 | outcome |
|---|---|---|---|---|---|---|
| D1 gate treatment | b→d | d | b→a | c (+a cached) | d (b reported) | **no adoption; (b) as a reported CPU diagnostic** |
| D2 TEST table | a | a | a | a | b | **a (4–1)**, P5's dissent recorded |
| D3 more gold | no | no | no | no | no | **no** |
| D4 second annotator | yes | yes | yes | yes | yes | **yes — Adam names one person** |
| D5 gated text tags DEV | yes | yes | yes | yes | yes | **yes, derived from ours, no decision hangs on it** |
| D6 AudioSet-Strong report | yes | yes | yes | yes | yes | **yes, report only** |
| D7 synthetic mixes | no | no | no | no | no | **no** |
| D8 qualifier count | yes | yes | yes | yes | yes | **yes, now** (from the subjects file; command logged) |

Agreed by all five: ΔF1 vs blind cannot become significant at any reachable n (MDE ≈ 0.13 on DEV, ~320–570
clips for a true +0.03–0.05); the honest gain is a **reporting** change, not a build.

## A. The results hierarchy (committed before any new number is read)

1. **Primary**: per-sound ΔF1 ours − blind, stated first, **null**, with the MDE and the oracle-label reason
   (equal-weight F1 cancels by construction).
2. **Family 1 — ours − blind, Holm.** The seven secondary rows `score_per_sound.main()` prints beside F1:
   ΔF0.5, ΔwF1, ΔP, ΔR, ΔFA/clip, Δcost (β = 2), Δclean-clip accuracy. Two-sided bootstrap
   p = min(1, 2·min(P(d ≥ 0), P(d ≤ 0))) over the 2000 paired draws (ties counted on both sides), computed from
   the draws (`paired_ci` extended to return the diff array; seed 0 reproduces). Resolution 1/2000; p = 0
   printed "< 0.001". Holm at 0.05 over the seven, **separately for each table** (DEV like-for-like; 23 Sep
   TEST; the D2 TEST table if scored). The sign is printed beside each p: ΔR is expected negative, and a
   surviving ΔR is a surviving *loss*. F1 stays outside the family. No row dropped.
3. **Family 2 — ours − silence, Holm** (P3) **[merge: P3's family, P2's label]**: ΔF1 all clips, Δcost all
   clips, Δcost on the unseen stratum — three rows, own Holm, per table. The unseen row is labelled "one
   pre-declared subgroup (amendment 5) of a post-hoc metric (amendment 9)". Declaring it here, with the two
   all-clip rows, is what keeps the +3.08 cell from reading as chosen after the fact.
4. **Other category rows** (ours − blind by category): CIs only, no stars, not in any family (P2).
5. **The β operating curve** (`cost_curve.py`) with break-even p > β/(4+β) and the crossover CI; never one β
   as a headline. Viewer cost keeps its label "declared post hoc; constants from the September gate sweep".
6. **Story, stratified by the gate's own condition**: seen-source clips → the gate's value is ΔP/ΔFA/Δcost vs
   blind; unseen clips → the system's value is cost vs silence.
7. **History tables**: every TEST read (22 Sep planned, 23 Sep accidental, 23 Sep single look, 24 Sep timing,
   and any read below, footnoted as the fifth); the primary's history (judge → per-sound F1, 19 Sep before gold →
   cost, 22 Sep post hoc).
8. **Timing subsection**: onset rule = trace-proven bug fix (223 of 442 starts moved earlier → 0); gain
   DEV-selected; TEST inconclusive; no cap by principle; median lateness 0.04–0.14 s.
9. **Wording fix**: "caption" → "ungated text tags" wherever the ungated arm is meant.
10. **Pictures**: the sealed confirmation sitting is the only picture claim; judge rows carry their asymmetries
   (text tag = label, 512 px, ≤ 4 pictures).

## B. Work with zero Adam time (order)

1. **D1(b) count** (CPU, minutes): DEV specs in `dev_monocap_v31/*/augmentations.json` whose reason starts
   with "a kind of" (`reason.py` l.1339). Rule written now: if ≥ 5, re-decide on CPU via the
   `gate_redecide.py` path — keep the kinship silence only if the **parent's** logged `named` phrase (per
   stretch in `gate_votes.json`, stretches with `seen` true) satisfies `_about_the_sound(named, member label)`;
   else re-admit the member as a planned picture. Report, paired per clip against the current render: needed
   sounds recovered (Δhit) and false alarms re-admitted (Δ(visible + cross + phantom)), with
   "2 × recovered > re-admitted" as the β = 2 reading. Printed caveats: word overlap errs toward re-admitting;
   corroboration/plausibility/dedup not re-run, so recovery is an upper bound. **Never adopted**, no TEST read.
   If < 5, one diagnostic paragraph. Possibility-vs-attribution (P3) goes to future work.
2. **D8 count** (CPU): of the 81 frozen confirmation subjects (the V31G subjects file, not the pictures), how
   many carry a kind beyond the family name; command logged.
3. **Holm tables** (CPU): DEV like-for-like (`*_dev_monocap_v31`) and the 23 Sep TEST look. Re-scoring the 23
   Sep TEST renders for p-values is allowed only if every point estimate reproduces exactly (guard printed); it
   adds no new quantity.
4. **D5 gated text tags on DEV — derived, not re-rendered** (P4) **[merge: P4 over P1 — P1's new system name
   would re-run the 27B gate, and a fresh gate pass differs by ±1 picture per card class, so spans would not be
   identical]**: for each DEV clip copy `protocol_proposed_dev_monocap_v31/<clip>/augmentations.json` and
   `media.json` into `protocol_audio_caption_<newtag>/<clip>/` (no images); symlink `protocol_proposed_<newtag>`
   → `protocol_proposed_dev_monocap_v31`. No GPU for the render; the ungated render is untouched.
   `judge_trust.py` B1/B2 on the new text arm first, then `judge_direct.py --tag <newtag> --systems
   proposed,audio_caption`. Declared: no decision hangs on it; the judge table shows both text rows (ungated =
   the baseline, derived-gated = the modality comparison); expected tie (MDE ≈ 0.5).
5. **D6 AudioSet-Strong report**: shipped stack at shipped bars on the 280-clip calibration set (PANNs
   framewise pass first) — onset-recall at [−0.5, +1.0] s, false spans/min, onset MAE, masked-event recall for
   BEATs / FlexSED / union / union + vetoes. Labelled "descriptive; no selection" (the shipped bars were set on
   DEV gold; the calibration-fitted per-family bars were not adopted, so the set is out of sample).
6. **D2 TEST renders**: ours, blind and ungated text tags at the shipped configuration, all on one card class
   (H200/A100); each arm copies the base env of `test_monocap_v31` (`V4=590 FBAR=0.8 VETO=0.3 PVETO=0.05`,
   onset rule on, `MAXSPAN=none`), verified in the job log lines `[v4] …` before the render is kept (P1). The
   derived gated text arm of B.4 may be made from the TEST ours render at no cost, scored only under the same yes.
   Committed before any render: *"If Adam says yes, this table replaces the 23 Sep table as the thesis's TEST
   table whatever it shows; the 23 Sep table moves to the read history. If Adam says no, the 23 Sep table stays
   the headline and the onset fix is reported as DEV-selected, TEST inconclusive."*
   Guards (P5, P2, P3): the TEST render jobs run with no scorer, judge or audit invoked on the TEST tag;
   `score_per_sound.py`, `arm_compare.py`, `judge_direct.py` and `timing_audit.py` are not pointed at it, and
   the TEST render directories are excluded from every script that loops over both halves. The assistant reads
   TEST job logs for completion and errors only; no per-clip line (gate verdicts, subjects) is quoted anywhere
   until Adam's yes. Adam's yes/no is written into `docs/history/preregistrations/prereg_v4.md` with a timestamp before any number
   exists. If no, the renders are never scored in this thesis. If yes, the table is scored once with families 1
   and 2 and the category rows, and the read joins the history table as the fifth. No other TEST number is read
   after it. (P5's dissent — a fifth read — recorded.)
7. **D4 packet**: the second-annotator page (existing tool `build_tool.py` / `tool_template.html`, 30 clips
   stratified by `split.json`, 15 DEV + 15 TEST, the mixed stratum over-represented), **Adam's family list per
   clip pre-filled, start/end left blank for the annotator to mark**, ticks visible / obvious / importance
   (P1). Unsent. Reported: Cohen's κ on `needed` and on `visible` with bootstrap CIs (≥ 0.6 / < 0.4 per
   `FOR_SUPERVISOR.md` §3a), weighted κ on importance, median |Δstart|, and one sensitivity row (the primary
   re-scored with the second person's ticks on those clips). Never merged into the gold, never a gate. The same
   person may rate the 168 confirmation cards after Adam's answers are sealed (κ on recognition, reported only).

## C. What Adam decides (two lines)
- **Yes/no**: score the D2 TEST table (a fifth TEST read, rule above).
- **One name**: a person for the second annotation (~1 h).

## Not doing
New gate mechanisms for adoption, more gold, synthetic mixes, a 4th automatic picture instrument, lower
detector bars, a cap, any change to the sealed confirmation sitting.

---
**Round 4 question to each reviewer:** final SIGN or one blocking objection (≤ 4 lines). Check the two
[merge] choices in particular.

## Round 4: SIGNED 5/5 (2026-09-26)
Execution notes adopted (`docs/history/panels/panel_2026-09-26_round4.md`): B.3 Holm via a new helper returning the draws,
with an equality assert against the printed row; B.4 derived text tags restricted to ours' **placed** spans, with a
per-clip span-equality print; B.7's re-score listed in the read history.
