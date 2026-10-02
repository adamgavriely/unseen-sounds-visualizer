# Panel 2026-09-26 — round 3 answers (sign or amend)

Plan: docs/panel_2026-09-26_plan.md

## P1 — round 3

**AMEND** (three points, verified against code; Holm family signed: `score_per_sound.main()` prints exactly seven ours − blind rows beside F1 — dF0.5, dwF1, dP, dR, dFA/clip, d cost/clip, d clean-acc).

1. **B.4 (D5) as written cannot run.** `run_protocol.py` l.156–161 hard-codes `GATE_ENABLED=False, VLM_VISIBILITY=False` for `audio_caption`; `SMALL_OK=1` means "no VLM" (`slurm/job_protocol.sh` l.82), so a gated tag arm needs the 27B gate and the base's card class. Replacement: *"D5: add a new system name `audio_caption_gated` in `run_protocol.py` (the `proposed` branch's flags with `RENDER_MODE='minimal'`, `GEN=placeholder`); the existing `audio_caption` branch is not edited. Runs on H200/A100 with the base env copied (`V4=590 FBAR=0.8 VETO=0.3 PVETO=0.05`, onset rule on, `MAXSPAN=none`); DEV only."*
2. **B.7 (D4)** contradicts itself (times pre-filled, yet onset spread reported). Replacement: *"Adam's family list per clip pre-filled, start/end left blank for the annotator to mark, ticks visible / obvious / importance."*
3. **B.6 (D2)** add: *"each arm copies the base env of `test_monocap_v31` (`V4=590 FBAR=0.8 VETO=0.3 PVETO=0.05`, onset rule on, `MAXSPAN=none`), verified in the job log lines `[v4] …` before the render is kept."*

Nothing in B touches a sealed thing or a TEST number. With these three edits: **SIGN**.

## P2 — round 3

**AMEND** (three places; everything else signed). `score_per_sound.paired_ci` (l.329–344) returns strict `P(d>0)` and discards the draws; for negative estimates `2·min(P(d>0), 1−P(d>0))` excludes ties from the small side (anti-conservative on ΔR, ΔFA, Δcost; ties are real for discrete statistics). Family of seven is right but applied per table.

> **A.2** Two-sided bootstrap p = min(1, 2·min(P(d ≥ 0), P(d ≤ 0))) over the 2000 paired draws (ties counted on both sides), computed from the draws themselves (extend `paired_ci` to return the diff array; seed 0 reproduces). Resolution 1/2000; p = 0 printed as "< 0.001". Holm at 0.05 over the seven ours−blind secondaries, **separately for each table** (DEV like-for-like; 23 Sep TEST; the D2 TEST table if scored); F1 stays outside the family as the primary. No row dropped.
> **A.4** Category rows carry CIs only, no stars, not in the Holm family; "unseen 13 clips, cost vs silence +3.08 [+1.08, +5.38]" is labelled "one pre-declared subgroup (amendment 5) of a post-hoc metric (amendment 9)".
> **B.3** Re-scoring the 23 Sep TEST renders to obtain p-values is allowed only if the point estimates reproduce exactly (guard printed); it adds no new quantity. **B.6** The TEST render directories are excluded from every script that loops over both halves (the amendment-13 accident) until Adam's yes.

D2 wording acceptable with the B.6 guard. Nothing else in B touches a sealed thing or a TEST number.

## P3 — round 3

**Checked B1 against the code.** (i) Reason string is exactly `"a kind of " + g.event_label + ", whose source is visible - stay silent"` (`reason.py` l.1339); `gate_redecide.py` l.53–57 parses it. (ii) The `named` phrase to test is the **parent's**, not the member's (the member was judged not visible); per stretch in `gate_votes.json` (`VOTE_LOG` l.1299–1301). (iii) `_about_the_sound` is word overlap: errs toward re-admitting — state it. (iv) The re-decide skips corroboration, plausibility, disambiguation, dedup → recovery is an upper bound. (v) The ledger must count every false alarm (visible + cross + phantom), all priced 2 in `viewer_cost` (l.316).

**AMEND B1 — replacement text:**
> 1. **D1(b) count** (CPU, minutes): DEV specs in `dev_monocap_v31/*/augmentations.json` whose reason starts with "a kind of" (`reason.py` l.1339). Rule written now: if ≥ 5, re-decide on CPU via the `gate_redecide.py` path — keep the kinship silence only if the **parent's** logged `named` phrase (per stretch in `gate_votes.json`, stretches with `seen` true) satisfies `_about_the_sound(named, member label)`; else re-admit the member as a planned picture. Report, paired per clip against the current render: needed sounds recovered (Δhit) and false alarms re-admitted (Δ(visible + cross + phantom)), with "2 × recovered > re-admitted" stated as the β = 2 reading. Caveats printed with it: word-overlap errs toward re-admitting; corroboration/plausibility/dedup not re-run, so recovery is an upper bound. **Never adopted**, no TEST read. If < 5, one diagnostic paragraph. Possibility-vs-attribution (P3) goes to future work.

**Holm family.** Seven printed rows is right; ΔR is expected negative — print direction beside the corrected p. The **vs-silence** rows used in A4 (ΔF1, Δcost, unseen-stratum Δcost vs silence) are a second comparison — declare them as their own three-row family with its own Holm, otherwise the unseen +3.08 cell reads as chosen after the fact. Stratified ours − blind rows stay descriptive (no p-values).

**D2 wording.** Acceptable, including the dissent line. Addition to B6: the render logs print each gate verdict (`[stage5] silent: …`, l.1311) and every subject; the assistant checks TEST job logs for completion and errors only, and no per-clip line is quoted anywhere until Adam's yes.

**Vote: AMEND (B1 text above; Holm/vs-silence and B6 log lines as additions).**

## P4 — round 3

**D8:** accept the 4–1 outcome; count from the subjects file, not the pictures folder, and log the command.

**Holm family:** seven is right. Print ΔR's sign beside its p (a surviving ΔR is a surviving *loss*); smallest reportable p is 0.001, so Holm's first threshold (0.05/7 ≈ 0.007) is resolvable.

**D2 wording:** acceptable.

**B4 — AMEND.** No switch exists: `run_protocol.py` `configure()` l.156–161 hard-codes the gate off for `audio_caption`; turning it on re-runs Qwen3.8-27B (H200/A100), and a fresh gate pass differs by ±1 picture per card class (NIGHT_REPORT §8), so spans would not be identical. `judge_direct.py` l.87/135 and `score_per_sound.load_pictures` l.174 key the text-tag path on the literal name `audio_caption`; `judge_direct.py` l.123 uses one `--tag` for all systems. Replacement:

> **D5 gated text tags on DEV — derived, not re-rendered.** For each DEV clip copy `protocol_proposed_dev_monocap_v31/<clip>/augmentations.json` and `media.json` into `protocol_audio_caption_<newtag>/<clip>/` (no images), and symlink `protocol_proposed_<newtag>` → `protocol_proposed_dev_monocap_v31`. The tags share ours' gate decisions and spans exactly; no GPU, no new gate pass, the ungated render is untouched. Run `judge_trust.py` B1/B2 on `protocol_audio_caption_<newtag>`, then `judge_direct.py --tag <newtag> --systems proposed,audio_caption`. Declared: no decision hangs on it; the judge table shows both text rows (ungated = the baseline, derived-gated = the modality comparison); expected tie (MDE ≈ 0.5). The same derivation may be applied to the TEST render in B.6 for free, scored only under the same yes.

**Sealed things / TEST numbers in B:** none touched. **SIGN, with the B.4 amendment above.**

## P5 — round 3

**SIGN**, with my D2 dissent recorded as the plan states, and one amendment to B.6 so that the "yes/no" cannot become a look by accident (the 23 Sep accidental read came from a script that printed both halves).

**AMEND B.6 — replacement text (append after the committed sentence):**
> The TEST render jobs run with no scorer, judge or audit invoked on the TEST tag; `score_per_sound.py`, `arm_compare.py`, `judge_direct.py` and `timing_audit.py` are not pointed at it, and no log line prints a hit, miss, F1, P or cost for a TEST clip. Adam gives his yes/no before any such number exists, and it is written into `docs/prereg_v4.md` with a timestamp. If no, the renders are never scored in this thesis. If yes, the table is scored once, with the Holm family of A.2 and the per-category rows, and the read is added to the history table as the fifth. No other TEST number is read after it.

**On the three questions asked.** (1) Holm family of seven, all printed rows, two-sided p from `paired_ci` with the 1/2000 resolution stated, applied identically to the DEV like-for-like and the 23 Sep TEST look: right, and conservative in the direction an examiner wants; ΔR staying in the family is correct — its significant loss is part of the same trade and must not be dropped. (2) D2 wording: acceptable with the amendment above; the "whatever it shows" clause committed before rendering is what makes it defensible, and I would still footnote in the thesis that this is the fifth read. (3) Nothing in B touches a sealed thing: D1(b) and D5 are DEV-only; D8 reads subject texts the assistant wrote, no answers or arm key; D6 is descriptive on the calibration set; the D4 packet shows the second annotator TEST *clips*, not system output or scores, which is not a read. B.3 re-analysing the already-read 23 Sep TEST numbers under a correction declared in A.2 is not a new read, provided A.2 is committed before the table is computed.

