# Night report, 2026-09-14 (01:00–07:00)

Branch `night-2026-09-14`, 25 commits, all pushed. Every non-trivial decision was put to
Fable first and taken only with its support; the decisions are named below.
PR: https://github.com/adamgavriely/MscFinalProject/pull/new/night-2026-09-14

## Results

**v2 (the 2026-09-12 configuration) under three references, 100 clips, Mistral judge**

| reference | proposed | blind a2i | caption | proposed − blind (95% CI) | oracle gate |
|---|---|---|---|---|---|
| model-derived (`v2`) | 2.27 | **2.97** | 2.68 | −0.70 [−0.98, −0.43] | 2.89 |
| human-grounded (`v2_grounded`) | 2.90 | **3.14** | 2.79 | −0.24 [−0.47, −0.02] | 3.19 |
| independent (`v2_indep_list`) | 2.41 | **2.96** | 2.61 | −0.55 [−0.82, −0.30] | 2.88 |

Blind wins under every reference; the circular reference was not what held the gate back.
The oracle row says a perfect gate would beat blind by at most +0.05 on this benchmark.

**The independent reference cannot score the gate.** Its "nothing beyond the picture" decision
agrees with the human tag on 55/100 clips — chance — under both visibility variants tried
(Idefics3 list + MiniLM match; per-sound forced question to Idefics3). Cause: "silent only if
every verified claim is visible" meets 5–8 audio-LM claims per clip, several never visible
("video game sound", "wind noise (microphone)", "clock ticking"). It is biased toward
"something is missing" and rewards blind on seen-ambient clips (3.08 vs 2.48). Reported as such
in the report and LIMITATIONS; the grounded reference remains the only one that scores the gate.

**DCASE onset evaluation (255 gold onsets, 9 classes, plan item C)**

| stamping | matched | mean signed | MAE | ≤0.25 s | ≤0.5 s |
|---|---|---|---|---|---|
| sliding-window stamp | 50 | +0.29 | 0.43 | 44% | 68% |
| + hysteresis | 66 | +0.22 | 0.35 | 58% | 74% |
| + hysteresis + occlusion (ships) | 66 | +0.16 | 0.35 | 45% | 80% |

Hysteresis is the gain (same 50 events: MAE 0.43 → 0.31, +16 events matched). Occlusion is a
wash: better on 25 / worse on 39, paired MAE difference 0.00 [−0.11, +0.10]. The 7-onset
anecdote (+0.9 → +0.3) overstated it and is retired. Only 66/255 gold events are matched at the
0.35 bar — detector recall on 5 s clips is the larger limit.

**v3** (job 28942456): proposed renders 80/100 after 3.2 h, 3 OOM failures (the describe phase
re-renders them; an OOM retry is in the code for the next submission). Expected to finish
~12:00. Nothing in v3's config or prompts was touched. Do not restart it; resubmit
`job_v3.sh` if it hits the 12 h wall (it resumes).

## What was built

- `docs/report/report.tex` (+pdf): technical report — Method, Benchmark, Evaluation and
  Results sections drafted from the notes with every number keyed to its result tag; v3 cells
  marked pending. `LIMITATIONS.md` at the root, seeded from the catalogue and extended tonight.
- `scripts/paired_stats.py` (A), `benchmark/eval_dcase_onset.py` + `slurm/job_dcase_onset.sh`
  (C; fetches only the 222 needed DCASE wavs from Zenodo by range request — the video files
  carry no audio), `scripts/ref_silence_check.py` (gate before judging),
  `slurm/job_ref_regate.sh` (re-verify/re-write/re-judge keeping previous variants).
- Independent reference fixes (all found by reading logs): transformers-5 renamed
  `audios`→`audio` and silently ignored it, so the audio LM heard nothing (100 × "none");
  CLAP outputs are ModelOutputs; Demucs pulls tonal sounds into the vocals stem (now stem ∪
  mix); the 2σ text-decoy gate left 58/100 references empty (now audio-decoy rank top-20 with
  cos>0, sensitivity table in LIMITATIONS); Llama-3.1 is gated for your account (403) →
  Phi-3.5-mini writer.
- `slurm/sync_data.sh` now strips CRLF on the remote after every sync (a mid-run sync killed
  a judge step). `run_protocol.py` retries a clip once after CUDA OOM.
- Freed 57 GB of unused cached checkpoints (SDXL, sdxl-turbo, dreamshaper, SD1.5, animatediff,
  clap-unfused) — disk was at 98%; Idefics3 and Phi downloads then fit. 27 GB free now.

## Decisions taken with Fable (in order)

1. Cancel/resubmit the reference job after the `audios` bug; add a first-clip sanity gate.
2. Listen to stem ∪ mix (option B, not dropping the "none" exit).
3. Delete six unused checkpoints; do not download the 10 GB DCASE audio zip at 98% disk —
   partial range fetch instead (Fable's suggestion, worked).
4. Submit the onset eval on a free node while v3 runs.
5. Verification gate: audio-decoy rank top-20 AND cos>0, chosen once on the
   empties-vs-no_ambient comparison; sensitivity table published.
6. Per-sound visibility question for the reference; judge only above 70% silence agreement —
   it scored 55%, so it was not judged; the list variant is what is reported.
7. Occlusion: keep for v3 (no config change); recommend below.

## TLDR

Under every reference, including the new independent one, the blind baseline beats the gated
system on v2 (−0.24 to −0.70); a perfect gate would gain at most +0.05. The independent
reference's silence decision is at chance, so only the human-grounded reference can score
the gate. Hysteresis fixed onset timing (MAE 0.43→0.31 s on 255 DCASE onsets); occlusion
adds nothing. Report skeleton + four drafted sections, LIMITATIONS.md, all committed.

## Actions (yours)

1. Merge `night-2026-09-14` (PR link above) or tell me what to change.
2. Occlusion: no measurable timing cost or gain. Drop it (`ONSET_CAM=False`) for simplicity,
   or keep it for the class-conditional onset rationale — either way the 7-onset number is gone.
3. Llama-3.1-8B writer: accept Meta's licence on the HF model page if you want it; otherwise
   Phi-3.5-mini stays and is stated.
4. When v3 finishes (~12:00): `python scripts/paired_stats.py v3 v3_grounded`; decide whether to
   run the independent reference for v3 at all (`TAG=v3 KEEP=none sbatch slurm/job_ref_regate.sh`
   after a reference pass) given it cannot score the gate.
5. Report: Introduction, Related work, Discussion (RQ1–5), Conclusion are still `\todo`; the
  rest is drafted. Bibliography to copy from the notes.
