# Executive summary — Visual Augmentation of Audio Semantics for Accessibility

*For someone taking over this project today (updated 2026-09-14, 18:00). Plain language,
every claim backed by a file you can open. The detailed record is `docs/project_notes.tex`
(the dated lab notebook), `docs/literature_review.tex` (background), `docs/report/report.pdf`
(the thesis draft, 14 pages, complete), `LIMITATIONS.md` (what the numbers do not support) and
`docs/failure_catalogue.md` (every failure seen, with its status).*

---

## 1. The problem in one paragraph

A deaf viewer gets a video's speech through subtitles but not the rest of its soundtrack: a
siren behind the camera, a dog barking in the next room, glass breaking off screen. Caption
guidelines even tell captioners to skip any sound the picture already shows, and in practice
most ambient sound is never captioned. This project built a system that hears a video's
ambient sounds, decides for each whether its source is **already visible on screen**, and —
only when it is not — shows a small picture of that sound beside the video while it lasts.
The "only when not visible" rule is called the **gate**; it is the project's central idea and
the thing the evaluation tests.

## 2. Background: what exists and why it is not enough

- **Subtitles/SDH** carry speech well and non-speech sound badly and inconsistently; sign
  language is richer but scarce and not automatable.
- **Sound-awareness tools for deaf users** (HomeSound, SoundWatch, ProtoSound) recognise ~20
  everyday sounds in the real world; users prefer icons/pictures to text. They do not
  address media.
- **Audio event detection** is mature: models trained on AudioSet (527 sound classes) name
  what is in a soundtrack. We use **BEATs** (PANNs was first and was replaced).
- **Vision-language models (VLMs)** answer open questions about frames — the only way to ask
  "is the thing making this sound visible?" without a fixed list of objects.
- **Audio-to-image generation** (Sound2Scene etc.) turns sound into a picture without looking
  at the video. Those systems are our **blind baseline**: "draw everything you hear".
- Nobody had combined detect → reason about heard-but-not-seen → show a picture. That
  combination, its benchmark and its evaluation are the contribution.

Constraints from the supervisor: **no training** (glue existing open models), **minimum human
involvement** (decisions by models, not hand-written rules), finish this week.

## 3. What the system does (seven stages, `src/`)

| stage | what | model |
|---|---|---|
| 1 | extract audio | FFmpeg |
| 2 | quick check of which objects appear anywhere in the clip; stage 5 makes the actual visibility decision | OWLv2 on the cluster (SigLIP on a CPU-only machine) |
| 3 | transcribe speech — context for the gate only, never drawn | Whisper |
| 4 | detect ambient sounds with start/end times | BEATs, 527 AudioSet labels, bar 0.35 |
| 5 | **reasoning**: is the source visible? what kind of sound? what should the picture show? which sounds are one source? | Qwen2.5-VL-7B |
| 6 | draw the picture, compose the side panel | FLUX.1-schnell |
| 7 | evaluate | see §6 |

Behaviours, all decided by models at run time (`config.py` holds every knob, commented):
- **Visibility**: per sound, per 5-second stretch, three differently phrased questions to the
  VLM (name the source; "is it visibly happening", asked in both option orders; describe the
  frames) — majority vote; silent only if visible in *every* stretch. "Visible" means *seen
  making the sound*: a baby held in arms is not visibly crying; a woman laughing is.
- **Timing**: a sound's start is stamped when its score begins to rise, not when it crosses
  the bar ("hysteresis"; cut timing error 0.43 → 0.31 s on DCASE gold); a second "occlusion"
  adjustment (`ONSET_CAM`) was measured to add nothing. **One continuous sound → one
  continuous picture**, held ≥1.5 s; the panel is empty when nothing is heard.
- **The picture shows the event** ("glass shattering"), built from the label, the frames and a
  short place, never from dialogue or example sentences; validated by a forced choice.
- **One picture per source**: the AudioSet ontology merges "Laughter"/"Giggle" first; a text-
  similarity merge (`DEDUP_SIM = 0.80`, only for sounds that overlap in time) still catches
  paraphrases the ontology cannot see and is known to be coarse (catalogue #14).
- **Speech as context**: people reacting to a sound raise its priority; never overrides
  visibility.

## 4. How it was built, and what was learned

Almost every early fix was wrong on the next clip. The reusable lessons:
1. **Fixed vocabularies fail** (a 30-entry "visible concepts" list could never mark laughter
   visible). Every table became a question to a model.
2. **Ask for a name, not a yes**: yes/no questions collect agreement; every two-way question
   is asked in both orderings because the 7B model prefers option (b) 22 times in 23.
3. **Read structured answers instead of inferring them**: the AudioSet ontology solved
   deduplication after three cleverer attempts failed.
4. **Set thresholds once on a dev split** (`docs/plan_robustness.md`): every failure is
   logged in the catalogue; a change needs three cases of one pattern or a dev-split metric.
5. **Evaluation code produces plausible wrong numbers**: an empty result scored 2/4; a
   similarity returned 0.0 for everything; an audio model received no audio and said "none"
   100 times; a job step died on a Windows line ending. Each found by reading logs; each guarded.

## 5. The benchmark (`benchmark/tags.json`, clips in `data/input/benchmark/`)

- **274 clips**, hand-labelled by Adam into four scenarios: **unseen** (off-screen ambient
  sound — picture due), **mixed**, **seen** (source on screen — no picture due), **no ambient**
  (speech/music only).
- **Test set**: 100 clips, 25 of each scenario. **Dev split**: the other 174 labelled clips
  (note: all 25 mixed clips are in test, none in dev).
- **Yield**: the proposal asked for 300; usable clips were 15–20% of candidates from four
  sourcing strategies — reported as a finding.
- **Label stability** κ = 0.60 (Adam's agreement with his own earlier labels on blind
  re-labelling — "moderate"; the instability sits in the positive class the gate is judged on).
  The first 209 tags are the unbiased sample; later candidates were model-pre-screened
  (`benchmark/screen.py`).
- **External check on DCASE 2025 Task 3** (30k clips with on/off-screen labels; 222 audio
  files fetched to `data/dcase2025_task3/`): on 258 gold events the visibility question agrees
  66.7% — it keeps 97% of off-screen sounds but silences only 35% of on-screen ones. Caveat:
  DCASE frames are cut from 360° captures, so "on screen" there is defined by a chosen field
  of view.

## 6. How it is evaluated (this decides what the numbers mean)

Per clip and system:
1. A VLM describes the augmented video ("a viewer would learn that…").
2. A **reference** sentence says what a hearing viewer gets that a deaf viewer misses.
3. A judge (Mistral-7B) scores the match 0–4. Showing nothing when something was missing = 0;
   nothing when nothing was missing = 4.

- **Baselines**: **blind** (draw every sound, never look) and **caption** (text instead of a
  picture).
- **Judge dependence**: a second judge reproduces the gated-vs-blind gap (κ = 0.753) but
  moves the caption baseline by +0.61 — read gated-vs-blind as robust, gated-vs-caption as a
  property of the judge.
- **Stats**: `python scripts/paired_stats.py v3 v3_grounded` prints everything, paired with
  bootstrap CIs, pooled and split by "picture needed" (unseen+mixed) vs "not needed"
  (seen+no-ambient).

Three references exist; none is neutral, and this is itself a result:
- **model-derived** (the proposal's): written by the system's own models from its own
  detections → circular; favours the system that repeats the detector by ~0.5.
- **human-grounded**: corrected by the label → the only one that scores the gate; needs labels.
- **independent** (built 2026-09-14: Qwen2-Audio + CLAP verification + Idefics3 +
  Phi-3.5-mini writer; Llama-3.1 was gated when it ran, access granted since, not re-run):
  removes circularity, but its "is the source visible" decision is at chance (55%) under two
  variants, so it rewards showing; its sentence-rejection threshold τ = 0.50 was never
  calibrated. Reported; not used for v3.

The rubric is asymmetric: a wrongly withheld picture costs ~4 points, a redundant one ~1, so
a gate must be right ~80% of the time when silent just to break even.

## 7. Results (v3 = the shipped system; 100 test clips; 0–4)

| reference | gated | blind | caption | gated − blind (95% CI) |
|---|---|---|---|---|
| model-derived | 2.64 | **3.04** | 2.79 | −0.40 [−0.66, −0.16] |
| **human-grounded** | **3.17** | 3.16 | 2.89 | **+0.01 [−0.15, +0.18]** — tie (21 wins / 64 ties / 15 losses) |

By scenario (grounded reference):
- **no picture due**: gated 3.56 vs blind 3.30 → +0.26 [+0.12, +0.40] — the gate wins.
- **picture due**: gated 2.78 vs blind 3.02 → −0.24 [−0.54, +0.04] — the gate loses.
- Gated beats caption by +0.28. A perfect gate would score 3.25.

In words: *the system does the right thing when nothing is missing, and still sometimes stays
silent when it should not.* The earlier configuration (v2) lost clearly under every reference
(−0.24 to −0.70), and on v2 even a perfect gate would have beaten blind by at most +0.05 —
which is why the week's work went into the detector and visibility stages (BEATs, per-sound
visibility, event depictions, ontology dedup), and that closed the gap.

**Where the remaining errors come from** (read from cached decisions, no GPU): of 11
wrongly-silenced test clips, 7 are the **detector** (ambient sound under speech/music never
reached the bar), 4 are label ambiguity (cars visible vs off-screen traffic), 0 are VLM
timing. On dev, ~80 of 100 redundant pictures are confident detector **phantoms** (whale at a
Christmas market) the VLM rightly cannot silence. So the detector, not the visibility
judgment, bounds the result.

**What was tried against that, and why it stopped:**
- Gate threshold sweep on dev (16 settings, rubric-derived costs): the shipped setting is
  already optimal; no v4 (`benchmark/gate_setting.json`).
- CLAP second opinion on every detection, calibrated on DCASE gold with a bar declared
  before running (95% recall at k ≤ 30): reached 50% → unusable. Reported as a measured
  limitation of zero-shot audio-text models (`benchmark/clap_setting.json`).
- Demucs separation before detection: assessed by two independent reviews and not run —
  it removes speech only (4 of the 7 masked misses are under music) and merging scores can
  only add phantoms. First future-work item, with acceptance bars written in the report.

Other measured results: onset timing on DCASE gold — of 255 gold onsets only 66 are matched
by a detection at the 0.35 bar (detector recall on 5 s clips is low); on those, hysteresis cut
the error from 0.43 s to 0.31 s and the occlusion refinement adds nothing
(`benchmark/eval_dcase_onset.json`);
generation vs stock-photo retrieval (+0.17 for retrieval, but 55% restrictive licences).

## 8. What is done, what is open

Done: pipeline, benchmark, evaluation protocol with three references, all runs (v2, v3),
onset and visibility evaluations on DCASE gold, the diagnosis, the report (14 pages, no
placeholders), LIMITATIONS.md, this summary. All on `main`, pushed, including the cached
gate votes and description caches the diagnosis depends on.

Open decisions for Adam: keep or drop the occlusion onset step (`ONSET_CAM`; no measurable
effect); whether to run the independent reference for v3 (it cannot score the gate);
polish/submit the report.

Future work, in order (also in the report): a detector-only benchmark on the labelled clips
(recall of off-screen events vs phantom rate) and against it a speech-aware detector (Demucs
residual, with the declared bars) or a polyphonic detector; an open-vocabulary audio model
where AudioSet has no word ("phone alert"); a second annotator; a study with deaf viewers.

## 9. Where everything is, and how to run it

- Run one video: `python main.py <video>` (see `README.md`); all knobs in `config.py`.
- Results: `benchmark/protocol_results_<tag>.json` (v2, v2_grounded, v2_indep_list, v3,
  v3_grounded), `benchmark/paired_stats.json`; cached gate votes `benchmark/gate_votes/`.
- Cluster (BIU Slurm; VPN is F5 at access.biu.ac.il, resource AC-Users; `slurm/RUNBOOK.md`):
  `bash slurm/sync_data.sh --code-only` uploads code (and strips Windows line endings, which
  otherwise kill job steps). You need your own Hugging Face token exported in `~/.bashrc` on
  the cluster (FLUX, Llama-3.1 are gated — accept their licences on HF first).
- The full experiment: `sbatch slurm/job_v3.sh` → four stamped passes of
  `slurm/job_protocol.sh` (render / describe / judge / grounded judge; ~10 h on one L4;
  resumable from stamps — resubmit, never restart) → `benchmark/protocol_results_v3*.json` →
  `python scripts/paired_stats.py v3 v3_grounded`.
- Documents: `docs/report/` (thesis draft), `docs/project_notes.tex` (everything, dated),
  `docs/night_report_2026-09-14.md` (the overnight run), `docs/failure_catalogue.md`,
  `docs/plan_robustness.md` (process rules), `LIMITATIONS.md`.
