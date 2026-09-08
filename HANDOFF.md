# Handoff briefing — MSc project "Visual Augmentation of Audio Semantics for Accessibility"

Paste this into a fresh session. Everything below is current as of 2026-09-08.

---

## 1. What the project is

CS MSc project (Adam Gavriely, Bar-Ilan). Authoritative spec is `Final_Project.pdf` in the
repo root — **read it before making plans**; it is easy to drift from.

**Goal.** Make NON-SPEECH ambient sound in a video accessible to deaf/hard-of-hearing
viewers by (a) detecting ambient sounds, (b) deciding which sounds' sources are NOT already
visible on screen ("cross-modal gating"), and (c) showing a complementary generated image
ALONGSIDE the original video (never replacing it).

**Hard constraint:** train nothing. Glue off-the-shelf foundation models.

**Proposal deliverables (§1.3):**
1. Formulate the semantic audio augmentation task — ✅ done
2. Inference pipeline from existing foundation models — ✅ done (7 stages)
3. Evaluation benchmark (~300 clips, 10–20 s) — 🟡 274 labelled
4. **Automatic evaluation protocol (§6.1)** — ✅ BUILT (`src/stage7_evaluation/protocol.py`,
   `benchmark/run_protocol.py`, `slurm/job_protocol.sh`) but ❌ **never successfully run** —
   the pilot job vanished without writing a log. Unblocking it is the current priority.
5. Comparison with baselines (§7) — ❌ blocked on 4 actually running

**Research questions (§2), status:**
- Which audio information contributes most to scene understanding? — ❌ open
- What semantic granularity for ambient sounds? — ✅ answered (measured; `docs/project_notes.tex` §Sound granularity)
- Can generated augmentations improve accessibility beyond subtitles? — ❌ open
- How should such systems be evaluated? — ❌ open (this is deliverable 4)
- How does the pipeline compare to audio-to-visual generation? — ❌ open

---

## 2. The pipeline (all implemented, `src/`)

| Stage | Module | State |
|---|---|---|
| 1 Audio extraction | `stage1_audio_extraction` | FFmpeg, real |
| 2 Video understanding | `stage2_video_understanding` | 4 backends: `clip`, `siglip` (default), `owlv2`, `vlm` (Qwen2.5-VL). Switch via `config.VIDEO_BACKEND` |
| 3 Speech recognition | `stage3_speech_recognition` | faster-whisper, real |
| 4 Audio event detection | `stage4_audio_event_detection` | PANNs CNN14 SED, real |
| 5 Cross-modal gate | `stage5_cross_modal_analysis` | ⚠️ **rule-based stub**; proposal §4.5 specifies an LLM (Qwen3/Llama 3.1) |
| 6 Visual augmentation | `stage6_visual_augmentation` | `retrieve` (Openverse), `diffusion` (SDXL), `placeholder`; compositor renders a side panel with stable per-sound slots |
| 7 Evaluation | `stage7_evaluation` | `gating_accuracy()` + the §6.1 protocol in `protocol.py` (separate judge model, refuses to run if describer == judge). Built, not yet run |

Entry point: `python main.py --input <video>`. Config in `config.py`.

---

## 3. Benchmark

`data/input/benchmark/<category>/` (gitignored). Labels in `benchmark/tags.json`
(committed), keys are `"<folder>/<file>"`.

**274 usable labels:** 45 unseen_ambient · 25 mixed · 128 seen_ambient · 58 no_ambient.
Plus 46 `bad` (technically defective) and 429 `_dropped` (fine clips, not a needed case).

Tagging tool: `tag_videos.bat` → `benchmark/tagger.py` (web app on :8000).
**Never launch the tagger from inside Claude Code** — it froze the app once. Adam runs it.

**Sourcing is FROZEN. Do not source more clips.** Yield was 15–20% usable positives across
four independent strategies (Wikimedia/keyword scraping, AudioSet temporally-strong,
curated film scenes, UnAV-100 overlap moments) and did not improve with three stages of
automated filtering. Scripts live in `scripts/source_*.py` if ever needed again.

---

## 4. Results so far (the important part)

### Four Stage-2 backends, identical clips and labels

| backend | architecture | acc | F1 | event-level AUROC |
|---|---|---|---|---|
| CLIP | contrastive embedding | 49.3% | 48.0% | (cache predates score logging) |
| SigLIP | sigmoid embedding | 49.6% | 44.6% | **0.647** |
| OWLv2 | open-vocabulary detection | 50.4% | 46.4% | **0.616** |
| Qwen2.5-VL 7B | generative VLM | 51.2% | 47.7% | (being re-run) |

**Two percentage points separate all four.** Architecture does not matter here.

### Trivial baselines — must be reported, an examiner will compute them

| system | acc | F1 |
|---|---|---|
| always augment (**= the blind audio-to-image baseline the thesis claims to beat**) | 26.6% | 42.1% |
| always stay silent | **73.4%** | 0.0% |

Accuracy is dominated by the 73% negative class, so "always silent" wins on accuracy while
being useless for accessibility (it never augments anything). **F1 against the blind
baseline is the meaningful comparison**, and the gate wins it by ~+5 F1.

### Three findings, written up in `docs/project_notes.tex` §Results and Findings

1. **Object presence ≠ event visibility (the substantive finding).** A blind re-adjudication
   recovered Adam's actual criterion in his own words: *"only building seen but bell is
   heard"*, *"alarm ringing and is seen but the 'ring' is not indicated"*, *"bird is seen,
   not singing"*. He judges whether the sound-producing EVENT is visible; every model
   answers whether the OBJECT is visible. This explains the flat 49–51%, the annotator's own
   inconsistency, and why better grounding cannot fix it.
2. **Label stability:** Cohen's κ = 0.60, 78% self-agreement. Instability is concentrated in
   the positive class (seen_ambient reproduced 19/20; unseen_ambient only 12/20).
3. **Dropped clips are genuine negatives**, not withheld positives (only 1 of 20 converted),
   so the benchmark cannot be completed from material already collected.

Tools: `benchmark/evaluate.py` (gating accuracy, confusion, threshold sweep, per-backend
caches), `benchmark/auroc.py` (threshold-free AUROC + trivial baselines),
`benchmark/readjudicate.py` + `scripts/readjudicate_report.py` (label-stability study).

---

## 5. Infrastructure

**Local:** `P:\MscProj`, branch `main`, remote `github.com/adamgavriely/MscFinalProject`
(private). Python `C:\Users\adamg\anaconda3\envs\msproj\python.exe`, ffmpeg on
`C:\ffmpeg\bin`, MiKTeX pdflatex at
`C:\Users\adamg\AppData\Local\Programs\MiKTeX\miktex\bin\x64\`.

**Cluster (BIU Slurm):** `ssh adamg@slurm-login1.lnx.biu.ac.il`, requires the F5 VPN at
access.biu.ac.il → resource AC-Users. Env built by `slurm/setup_env.sh` (conda-forge only —
Anaconda's default channels demand a commercial ToS). Runbook: `slurm/RUNBOOK.md`.

Gotchas that cost real time:
- Git Bash has **no rsync** → `bash slurm/sync_data.sh --code-only` for code (seconds);
  the full form re-tars ~2 GB of clips.
- Git checks `.sh` out as CRLF on Windows and bash on the cluster then dies with
  `set: pipefail: invalid option name`. `.gitattributes` pins `*.sh` to LF; if a synced
  script fails, run `sed -i 's/\r$//' slurm/*.sh`.
- Never clone on the cluster (private repo, GitHub refuses password auth) — the sync
  script uploads code over ssh instead.
- Qwen2.5-VL (~16 GB) + SDXL (~7 GB) will **not** both fit on a 23 GB L4. The Stage-7
  protocol therefore runs as two passes (`--phase describe`, then `--phase judge`) so
  the describer and the judge are never resident together.
- A partition state ending in `-` (e.g. `mixed-`) means **draining**: a job submitted
  there pends forever with no log. `generic` usually has idle nodes but they are
  **GTX 1080 Ti with only 11 GB** — too small for Qwen2.5-VL 7B, which dies during
  model load leaving an empty `.err` and a truncated `.out`. Use `L4-4h` (23 GB),
  `A100-4h` or `L40s-4h` for anything with a 7B model. `job_protocol.sh` now
  preflights the card size and exits with a clear message instead.
- H200 partitions are often drained; `L40s-4h` and `generic` usually have idle nodes.
  Check with `sinfo -o "%15P %5a %8T %6D %N"`.

Jobs: `slurm/job_smoke.sh` (sanity, ~2 min), `slurm/job_eval_vlm.sh` (four-backend
comparison + AUROC), `slurm/job_diffusion.sh` (12 SDXL demo renders from
`benchmark/demo_set.json`).

---

## 6. Running right now

- `job_diffusion.sh` — 12 demo videos → `data/output/*_augmented.mp4`, fetch with
  `bash slurm/fetch_results.sh` from the PC
- `job_eval_vlm.sh` — re-run to fill the CLIP and VLM AUROC rows

---

## 7. Next work, in priority order

**See `docs/PLAN.md` for the dated two-week plan.**

1. **Run** the Stage 7 protocol (already built to proposal §6.1):
   analyse the original → generate augmentations → a VLM describes the augmented scene →
   an LLM derives a semantic reference from the original multimodal input → an *independent*
   LLM judge scores semantic consistency. `config.JUDGE_MODEL` is set to
   `mistralai/Mistral-7B-Instruct-v0.3` — deliberately a different family from the
   describing VLM, since a model scoring its own output measures self-consistency.
   → deliverable 4 + RQ "how should these systems be evaluated"
2. **Run the two §7 baselines through that same protocol:** direct audio-to-image
   (blind generation, i.e. the gate disabled) and audio-captioning-only.
   → deliverable 5 + RQ "how does the pipeline compare"
3. **Promote Stage 5 from the rule-based stub to a grounded LLM gate** (§4.5). Feed it the
   Stage-2 entity list, Stage-3 transcript and Stage-4 events as structured text — never raw
   media — to curb hallucination.
4. Optional, high value for RQ3: a small DHH user study (≈10 participants × 10 clips)
   comparing subtitles alone vs subtitles + augmentations. **Check the ethics/IRB timeline
   early** — if approval takes weeks the option disappears.

**Do not:** source more clips, build more tagging tooling, or chase gating accuracy as the
headline metric. All three have been over-invested in; deliverable 4 is what is missing.

---

## 8. How Adam wants to be worked with

- Be concise; answer first, expand only when asked.
- Work autonomously, take trivial decisions, only block on genuinely crucial ones.
- Commit after each meaningful change and push.
- Aim at the full proposal targets; treat time as the flexible variable, not scope.
- He catches real bugs from watching outputs (the CLIP softmax-dilution bug, the
  unfalsifiable-visibility bug, the black demo panels all came from his observations) —
  take his reports seriously and verify claims rather than agreeing reflexively.
