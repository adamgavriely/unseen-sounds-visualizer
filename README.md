# Visual Augmentation of Audio Semantics for Accessibility

MSc final project. A system that makes the **non-speech** information in a video's soundtrack
accessible to deaf and hard-of-hearing (DHH) viewers. Subtitles and sign language convey speech but
largely drop ambient and environmental sound — flowing water, footsteps, a distant siren, rain,
birds, a reacting crowd. This system detects that missing audio semantics and renders it as
**complementary visual augmentations displayed alongside the original video** (supplementing, not
replacing, the video or its captions).

**Core idea — cross-modal gating:** the *video* is also analyzed, so the system augments only what
the audio conveys but the picture does **not** already show (a siren whose source is off-screen is
worth showing; one already on screen is not).

No models are trained — the contribution is the task formulation, an inference pipeline that
orchestrates open-source foundation models, a curated benchmark, and an automatic evaluation
protocol.

> Full proposal: `Final_Project.pdf`. Living research/design record: **[`docs/project_notes.tex`](docs/project_notes.tex)**.

## Pipeline (7 stages)

| # | Stage | Role | Candidate models |
|---|-------|------|------------------|
| 1 | [Audio extraction](src/stage1_audio_extraction) | standardize audio | FFmpeg |
| 2 | [Video understanding](src/stage2_video_understanding) | what's already visible | **OWLv2** (also CLIP / SigLIP / Qwen2.5-VL) |
| 3 | [Speech recognition](src/stage3_speech_recognition) | transcribe speech (secondary) | **Whisper** |
| 4 | [Audio event detection](src/stage4_audio_event_detection) | non-speech sounds | **BEATs iter3+** (PANNs CNN14 as fallback) |
| 5 | [Cross-modal analysis](src/stage5_cross_modal_analysis) | **what to augment, and what to draw** | gate over stages 2 and 4 + **Qwen2.5-VL** reasoning |
| 6 | [Visual augmentation](src/stage6_visual_augmentation) | produce the visuals | **Openverse retrieval** (beat SDXL) |
| 7 | [Evaluation](src/stage7_evaluation) | VLM-describe -> LLM-reference -> independent LLM judge | **Qwen2.5-VL + Mistral-7B** |

Each stage is a self-contained module with its own README (purpose, I/O, models, status).

### What the panel shows, and when

A picture appears when its sound is heard and leaves when it stops. Its position is fixed for as
long as it is on screen -- moving targets cost a viewer who is already splitting attention between
the video and the panel -- but a cell with no sound in it is empty, not a dimmed leftover. The panel
is divided by how many sounds are heard *at once*, so three sounds that never overlap share one
full-size cell in turn.

Two rules keep it readable rather than twitchy: a picture stays up for at least 1.5 s even if the
detected span was shorter, and two bursts of the same sound less than 0.8 s apart are one appearance.

### How a sound gets a picture (or does not)

No stage consults a list of known sounds. Every judgement is a question asked at run time about the
sound that was actually detected, so an unfamiliar sound is handled like a familiar one.

1. **Is it already on screen?** The VLM is shown the frames spanning *that sound* and asked to name
   the thing making it, answering `nothing` if it cannot see one. A named object is then checked in
   words -- *does a cowboy hat make a Vehicle sound?* -- because what is in the frame is a question
   about pixels and what makes a sound is world knowledge. If the source is visible, the system stays
   silent.
2. **What should the picture be?** The scene is described once, as text; the depiction step is then
   text-only, with the sound as the subject and the scene as a modifier. The audio says *what* it is;
   the video only says *which kind* and *where*.
3. **Is it still about the sound?** The model is made to *choose*: given the depiction, and given
   the other sounds detected in this same clip plus "none of them", which sound does this picture
   show? A forced choice is a discrimination task, unlike the yes/no that once approved "Man on
   motorcycle drives past silver van" as a picture of laughter. A rejected depiction gets one retry
   with the scene withheld before falling back to the plain label.
4. **Is it one source or two?** The AudioSet ontology -- the taxonomy PANNs' own labels come from,
   vendored as `src/audioset_parents.json` -- says so directly: if one detected label is a more
   specific kind of another (*Giggle* under *Laughter*, *Hoot* under *Owl*), they are one source.
   Ancestor, not shared parent, or *Animal* would swallow a dog and a sheep together. No threshold,
   and it generalises as far as the detector does. Depiction similarity remains as a secondary
   signal for paraphrases the taxonomy cannot see.

## Status: complete and evaluated

The pipeline runs end to end, the 274-clip benchmark is frozen and labelled, and the automatic
evaluation protocol of proposal section 6.1 has been run over 100 clips under all three systems of
section 7. Every stage is real; nothing is a stub.

```bash
python main.py --input data/input/clip.mp4          # augment one video
```

### Results

Judge scores 0-4, 100 clips per system, 300 records. `silent` is how often a system showed nothing;
`right to be` is how often that was correct.

Two references are reported, because the choice of reference changes the numbers and neither
choice is obviously right.

**(a) Model-derived reference** -- as the proposal specifies: an LLM states what a hearing viewer
gets that a deaf viewer misses, from the detected audio and the visible scene.

| system | mean | >=3 | silent | right to be |
|---|---|---|---|---|
| blind audio-to-image | **2.94** | 84% | 12% | 100% |
| audio captioning | 2.68 | 57% | 12% | 100% |
| proposed (gated) | 2.41 | 70% | 31% | 39% |

**(b) Human-grounded reference** -- on clips the annotator marked `seen_ambient` or `no_ambient`,
nothing is missing, so showing nothing is the correct output.

| system | mean | >=3 | silent | right to be |
|---|---|---|---|---|
| blind audio-to-image | **3.09** | 91% | 12% | 100% |
| proposed (gated) | 2.94 | 85% | 31% | 71% |
| audio captioning | 2.79 | 67% | 12% | 100% |

Under (a) the model-derived reference reports a sound as missing even when its source is plainly on
screen, so a system that correctly stays silent scores 0. (b) fixes that using the annotator's own
labels -- the only non-circular source, since grounding it in the system's own visibility output
would grade the gate against its own decision. The correction is worth +0.53 to the gated system and
+0.11 / +0.15 to the baselines; that asymmetry is the arithmetic consequence of the gated system
abstaining on 31% of clips against the baselines' 12%, and is stated wherever the number appears.

**The gate does not beat direct audio-to-image generation under either reference.** The useful
result is the per-category split:

| where the correct output is... | proposed | blind | |
|---|---|---|---|
| nothing (`seen_ambient`) | **3.36** | 3.24 | gate wins |
| nothing (`no_ambient`) | **3.48** | 3.28 | gate wins |
| something (`unseen_ambient`) | 2.64 | **3.04** | gate loses |
| something (`mixed`) | 2.28 | **2.80** | gate loses |

**The gate's suppression is worth having; its selection is not.** The two categories it wins are 186
of the 274 labelled clips. Where it loses, the cause is Stage 2 answering *object presence* when the
task needs *event visibility*: a police car visible in frame makes the system suppress a siren whose
off-screen companions are the reason the clip was tagged `mixed`.

Supporting experiments:

| experiment | outcome |
|---|---|
| judge reliability (2nd, unrelated judge) | kappa 0.753; the blind-proposed gap is **+0.53 under both judges** |
| generator ablation (SDXL vs retrieval) | **retrieval wins** (+0.18 blind, +0.17 gated, +0.00 caption control) |
| threshold tuning on a held-out half | **no optimism** -- test F1 exceeds train F1 in 4 of 5 splits |
| four Stage-2 backends | within ~2 points of each other; architecture does not decide this task |

All result files are in [`benchmark/results/`](benchmark/results); the full write-up, including the
evaluation failures found along the way, is in [`docs/project_notes.tex`](docs/project_notes.tex).

### Reproducing the evaluation

Needs a GPU with 24 GB (or set `GEN=retrieve`, which scores better anyway and needs none):

```bash
bash slurm/sync_data.sh --code-only       # push code to the cluster
LIMIT=100 bash slurm/submit_chain.sh      # pilot -> gate -> main run -> judge2 + ablation
```

The chain gates itself on a pilot and aborts any phase where more than a quarter of clips fail; see
[`slurm/RUNBOOK.md`](slurm/RUNBOOK.md). Individual pieces:

```bash
python -m benchmark.run_protocol --limit 12        # the protocol, end to end
python -m benchmark.split_eval                     # train/test threshold honesty check
python -m benchmark.select_demo --from-protocol    # rank clips for demo videos
python scripts/compare_runs.py protocol_results.json protocol_results_judge2.json
```

## Repository layout

```
docs/project_notes.tex        # living record: lit review, design, results, decisions
docs/PLAN.md                  # plan and current state
src/stage1..stage7/           # the 7 pipeline stages, all implemented
benchmark/                    # tagger, protocol runner, metrics, 274 labels
benchmark/results/            # every results file from the evaluation
slurm/                        # cluster jobs; RUNBOOK.md is the entry point
scripts/                      # sourcing, screening and analysis utilities
data/input|work|output/       # clips and intermediates (gitignored)
Final_Project.pdf             # the approved proposal
```

## Environment

- conda env `msproj` (Python 3.11); interpreter `C:\Users\adamg\anaconda3\envs\msproj\python.exe`.
- FFmpeg at `C:\ffmpeg\bin`.
- GPU work runs on the BIU Slurm cluster (`L4-12h` partition); `slurm/setup_env.sh` builds the
  environment there. Local CPU is enough for everything except the 7B models.
