# Scope v2 — the supervisor's five points, costed

*Adam Gavriely, 17 September 2026, after the first supervisor meeting. Prepared with a
five-reviewer Fable panel. The current thesis (v3: tie with the blind baseline, detector
bottleneck measured, three automatic references failed) stays as the anchor; every new item
below is additive, has a pass bar declared before it runs, and can be dropped cleanly.*

## The question that decides everything

**"Given the wider scope, what submission date do you have in mind — two weeks, six weeks,
or end of semester — and is the current thesis acceptable as a fallback if the new work does
not land?"**

| answer | in | out | headline becomes |
|---|---|---|---|
| **A · 2 weeks** | 5 (gold set, ≥ 40 clips, two lab annotators) → 3 as a demo only (one or two H3 clips) | model swaps, audio-image scoring, H3 as a scored baseline | "v3 tie, confirmed on a multi-annotator gold set; H3 shown as the future direction" |
| **B · 6 weeks** (recommended) | 5 → 1 (one swap at a time: detector first, then VLM, then image model) → 3 as a 5-clip demo → 4 as a dev check | video output; H3 as a scored baseline | "which stage matters: a SOTA detector swap moves the score by X, the generator by Y" |
| **C · end of semester** | all five: 5 → 1 → 3 demo → 3 as a scored baseline (if the judge can score video) → 4 | — | "generating the unseen: a gated audio-to-picture system vs a conditioned video baseline, on a gold set" |

Plan A is the written fallback for B and C.

## The five points

| # | point | what it means concretely | cost | value for the thesis | verdict |
|---|---|---|---|---|---|
| 1 | **SOTA models, bigger GPUs, multi-GPU** | detector: **FLAM** (Adobe, frame-level, open-vocabulary, < 4 GB) or DASM; VLM: **Qwen3.5-27B** (54 GB, one A100-80 / RTX Pro 6000); images: **Z-Image-Turbo** (6 B, drop-in) or FLUX.2 [dev] (needs an H200 or FP8). Each swap = one day on one big GPU + a 10-GPU-h re-run of the 100-clip protocol if the dev bar passes. Multi-GPU helps as clip-parallel Slurm arrays (8 clips at once), not model sharding. Second conda env needed. | 1 day + 10 GPU-h per swap | **detector: fair and real** — it is the measured bottleneck; VLM: half fair (one day); image model: cosmetic for the claims, good for the demo | do the detector first |
| 2 | **Condition a generator on the video + audio, "generate what is not seen"** | with MiniMax-H3 (33 B, open weights, text + up to 3 reference videos with sound + 3 audio clips + 9 images in; 768p video with stereo audio out): (a) a side clip from our audio + text prompt = a moving version of our panel — a **legitimate baseline**; (b) outpainting the scene — H3 does not do this; (c) regenerating the whole video — a different product, not scorable by our judge | see 3 | (a) yes, as "video instead of a picture" | (a) only |
| 3 | **ComfyUI demo on our GPUs** | ComfyUI ≥ 0.30 has native H3 nodes (Ref2VA); int8 stack ≈ 42.5 GB fits one **RTX Pro 6000 (96 GB)**; ~3–5 min per 5-s clip; headless on a Slurm node via its API; weights 45 GB (cache has 70 GB free); its own conda env (torch newer than ours — the main setup risk) | 1 day setup + 1 day for 5 clips | a working demonstration the supervisor asked for; the evaluation question "does a moving picture convey the missing sound better than a still?" | 5-clip demo with a bar (H3 ≥ FLUX panel + 0.5/4 on the same judge) before any 100-clip baseline |
| 4 | **Audio-to-image per second as a score** | audio caption → image every second, compared with the frame; low similarity = "heard but not seen" | 3 GPU-h dev check | it re-asks the visibility question through a hallucinated picture, and image similarity is dominated by style and viewpoint; three references already failed at exactly this point; as a *reference* it is circular (rewards the blind baseline); as a *system* it is blind at per-second cost | run only if he wants it on record; expected to fail; write as the fourth reference attempt |
| 5 | **Gold set annotated by lab members** | two annotators (not Adam), all 100 test clips, ~2.5 min each ≈ 8.5 annotator-hours; per sound: family, 5-s stretch, "masked by speech/music", "source visible"; per clip: picture due; **one human sentence "what a hearing viewer gets that a deaf viewer misses"** — the item that replaces the three failed references; κ ≥ 0.6 target; Adam adjudicates | 1 day tool + 1 h calibration + 2 × 2 h labelling + 1 day analysis | **highest value per hour**: every plan needs it; makes any new model's score trustworthy; lets the judge see detector misses | **first, whatever the deadline** |

## Order of work (dependencies)

1. Tag the current state as **v3** — the reference row in every future table.
2. **Gold set** (5): tool, 10-clip calibration, labelling, agreement, headline re-run on human labels.
3. **Detector swap** (1): FLAM on the dev split, bar = masked-miss recall up ≥ 15 points at BEATs' false-alarm rate on DCASE gold; then the 100-clip re-run.
4. **H3 demo** (3): environment, weights, 5 clips, side-by-side video, workflow JSON in the repo.
5. VLM swap, image-model swap (1) — each with its bar.
6. H3 as a scored baseline (2) — only if the judge proves it can score video on dev.
7. Audio-image-per-second (4) — dev check only.

## What must not be lost

Pre-declared bars; the negative findings (seven detector fixes, three references); the
human-grounded headline as the anchor; the cost axis (panel-on time); v3 in every table.

## Next 48 hours, whatever he answers

Freeze and tag v3; build the annotation web tool and run the 10-clip calibration (no GPU,
needed by every plan); set up the second conda env for FLAM (the first swap) on an A100.

## The risk

Scope creep without a fallback — chasing H3 and swaps while the finished thesis rots. Guard:
get it in writing that the tagged v3 thesis is submittable as-is; each new point is an
additive chapter with a kill criterion.
