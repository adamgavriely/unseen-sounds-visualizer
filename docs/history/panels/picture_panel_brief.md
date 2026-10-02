# Brief for the picture panel (5 reviewers, 3 rounds) — 2026-09-24

## The system
A no-training pipeline for deaf / hard-of-hearing (DHH) viewers: beside a video it shows a generated
picture for ambient sounds whose source is **off screen**. Stage 4 detects sounds (BEATs + FlexSED,
AudioSet's 527 labels). Stage 5 (VLM Qwen3.8-27B) decides visibility and writes a short `subject` for
each sound. Stage 6 draws `subject + ", plain white background, clearly visible"`. Code:
`src/labels.py` (consolidate_families, merge_by_label), `src/stage5_cross_modal_analysis/reason.py`
(DEPICT_PROMPT, DEPICT_PROMPT_V2, KIND_PROMPT, KIND_PROMPT_V2, _without_place, _drop_place_phrase,
decide_subjects), `src/stage6_visual_augmentation/__init__.py` (plain_prompt, _diffusion_image).

## How the specific sound gets lost (read in the code, verified on data)
1. `consolidate_families` relabels every detected sub-label to its **canonical family** (Ambulance
   (siren) → Siren; Bus, Car → Vehicle; Crowing → Bird). The family is what the gate reasons about.
2. The most confident specific child is kept in `detail`, but **only if it passed the display bar**;
   often `detail` is empty or generic ("Rail transport", "Thunderstorm").
3. The depiction prompt is given `label = family` and the detail "in brackets"; the VLM writes the
   subject **without seeing the frames**, from the label + a one-line place description.
4. Measured on the 33 drawn DEV sounds: **7 of 33** had a more specific detector label in the same
   family, overlapping in time, that never reached the subject (e.g. detector heard *Bus* → drew "Car
   driving"; heard *Toot* (a horn) → drew "Train moving"; heard *Car* → drew "Vehicle driving"; heard
   *Thunder* → drew "Sky rumbles"; heard *Boom* → drew "Bomb explodes").

## What was tried last night (docs/NIGHT_REPORT_2026-09-24.md, docs/picture_quality_prereg.md)
- DEPICT_V2: object that makes the sound first; never sky/body part alone; homonyms qualified.
- KIND_ALWAYS + KIND_PROMPT_V2: ask the frames which kind (setting + era). **Said "unknown" 32 of 33.**
- Place-phrase strip: removes "in palace", "on street" (the scene was leaking into the picture).
- Generator: Qwen-Image-2512 instead of FLUX.1-schnell (~20 s vs <1 s per picture, needs 80 GB).
- Automatic picture markers (Idefics3 forced choice, CLIP-L, Idefics3 open two-line) **all failed**
  calibration against a human. Idefics3's answer to "what object is drawn" did expose missing
  objects ("man with face mask" for a shaver picture with no shaver).
- A verify-and-redraw loop was **not** built: previous reviewers warned "redraw until the checker says
  yes" keeps exactly the pictures the checker is lenient on.

## Adam's blind ratings this morning (the only independent measurement)
"Would a viewer understand this sound at a glance?" Today's pictures **12/31 yes**; new (V2 rules +
place strip + Qwen-Image) **18/32 yes**; paired 11 better, 5 worse; holds over 3 seeds. He is much
stricter than the author (kappa 0.46). Flaw: the rating page showed the **family** name under each
picture. New pictures he still said no to include: "Train wheels grinding", "Red fire alarm bell",
"Thunderclouds rumble" (x2), "Hands clapping together" (for Crowd/Applause), "Car engine" and "A
truck engine" (for Vehicle), "Train" (for Vehicle — the detector heard a horn), "Ambulance siren",
"Cupboard door swings open", "Pigeon flapping wings", "Steam engine", "Firecracker exploding".

## Adam's requirements, in his words
- "Many images are generated from the family tag instead of the specific sound, like train is vehicle
  or ambulance becomes siren. This is terrible and loses context for DHH."
- "The SCENE must be used to get more context, so that if a car door opened is heard in the scene you
  don't show a house door. If a rooster is heard (cock-a-doodle-doo) you won't show a generic bird."
- "The prompt (and probably the model) needs to be much better."
- "If Gemma is stronger use it; also consider SOTA models and maybe a better technique, like: ask the
  model what we see, then generate, then ask what is seen and whether it fits the sound AND the scene."

## Hard constraints
- No training / fine-tuning. Runs on BIU GPUs (A100-80GB, H200 141 GB); models must be open weights.
- A picture must never tell a DHH viewer something false (a fire engine drawn in flames says "fire").
- Earlier rule from Adam: **no example sentences inside prompts** (examples leak their content into
  the answer). Earlier concern: the scene must not be *painted* into the picture (the palace crowd);
  Adam now wants the scene used to pick the right *source* — those are different things.
- Evaluation must stay honest: a model that checks pictures inside the loop cannot also be the one
  that grades them; the thesis headline metric (per-sound timing against Adam's labels) is unaffected
  by picture content.

## Cached on the cluster
Qwen3.8-27B (VLM), Gemma-4-31B-it (passed both judge trust checks), Idefics3-8B, Qwen-Image-2512,
FLUX.1-schnell, CLIP-L, SigLIP-so400m, MiniLM, mpnet. The login node has internet for downloads
(~80 GB of home quota free; a 2 TB lab share exists).
