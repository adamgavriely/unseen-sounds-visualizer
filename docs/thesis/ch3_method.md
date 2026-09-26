# Chapter 3 — Method (draft, 27 Sept 2026)

*The system as scored in the final TEST table (amendment 21, tag `test_final_v33`); one call reproduces it:
`config.use_shipped()`. Every choice below was made by a test with a bar written before the run; the alternatives
that failed are listed with the reason (full ledger: `docs/LEDGER_2026-09-26.md`).*

## 3.1 Task and output

Input: a 10–30 s video with its soundtrack. Output: the same video with a side panel (720 px) that shows, while a sound
is heard, a picture of that sound — but only when the viewer cannot see where the sound comes from. Up to three pictures
at once (the most sounds ever heard together in the benchmark), each in a fixed slot; a slot is empty when its sound is
silent. No component is trained or fine-tuned; the contribution is the chaining and the decision of *what not to show*.

## 3.2 Stages

| # | stage | model (shipped) | what it decides |
|---|---|---|---|
| 1 | audio extraction | ffmpeg, 16 kHz mono | — |
| 2 | speech | Whisper base (int8) | speech is never drawn; transcript used only as context |
| 3 | scene objects | OWLv2 (6 frames) | context for later stages; its own visibility verdict is not used (amendment 15) |
| 4 | sound events | BEATs (bar 0.35) ∪ FlexSED (bar 0.8) + FlexSED veto 0.3 + PANNs veto 0.05 | which sounds, when |
| 4b | onsets | occlusion refinement, clamped: a start may never move before its anchor | when each picture starts |
| 4c | drawable? | ontology-rule label filter (speech and music never drawn) | which sounds may become pictures |
| 5 | visibility gate | Qwen3.8-27B, 6 frames per ≤ 5-s stretch, 3 questions, majority | draw only if the source is off screen in some stretch |
| 5b | subject | Qwen3.8-27B + fixed word lists | the specific thing to draw |
| 6 | picture | FLUX.1-schnell (shipped); Qwen-Image-2512 (tested, §5.9) | the image |
| 7 | display | fixed slots, ≥ 1.5 s on screen, repeats < 2 s merged, no length cap | when the picture is visible |

**Stage 4 in detail.** BEATs (AudioSet-2M, 527 classes) is the base detector. Under speech or music it misses quiet
sounds: at every masked needed sound its top labels were Speech or Music. FlexSED — a text-queried frame-level detector
asked one label at a time over 215 drawable family names — scores those sounds far higher, though mostly below its
shipped bar (1 of 7 masked DEV sounds reaches 0.8), and is added as a union with its own bar; a
family both detectors report at the same moment keeps the earlier start. Two one-sided vetoes remove what one model
alone claims: a BEATs label that FlexSED never hears in the clip (τ = 0.3), and a FlexSED-only span that PANNs CNN14 does
not support (τ₂ = 0.05). A span both detectors raised is never removed.

**Stage 5 in detail.** For each drawable sound and each stretch of ≤ 5 s, six frames are shown to the VLM with three
questions: name the thing making the sound (open naming, then "could that thing make it?"); a forced a/b choice asked in
both orders ("you can see it happening" vs "not visibly happening"); and a description followed by the same check. The
sound is silenced only if the majority says "visible" in **every** stretch. A sound whose parent family is visible at an
overlapping time is silenced too (the kinship rule).

## 3.3 Why these models (and not the others)

| stage | chosen | also tried | deciding evidence |
|---|---|---|---|
| detection | BEATs ∪ FlexSED | PANNs, PretrainedSED (5 backbones), FLAM ×2, AST, CED, SSLAM, CLAP verifier, Qwen2-Audio verifier | 13 pre-registered attempts; only the union raised recall of needed sounds at half the false labels (amendment 8) |
| false-alarm filter | FlexSED veto + PANNs veto | AST/CED votes, Demucs/HTDemucs/DeepFilterNet views | the vetoes gave the precision gain that holds on TEST (+0.137, Holm-significant) |
| visibility | Qwen3.8-27B | CLIP, SigLIP, OWLv2, SAM 3, Qwen2.5-VL-7B/32B, video input, audio-visual sync | balanced accuracy on the gold ticks 0.62 (OWLv2 0.50 = chance); three extra mechanisms each landed at the β = 2 break-even |
| subject text | word-list guards | free VLM prompt (V3) | V3 invented an object (a thud drawn as a door) in the blind rating |
| picture | Qwen-Image-2512 (Apache-2.0) | retrieval, SDXL, SDXL-Turbo, PixArt-Σ, FLUX, Qwen-Image-2.1 | +22 points recognised in a blind glance test; 2.1 no better and research-only licence |
| judge (secondary) | Gemma-4-31B, sees the panel | Mistral-7B, describe-then-judge, six picture checkers | passes all three trust checks against the annotator; Qwen models excluded (same family as the gate/subject model) |

## 3.4 What the system does not do

It does not localise sources in the image, does not generate video, and does not decide *how much* of a picture a viewer
needs (no importance ranking beyond confidence). Pictures end when the detector's span ends; with no length cap, 4 of 15
TEST pictures outlast their sound by more than 2 s (a principled end rule could not be tested without a DEV case).
