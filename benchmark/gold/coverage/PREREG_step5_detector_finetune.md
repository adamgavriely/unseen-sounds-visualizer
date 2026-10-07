# Pre-registration: Step 5, fine-tune a PretrainedSED frame detector on open data (DEV is the only judge)

Written and committed 2026-10-07 before any data was downloaded or any model trained. Replaces the draft
(PREREG_DRAFT_detector_finetune.md). No YouTube download; TEST untouched; nothing adopted without Adam.

## What failed before, and what this changes

Attempt ten (prereg_psed.md, 18 Sept, release v1.2.0): PretrainedSED BEATs-strong as the ear was precise and
well-timed (DCASE masked recall 34% vs 9.5%, onset error 0.19 s) but kept only 7 of 23 labelled real sounds on the dev
clips: it scores long ambient sounds under music or speech near zero (helicopter under music 0.16, rain 0.02).
Attempt eleven (5-backbone average) did not raise hidden-sound recall either; on the gold re-run (22 Sept) gated
PSED - gated BEATs dF1 was -0.007. This step targets exactly that failure: fine-tune the frame model on events placed
5-20 dB UNDER speech, music and crowd beds, with the original model as teacher everywhere else so it keeps what it
knew.

## Data (official sources only; openly licensed)

FSD50K dev + eval (Zenodo 4060432), ESC-50 (GitHub karolpiczak/ESC-50), MUSAN (OpenSLR 17), LibriSpeech
train-clean-100 (OpenSLR 12). Leakage: FSD50K and ESC-50 come from Freesound, not YouTube; still, every file id is
checked against every YouTube id / file name under benchmark/ and data/ (the scan of audioset_fresh.py) and any match
is removed.

## Mixtures (own numpy mixer, Scaper-style; strong labels by construction)

- 10-s mixtures at 16 kHz. Bed: one of MUSAN speech, MUSAN music, MUSAN noise, LibriSpeech (speech), or an FSD50K clip
  of a crowd-type class (Crowd, Chatter, Hubbub), level -20 dBFS RMS.
- Foreground: 1-3 events from FSD50K / ESC-50 whose label maps to an AudioSet-Strong train class whose canonical family
  passes the shipped "depictable" filter (src/labels.py) and is not speech / music. FSD50K names are AudioSet names
  (inverse of psed_infer.STRONG_TO_ONTOLOGY where renamed); ESC-50 by the committed 50-row map `esc50_map.json`.
- Event level: uniform 5-20 dB under the bed (SNR -20 .. -5 dB, RMS over the event's active frames).
- Activity inside a source clip (FSD50K / ESC-50 have no in-clip times): 40-ms frames with RMS above -30 dB of the
  clip's max, gaps < 0.2 s bridged, runs < 0.2 s dropped. This is label noise we introduce; it is stated.
- Frame labels on the model's grid: 40 ms (hop 160 x pooling 4, 250 frames per 10 s). 20 000 mixtures train, 1 000
  held out for training loss only. Sources split 90/10 by file before mixing.

## Training

ATST-F_strong_1 (primary), BEATs_strong_1 (second, if time). Own loop on PredictionsWrapper + mel_forward. Teacher =
the frozen original checkpoint run on the same mixture. Target = the teacher's probability for every class and
frame, overridden to 1 on the inserted events' active frames for their classes (and their AudioSet-Strong ancestors
present in the class list). BCE on all 447 classes. AdamW lr 1e-5 (backbone) / 1e-4 (head), 10 epochs, batch 32,
cosine schedule, mixup off, one seed first (3 if tier 1 passes). Checkpoint every epoch; the epoch is chosen by the
lowest held-out mixture loss (never by DEV).

## Calibration (no DEV)

Each ear's bar: the method of benchmark/detector_calib.json, on the 415 held-out AudioSet-Strong clips (labelled,
already used for detector work; NOT the 422 fresh set). The bar maps to 0.35 by psed_infer.rescale.

## Tier 1 (CPU after inference; decides whether tier 2 runs)

Ears: frozen BEATs (the pipeline's), original ATST-F strong (control), fine-tuned ATST-F. On the 1196 DEV bursts of
Step 3b: AUROC of the ear's family max probability in the burst for good vs not good (truth as Step 4; reference:
BEATs 0.710, FlexSED 0.809, DASM 0.885, measured before this was written). And needed-sound candidate recall: needed
sounds (importance 2-3) with a same-family span of the ear starting in the onset window, at the ear's bar.
Tier 2 runs only if the fine-tuned ear beats BOTH the original ATST-F and BEATs on both numbers.

## Tier 2 (the pass bar)

Full pipeline on merged DEV with the new ear, two modes: replace (BEATs ear swapped; ONSET_CAM off, frame model) and
join (frame-wise max with BEATs on shared class names). Job order: new ear caches -> listener pools and answers for the
new spans (clip_prep steps lpool, qwen, afn, open inventory; dasm_rescue listen) -> harness stage 4 -> stage 5 ->
gates -> the a/b gate re-decision (Step 2). Pass: more DEV hits than the a/b candidate (32) at <= 16 wrong, with
picture ends within 1 s of the frozen display rule. Reported next to the old PSED rows.
