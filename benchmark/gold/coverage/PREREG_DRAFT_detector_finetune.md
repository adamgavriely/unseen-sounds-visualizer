# DRAFT pre-registration: fine-tune a frame-level sound detector (not started; waiting for Adam)

Status: draft written 2026-10-07. Nothing downloaded or trained. To be finalised and committed again before any run.

## Goal

A frame detector that hears quiet off-screen sounds under speech / music better than the frozen BEATs + FlexSED stack,
so more needed sounds become candidates (DEV: 6 of 58 needed sounds are heard by no model; many more are heard too
weakly). Judged only by the frozen scorer on DEV, inside the frozen pipeline (it replaces or joins the BEATs ear).

## Model

PretrainedSED (Schmid et al., MIT) frame-level checkpoints, already on the cluster in `~/PretrainedSED/resources`
(2.3 GB folder): **ATST-F_strong_1.pt** (primary) and **BEATs_strong_1.pt** (second arm); also present: M2D, ASIT,
fPaSST. Training code `ex_audioset_strong.py` is present.

## Data

| set | use | on the cluster | size to download (approx.) |
|---|---|---|---|
| AudioSet-Strong train (frame labels, 456 classes) | main training data | labels only for the eval clips we use; no train audio | about 103k 10-s clips; only obtainable by downloading each YouTube video (some are gone); about 30-35 GB as 16 kHz WAV, about 50 GB in the PretrainedSED 32 kHz HDF5 |
| FSD50K | foreground events for Scaper mixtures | no | about 24 GB (zipped) |
| ESC-50 | foreground events | no | about 0.6 GB |
| MUSAN (speech + music + noise) | backgrounds | no | about 11 GB |
| LibriSpeech train-clean-100 | speech backgrounds | no | about 6.3 GB |

Disk: home quota 400 GB, about 120 GB free (6 Oct). All of the above plus mixtures (about 20 GB for 20k 10-s mixtures)
fits only just; the HuggingFace model cache already holds 276 GB. Ask Adam before downloading AudioSet audio from
YouTube (licence / terms question, and about a day of download time).

## Leakage rule

Every YouTube id of the benchmark (DEV, TEST, slice B, the 415 held-out, the 422 fresh set, all pilots and caches) is
removed from every training source before training: the id list is built by the same scan as
`benchmark/gold/audioset_fresh.py` (every id in any file or file name under benchmark/ and data/). FSD50K clips that
come from Freesound uploads of benchmark audio are not expected, but the FSD50K / ESC-50 file ids are also checked
against the benchmark sources.

## Mixtures (Scaper)

10-s mixtures: one background (MUSAN speech or music, LibriSpeech, or MUSAN noise) at 0 dB reference; 1-3 foreground
events from FSD50K / ESC-50 mapped to AudioSet classes (only classes that map one-to-one), event SNR uniform in
[-10, +5] dB against the background (the "quiet sound under speech" case). Frame labels from the Scaper annotations.

## Training

Start from the strong checkpoint; fine-tune on AudioSet-Strong train + mixtures (1:1), same loss and schedule as
`ex_audioset_strong.py`, 3 seeds; one H200 (GPU hours are not a constraint). Model choice by AudioSet-Strong eval
PSDS on clips NOT in the benchmark, never by DEV.

## Test (DEV only)

Replace the BEATs ear by the fine-tuned detector (same bars re-fitted on the AudioSet-Strong eval clips, not DEV),
keep every later stage frozen, harness on merged DEV. Pass bar: more hits than the a/b candidate (32) with wrong
<= 16, and needed sounds heard (any candidate of the family in the onset window) up. TEST untouched.
