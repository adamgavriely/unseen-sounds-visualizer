# Benchmark

A curated, human-labeled set of short clips (10–20 s) for evaluating the system. The ground truth is
the judgement at the heart of the project: **for each non-speech sound, is its source visible
on-screen or not?** A clip "benefits the system" when it contains a salient ambient/environmental
sound whose source is *off-screen* — that is what the system should visualize.

## Clip categories
- **`positive_offscreen`** — ≥1 salient non-speech sound with source **not** visible → system should
  augment. *The bulk of the set.*
- **`negative_control`** — salient non-speech sound(s) present but **all** sources visible → system
  should show nothing. *A minority, to test that the system correctly stays silent.*
- **`mixed`** — several events, some visible some not → tests selective augmentation.
- *(Speech-only / silence clips are excluded — nothing for the system to add.)*

## Ground truth
`gt_should_augment` (per clip) = events that are `salient && source_visible == false`. This is
labeled by a human from extracted keyframes, and is **independent of the detector** — a sound is
labeled even if PANNs misses it (e.g. a red-alert siren), so the benchmark is not biased by detector
gaps and can later test the open-vocabulary path. Background music is flagged (`music: true`) pending
the "is music in scope?" decision.

## Sourcing (hybrid)
- **Off-screen positives** → hand-curated CC clips (Wikimedia Commons: protests, streets, transport,
  weather/nature, emergency; plus Internet Archive PD / CC0 stock). Cleanest licensing; verify per
  clip.
- **Mixed / scale** → **UnAV-100** (CC-BY 4.0), untrimmed multi-event clips via its YouTube-ID CSV.
- **Negative controls** → **VGGSound** (HF mirror `Loie/VGGSound`); built so the sound source is
  visible → ideal source-visible controls.
- **MovieNet is not used for clips** — it does not ship raw video (registration + agreement, movies
  excluded). Movie scenes can be added by supplying local files to `curate.py`.

## Composition (phased)
- **Pilot ~30**: ~20 positive_offscreen, ~7 negative_control, ~3 mixed.
- **Scale ~120** (fallback target): ~65% positive / 20% control / 15% mixed.
- **Stretch ~300** (proposal target).

## Files
- `curate.py` — assisted-curation helper: ingest a clip (path/URL) + trim, run Stage 1 + Stage 4,
  extract keyframes at event peaks, write a review bundle to `data/work/benchmark/<clip_id>/` and a
  draft manifest entry. A human sets `source_visible`/`category`; the entry is merged into the
  manifest.
- `manifest.json` — committed ground truth (metadata + labels + URLs + licences; **no video**).
- `ATTRIBUTIONS.md` — per-clip source/licence/author record (CC-BY compliance).

## Usage
```bash
python -m benchmark.curate --input data/input/london_protest.webm --clip-id london_protest_01 \
    --source wikimedia --url <commons-url> --license CC-BY-SA-4.0 --attribution "..." --start 4 --end 20
```
Then review `data/work/benchmark/<clip_id>/` (frames + `events_plot.png`), set `source_visible` per
event and the clip `category` in `draft.json`, and merge into `manifest.json`.

**Status:** tooling built; pilot curation in progress.
