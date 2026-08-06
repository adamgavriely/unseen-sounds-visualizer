# Benchmark

Curated set of ~300 short clips (10–20 s) for evaluating the system, drawn from three public sources
and spanning three scenarios. See `docs/project_notes.tex` §Datasets for the full rationale.

**Scenarios:** (1) ambient environmental sounds; (2) localized acoustic events (sirens, glass,
explosions, applause, laughter, crying); (3) mixed audio scenes.

**Sources & role:**
- **VGGSound** — clean single-event ambient/environmental clips (sound source visible; can be
  inverted to build "source off-screen" cases). CC-BY.
- **UnAV-100** — mixed / overlapping-event scenes. CC-BY 4.0.
- **MovieNet** — narrative acoustic-event scenes with context.

**Selection criteria:** rich semantic audio; audio only *partially* represented by the visuals;
high-quality sync; environment diversity. Record source + licence per clip in a manifest.

**Status:** Not started (research phase).
**TODO (Adam):** verify dataset access/terms — MovieNet may require a signed agreement/application;
VGGSound ships as YouTube links (some videos offline → need a download+availability plan); UnAV-100
CC-BY is fine. Confirm the 300-clip target is feasible in the timeline or set a fallback (100–150).
