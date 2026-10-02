# Topic 2 — Image generation from audio and image data (3 reviewers × 3 rounds)

Read `docs/panel3_common_context.md` first. Adam: *"the image generation using the audio and image data."*

## How pictures are made today
- **Subject text** (stage 5b, `src/stage5_cross_modal_analysis/reason.py`): the detector's label → `choose_source` picks the
  most specific fired sub-label (e.g. Vehicle → "Tractor" only if the detector fired it). V3.1 adds a scene step
  (`RESOLVE_PROMPT`): the video frames may add ≤ 2 words saying which *kind* ("farm tractor"), never a new noun; then
  `DEPICT_PROMPT_V31` writes one phrase ("the whole thing making the sound, caught in the act"), checked by word-list guards
  (no forbidden names, no invented person, no other-branch maker, must name the source's chain).
- **Picture:** FLUX.1-schnell (shipped, 4 steps); Qwen-Image-2512 (1024 px, 50 steps, true CFG 4, negative prompt) tested.
  Plain white background, one large subject. Frozen final setup adds templates for hard sounds (thunder → lightning bolt,
  rain → drops on a window, car alarm, train horn, church bell, glass shatter) and comic word cards for sounds with no
  maker (WHOOSH, THUD, BANG, SMACK, WHACK). `docs/freeze_picture_setup_2026-09-25.md`, `benchmark/gold/gen_screen.py`.
- **What the audio contributes today:** only the detector's label names (and their ontology chain). The raw audio never
  reaches the picture. **What the video contributes:** only the place/kind qualifier (≤ 2 words); the frames' look (colour,
  style, the actual object's appearance) is never used.

## What we know about picture quality
- **Blind human glance test (Adam, 54 sounds, 50 fresh clips, 1.5 s at 384 px, type what makes the sound):** FLUX + shipped
  text 14/54; Qwen-Image + shipped text 26/54 (+22 pts, sig.); Qwen-Image + V3 text 32/54 (+33 pts) — V3 rejected for an
  invented object (a thud drawn as a house door). Qwen-Image + shipped text also showed one wrong object (a ringing phone
  drawn as a desk bell). Repeats 9/10 consistent.
- Failure kinds: sounds with no drawable maker (whoosh, thud) → "can't tell"; sibling confusions (sheep ↔ goat); weather
  (thunder, rain) unreadable; wrong object from a vague label (phone → desk bell; siren → megaphone-like horn).
- **Automatic picture checkers all failed calibration against the human** (Idefics3, CLIP-L, GLM-4.6V, Gemma verifier,
  per-picture judge, pairwise GLM "first picture" 187/190). The AI assistant's own blind screen agrees with Adam at κ 0.46.
- Tried generators: image retrieval (licence/coverage), SDXL, SDXL-Turbo, PixArt-Σ, FLUX, Qwen-Image-2512, Qwen-Image-2.1
  (no gain, research licence), AnimateDiff/two-frame motion (dropped). Not tried: FLUX.2, HiDream, image editing from the
  video frame, audio-conditioned generators (e.g. audio-to-image models), audio-LLM captions of the actual sound.
- Pending: one blind confirmation sitting of the frozen setup (81 sounds, sealed); only route to a "clean" picture claim.

## Questions (round 1 — ≤ 10 lines each, concrete)
1. How could the **audio itself** (not just the label) and the **video frames** improve what is drawn, honestly and in 6
   days? E.g. an audio-LLM describing the actual sound ("a heavy wooden door slams" vs "Thunk"), audio-conditioned image
   models, using a frame as a style/appearance reference (image editing, IP-Adapter), matching the scene's look — and the
   risk of each (inventing things the audio never established, drawing what is already visible, slower glance reading).
2. What should a picture *be* for a deaf viewer at a glance — photo, icon, pictogram, word card, picture + word? What does
   the accessibility literature (DHH captioning, sound visualisation) say, and what is testable this week?
3. The best generator/prompting setup to run now (models with public weights and a usable licence, VRAM on H200/A100).
4. How to judge picture changes without another heavy session from Adam (one rater, no working automatic checker).
5. The one thing not to do.
