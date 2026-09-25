# Panel of five, three rounds — the generator, and a scene-aware prompt written by the VLM (2026-09-25)

Adam: *"which image generator can we use among SOTA models, AND how can we refine the prompt to be more
specific — maybe let the VLM generate the prompt? If fire is heard and the scene is a forest, it could
generate fire crackling in a forest."* Also: *"most images are very bad in my opinion"* and *"clearly a VLM
could see most images here are bad; no need for a human to say this."*

Reviewers: G1 (generators, SOTA), G2 (visual communication, prompts), P2 (statistics, forking paths),
P4 (automatic evaluation), P5 (deaf and hard-of-hearing viewers). Read-only; repo `P:\MscProj`.

## Where we are (all measured)

* Adam's blind round 2 (54 sounds, 50 never-annotated clips, no name shown, he typed what he saw):
  FLUX today 14/54 right; Qwen-Image-2512 with today's text 26/54; Qwen-Image + V3 text 32/54 (+33 points vs
  today, significant; the generator gave +22 of it). V3 is vetoed by one false message (a Thunk drawn as a
  door). Failures: "can't tell" on sounds with no single object (whoosh, thunder, car alarm, smash, rain on a
  surface, train horn); "vague" on bird actions read as "bird"; misreadings (sheep as goat).
* A 2-reviewer generator panel (G1, G2; `docs/panel_2026-09-25_generator.md`) signed plan **GP-2**: compute
  gate, then screen HiDream-O1-Image (MIT) / Qwen-Image-2.1 (research licence) / Mage-Flow (MIT, gated)
  against Qwen-Image-2512 on the 54, prompt arms on the winner (whole thing in frame, mid-act with its
  visible effect, one large subject; guarded LLM expansion), fixed template subjects for sounds with a
  canonical maker (thunder → lightning bolt), a burst-with-word card for sourceless sounds, rule 2 widened
  to unnamed inventions, confirmation on 50 fresh unannotated clips, pass bar written first.
* Automatic judges: blind "what is this?" checkers failed against Adam (GLM kappa 0.45; the per-picture
  judge said "can't tell" to 502/567). A **named yes/no verifier** ("this should show X — would a deaf
  viewer recognise it at a glance?") is being calibrated right now on Adam's 170 round-1 yes/no answers
  (`docs/prereg_verifier.md`); if it passes, it screens generators instead of Adam.

## Why the place was removed from the picture (history the panel must weigh)

The depiction step used to get the *place* ("palace", "street") and the generator painted it: two crowds in
front of a palace, a fire engine placed in flames, scenery instead of the source; the place strips then had
to scrub it. The current rule: no place in the subject, a plain white background, a negative prompt against
scenery. The scene is still used — V3.1 lets it add one qualifier to the heard word ("car door", "farm
vehicle") under a list guard — but never as a background. The picture sits **beside** the video, whose scene
the viewer already sees.

## Adam's proposal

Let the VLM (it sees the clip's frames) write the image prompt itself, specific to the scene: fire heard in a
forest → "fire crackling in a forest"; possibly a longer, richer prompt (G1: these generators were trained on
40-80-word captions, so a 5-word prompt is out of distribution).

## Round 1 questions (at most eight lines each; your lens first)

1. **Generator:** does GP-2's candidate list stand, or is there a better SOTA choice now? (G1 leads.)
2. **Scene in the picture:** should the picture include the setting (forest fire vs a generic fire), and if
   so how much — a background, a small hint, or only a qualifier word? What does it cost a deaf viewer who
   already sees the scene beside it? (G2, P5 lead.)
3. **VLM-written prompt:** should the VLM (with the frames) write the whole image prompt? How do we stop it
   inventing a source the audio never heard (the sword, the police car, the door) — which guard, which
   check? (G1, G2, P4.)
4. **Measuring it without Adam where possible:** can the named verifier (if it passes) or another automatic
   check screen the prompt variants, and what must still be a human? (P4, P2.)
5. **Forking paths:** we have now changed the picture rules several times on the same clips. What is the
   cleanest order and set of clips so the final claim holds? (P2 leads.)

---

# Round 2 — merged plan GP-3 (sign or amend)

**New fact since round 1:** the named verifier **failed** calibration (catches 34 % of Adam's "no", rejects
3 % of his "yes", kappa 0.28): told the sound, it says "yes" to 135 of 170. So **Adam screens**; the
verifier's conditional role in Q4 does not apply. Both blind and named automatic checkers are now recorded
as failed instruments.

**Agreed by all five in round 1**
* **Generator:** GP-2's list stands — HiDream-O1-Image (MIT), Qwen-Image-2.1 (research licence: usable in
  the thesis, labelled non-deployable), Mage-Flow (MIT, gated), control Qwen-Image-2512. LLaDA-Image
  (Apache-2.0 on its card, 4 Sept 2026) is an optional fourth arm if disk and its licence tag allow. The
  compute gate's outcome per candidate is written into the prereg before any screening picture exists.
* **Scene in the picture: a qualifier word only, never a background.** The viewer already sees the scene;
  a background shrinks the subject, invites looking for the thing in the video, and is where false
  messages came from. The scene decides the *kind*, not the backdrop: Adam's "fire in a forest" →
  "campfire flames crackling".
* **VLM prompt, two steps:** (1) frames enter only through RESOLVE (V3.1, ≤2 qualifier words, list guard);
  (2) a **text-only** rewriter expands the resolved subject to 40-80 words describing only *how* — pose,
  act, its visible effect, framing (PE-T2I on Qwen-Image-2.1; Qwen3.8 elsewhere). A mechanical noun guard
  runs on the rewritten text: every noun must be the source, its chain, the qualifier, or a fixed
  effect/material word (pieces, splash, flash, smoke, steam, sparks, dust); no person outside human-sound
  families; no place noun. Any failure → the short prompt. Fallback rate reported per arm. A false message
  in this arm vetoes it (declared now).
* **Gemma never acts inside the pipeline** while it is the judge (P4, G1).
* **Forking paths (P2):** the picture rules were rewritten on three sets after reading results; nothing
  about V3.1 or the generator is yet confirmed out of sample.

**The order (GP-3)**
1. **Freeze the confirmation list now:** run the shipped detector + gate (pictures off) over the 489
   unannotated `unsorted` clips; take the first 50 by filename hash with ≥1 off-screen sound in group (a) or
   (b); commit the list before anything else is drawn on them.
2. Compute gate for each candidate; outcome committed.
3. **Screening on picture-DEV (the 54 + the 33 DEV bench sounds), all labelled "selected":** generators at
   seed_of(item)+1 on frozen V3.1 text; then prompt arms on the winner (image-prompt rules; two-step
   expansion); group (b) templates; group (c) burst cards; the V3.1 guard fix for the other-branch qualifier
   gap found on slice B. Adam rates, blind, glance fidelity (384 px, 1.5 s).
4. **Freeze commit** naming the final setup.
5. **One confirmation sitting** on the 50 frozen clips, blind, 10 repeats: primary = final setup vs **today**
   (the thesis claim); secondary = final vs Qwen-Image-2512 + V3 text (the generator's share). Bars fixed
   before: paired CI over sounds excluding zero, zero false messages under the widened rule 2, repeats
   ≥ 80 %. Ties keep the control. Every part-attribution stays "picture-DEV, selected"; only the composite
   is a claim.

## Round 2 questions (at most six lines each)
1. Sign GP-3 or amend points 1-5.
2. One disagreement: on a guard failure, P4 allows one retry ("do not mention X"); G1, G2, P2 and P5 say fall
   back to the short prompt, never retry. Which, and why?
3. Adam's own question, answered plainly: is there any form of "scene in the picture" worth testing as a
   screening arm, or is the qualifier word the whole answer?
4. **P4's last automatic proposal:** a *pairwise* check — two pictures of the same sound, "which would a deaf
   viewer more likely read as X, or neither", asked in both orders so a lenient or strict bias cancels —
   calibrated on round 1's discordant pairs (same sound, Adam said yes to one arm and no to another); bar
   written first: ≥ 75 % agreement with ≥ 90 % order-swap consistency. It may only *order* arms to shorten
   Adam's sitting; never choose per-sound pictures, never act in the pipeline. Allow this one attempt, or stop
   automatic screening here?

---

# Round 3 — final plan GP-4 (sign, or name the single point you cannot sign)

**Unanimous from round 2:** fall back to the short prompt on any guard failure, never retry (P4 withdrew the
retry). The scene enters only as (i) RESOLVE's ≤2-word qualifier choosing the *kind or state* of the source
("campfire flames", "burning branches") and (ii) effect/material/surface words where the sound lands
("rain on a window pane", "footsteps on gravel") — **no background arm**. Adam's "fire in a forest" is
answered by (i).

**GP-4**
1. **Freeze** (before anything is drawn on them): the confirmation clip list by the committed hash order
   (`benchmark/gold/pictures/confirm_hash_order.*`), the **sound list** the gate chose on them, the config
   hash of that run, and a check that none is in the 54, the 33, the gold set or slice B.
2. **Compute gate:** five fixed sounds per candidate, V3.1 text, no rating; seconds per picture, peak VRAM,
   negative prompt accepted, licence tag (LLaDA-Image) — committed. HiDream falls to Dev-2604 if too slow.
3. **Screening, two sittings, both "selected, re-exposed"**, arm hidden, shuffled codes, repeats, glance
   fidelity (384 px, 1.5 s): (a) generators at seed_of(item)+1 on frozen V3.1 text; (b) on the winner only:
   image-prompt rules / two-step guarded expansion (fallback rate reported) / group (b) templates / the V3.1
   other-branch-qualifier fix. Group (c) burst cards are rated but counted separately, never inside the
   correct rate. The 33 DEV bench sounds are never quoted alone.
4. **Pairwise check (majority: allow one attempt; P4: report-only):** GLM-4.6V-Flash (not Gemma, the judge;
   not Qwen3.8, in the pipeline); calibrated on the discordant pairs of round 1 **and** round 2 (both orders,
   384 px, "neither" = disagreement), bar ≥ 75 % agreement and ≥ 90 % order-swap consistency, reported with
   its CI; if it passes it may drop only the **bottom generator arm** before sitting 3(a), never prompt arms,
   never anything about false messages. Pass or fail it is the last automatic instrument; if it fails,
   automatic screening is closed and reported as three failed instruments.
5. **Freeze commit** naming the final setup.
6. **Confirmation, one sitting:** the frozen 50 clips' pictures drawn once at the committed seed and sealed
   until Adam's answers are in (nobody looks first); his first sight of those clips; glance fidelity; arms =
   final vs **today** (primary) and final vs Qwen-Image-2512 + V3 text (secondary). Primary quantity: correct
   incl. narrower, paired over sounds, CI excluding zero; zero false messages (widened rule 2) in the final
   arm; repeats ≥ 80 %. Power, stated now: ~55 sounds resolve a +0.3 difference, probably not +0.1 — a null
   secondary is "unresolved". The claim: "the pictures show the detector's source recognisably", not "the
   right sounds were drawn" (no gold on these clips).

## Round 3 question (at most five lines)
Sign GP-4, or name the single point you cannot sign and the fix.

## Round 3 outcome — GP-4 signed by all five, with these final lines (binding)

* **Point 4 (P2, P4):** the pairwise check may drop the bottom generator arm only when it is bottom by more
  than the check's own CI; the dropped arm is still drawn at the committed seed and sealed, a fixed 15-sound
  sample of it (chosen by hash before sitting 3(a)) goes into Adam's screening sheet under codes, and the drop
  is reported as confirmed or wrong against that sample.
* **Point 6, the order (G1, G2, P5):** draw and seal → the mechanical widened-rule-2 pass (checker's OBJECT
  line vs the subject text, and the blank guard) runs on the sealed pictures, output written to a file nobody
  reads → Adam rates → the by-eye rule-2 pass, with his answer file still closed → open and score. A veto is
  decided before anyone sees whether the final arm won.

Adam's question, answered: the scene chooses the *kind* of source ("campfire flames crackling"), and the
sound's visible *effect where it lands* may be drawn ("rain on a window pane"); a painted background is not
tested (all five). The VLM never writes the picture's noun; a text-only rewriter may lengthen the prompt with
how it looks, under a mechanical noun guard, falling back to the short prompt on any failure.

## Amendment — Adam: "don't make more work for me, only what is necessary" (before any screening picture is looked at)

Consulted P2 and G2 (both agree). **Adam's two screening sittings are dropped.** The author screens the
arms on picture-DEV (the 54 round-2 sounds), blind to arm (shuffled codes, arm key sealed until his answers
are typed), labelled "selected, author-screened" (author vs Adam kappa 0.46 on round 1: the author is lenient).
The three automatic instruments all failed and are not used. Adam rates only the **one confirmation sitting**
(repeats cut to 8, burst cards left out of the sitting).

**Author screening rule (written before looking):** default final setup = G2's pick: Qwen-Image-2512 + V3.1
with guard 2 (`PICTURE_SCENE_GUARD2`) + the rules tail + group (b) templates (+ burst cards in the pipeline,
not rated). An alternative arm replaces a part only if it beats the default by ≥ 8/54 in the author's blind
naming AND has no false message (widened rule 2): the Qwen-Image-2.1 arms for the generator, the expansion arm
for the text. By-eye checks before freezing (G2): rule 2 on every picture of the final arm; the six templates
at 3 seeds (the maker fills the frame); no part-only crops; no blanks; the guard log (fallbacks, no person
outside human sounds).

**Claims:** only the composite "final setup vs today" on the 50 frozen clips; every part stays "picture-DEV,
selected". Already reportable without more rating (P2): round 2's N0 vs today, the generator alone,
+0.22 [+0.07, +0.37], blind, fresh clips, no false-message veto against N0.

**Confirmation bars (unchanged):** correct incl. narrower, paired over sounds, CI excluding zero; zero false
messages in the final arm; repeats ≥ 80 %; ties keep today; the answer sheet for the 81 sounds committed
before Adam's file is opened; pictures sealed, mechanical rule-2 pass sealed, by-eye pass after his answers
are saved and before they are scored.

## Result of the pairwise check (GP-4 point 4) — FAIL; automatic screening closed

190 discordant pairs (140 round 1, 50 round 2), GLM-4.6V-Flash, both orders: agreement 46 % [39 %, 54 %]
(bar 75 %), order-swap consistency 48 % (bar 90 %). It answered "first" in 187 of 190 first orderings — a
position bias, which the order swap exists to expose. A probe (`scripts/pairwise_probe.py`) confirmed both
pictures reach the model (two image grids; it describes each correctly), so this is not an input bug. It
answered without its reasoning mode; running it again with reasoning on would be a second attempt at a
declared last instrument, so it is not done. **Three automatic instruments failed; the picture claim rests on
Adam's blind confirmation.**

## Author screening result (54 sounds, blind to arm; answers and decisions committed before the key was opened)

    arm                                              right   vague  wrong  can't
    Qwen-Image-2512 + V3.1 (control)                 39/54    12      3      0
    Qwen-Image-2.1 + V3.1                            38/54     7      4      5
    Qwen-Image-2.1 RGBA + V3.1                       36/54     7      4      7
    2512 + V3.1-guard2 + rules tail                  42/54     8      4      0
    2512 + V3.1-guard2 + rules + templates + cards   45/54     8      1      0     (the declared default)
    2512 + guarded expansion (70 % fell back)        41/54     8      5      0

vs control: RT +6/54 (+0.111 [0.000, +0.222]); R +3; X +2; Q21 -1; Q21-RGBA -3. By the rule written first,
no alternative beats the default by >= 8/54, so the **default stands: Qwen-Image-2512 + V3.1 with guard 2 +
the rules tail + group (b) templates + group (c) burst cards.** Qwen-Image-2.1 is not adopted (no gain in this
screen; research licence). Labelled "selected, author-screened"; the author is lenient (39/54 on the control
arm where Adam gave the similar N arm 32/54).
