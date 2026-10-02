# Picture panel — round 2: the merged plan (sign or amend each point)

Round 1 findings that everyone should now take as given:

- **P1 verified two deterministic bugs.** `_without_place` deletes the source word when the place
  names it ("Car door slams shut" in "car interior" → "door slams shut"; "Train horn blaring" /
  "train station" → "horn blaring"; "Church bell ringing" / "church courtyard" → "bell ringing") — the
  car-door failure Adam described, produced after the model got it right. And `consolidate_families`
  picks the most confident family member as `detail`, which is often an ontology *ancestor*
  ("Rail transport", "Thunderstorm") rather than a child, and blocks the real child.
- **P2:** the kind question failed 32/33 because it never saw the detector's sub-label (it asked "which
  kind of Vehicle" when the audio said Bus), used first-stretch frames for a whole-clip fact, and
  offered "unknown" as the honest exit. Also a **corroboration leak** in last night's KIND_ALWAYS: a
  guess from the setting could count as evidence that a faint sound is real. KIND_ALWAYS is off, so no
  run was affected.
- **P3:** no model swap fixes the main failures; the writer is blindfolded (family label, no frames).
  Keep Qwen3.8-27B and Qwen-Image-2512; the generator is called with `negative_prompt=" "`. And the
  judge's picture **describer is Qwen3.8-27B, the same weights that write the subject**.
- **P5, from Adam's ratings:** he rewards a *whole, recognisable* source *caught in the act* (rooster
  with beak open: yes; roosters with beaks closed: no; sparrow singing 6/6; pigeon in flight 0/6), and
  punishes parts. **4 of the 5 pictures that got worse were caused by last night's "name FIRST the
  solid object… do not add any object that is not the source" rule**: an engine block for a car, a
  wheelset for a train, two pairs of hands for a clapping crowd, a desk bell for a fire alarm. Rater +
  seed noise is 25-30% (identical beacons got yes and no); the solid evidence is the pooled FLUX 30%
  vs Qwen-Image 48% yes across seeds. DHH literature (Jain et al. CHI 2019; Findlater et al. CHI 2019):
  **source first; misattribution is the worst error; a family-level picture is acceptable when the
  system is unsure.**

## The merged plan

**1. Fix the two bugs first** (deterministic, no parameter): `_without_place` protects the detector's
label and detail words, not only the frames' kind; the corroboration step counts only a kind *seen*
in the frames, never one *inferred* from the setting.

**2. One `source` per burst, descending only as far as the evidence goes.** The family keeps every
gating role (visibility, dedup, rows, the thesis metric). Candidates for the drawn source: raw firings
in the family at or above AED_THRESHOLD (0.175) that overlap the burst in time; an ancestor of another
candidate is dropped; ranked by ontology depth, then confidence; a child counts only if its confidence
is at least REL_FLOOR × the family's peak in that burst. If the top two are siblings within a margin,
the frames make a closed choice between them (both orderings, "cannot tell" allowed), and "cannot tell"
means **their common parent** (Ambulance / Police car / Fire engine → Emergency vehicle). If nothing
qualifies, the source is the family. **The picture never goes more specific than the audio or the
scene establishes** (P1, P5, the literature).

**3. The scene picks, never adds.** Once per clip, structured fields from the whole-clip frames:
setting (≤4 words), era {period, modern, cannot tell}, vantage {inside a vehicle, on foot, static}.
The scene may (a) choose among the audio's candidates, (b) add one qualifier only where the place
*determines* the kind (car door inside a car; train on a platform) — checked by a two-ordering a/b —
(c) veto a faint sound through the existing `_fits_the_place`. It may never add a noun the detector
did not hear. Audio wins on *what*.

**4. Two prompts, no example sentences.** RESOLVE (text only, gets the candidates with their ontology
chain + the scene fields) → the source, ≤3 words, "unknown" allowed. DEPICT (text only, **no place at
all**) → **the whole recognisable source, shown in the act of making the sound** — replacing last
night's "object first, nothing else" rule, which drew parts. Keep: never sky/air/body part alone; a
homonym gets its qualifier. With no place in DEPICT, the place strips become backstops only.

**5. Generator:** Qwen-Image-2512 with a real negative prompt (scenery, landscape, room, street, sky
background, people who are not the source); three seeds per picture. Z-Image base as one extra
same-seed arm. No editing model yet.

**6. The checker loop Adam asked for,** as P4 designed it: draw → checker A sees only the picture
(not the sound, not the subject) and names the object and the sound → matched as text against the
source, its siblings, the fixed decoys and "nothing" → checker B gets the sound's frames plus A's
object name and asks, in both orderings, whether that thing fits this place and era → on failure
reseed once (same subject), then one rewrite with the source inserted (the checker's words are never
fed back), then a **text card with the source's name** instead of a picture. Checker = a model family
used nowhere else (GLM-4.6V-Flash 9B, P3's pick) with mpnet as the text matcher. **It may not reject
anything until it passes calibration** on Adam's 189 ratings, split by sound: catches ≥70% of his
"no", rejects ≤15% of his "yes", kappa ≥0.5. Until then it runs as a logged column only.

**7. Describer for the judge** becomes the same independent model as the checker, not Qwen3.8; add a
family guard in `protocol.py`.

**8. Measurement.** New sounds outside the 33 (DEV clips not in the bench), never TEST. Arms: today
vs the new setup, three seeds each. Adam rates in two passes: (1) picture alone, free text "what is
making a sound here?", scored blind against the source; (2) the source name plus one frame from the
clip: "does this fit, and does it say anything false?". ~10 exact duplicates to measure his own
consistency. Go/no-go fixed before the run: paired CI over sounds on pass-1 correctness excludes zero;
**zero confirmed false messages** (veto); no sound that is 3/3 yes today drops to 0/3.

## Questions for you
1. Sign or amend each of points 1-8, one line each. Say "sign" if you have nothing to add.
2. Two numbers are still open: REL_FLOOR and the sibling-tie margin. P1 wants them set on DEV by
   counting recovered vs spurious children; is that honest given the 33 are also where the rules were
   written, and if not, where?
3. What is the smallest version of this that can be built and put in front of Adam **today**, and
   what should wait?
