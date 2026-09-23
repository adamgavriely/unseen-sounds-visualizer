# How well does this pipeline use the video? (review, 2026-09-23)

Adam: *"how good we use the video resource, the image and video understanding etc"*, then
*"maybe it needed to see the motion itself, as in video not just images? maybe it needs to sync with
audio data?"*, then *"maybe we need a better model than OWLv2"*.

All three are fair. This is what the code actually does, what it leaves unused, what the literature
offers, and which of it is worth GPU time given a ceiling that has to be stated first.

---

## 1. What the code does with the video

    stage 2   6 frames spread evenly over the WHOLE clip -> OWLv2, queried only with the labels the
              detector already heard -> visible_entities + per-family scores + a `setting` string
    stage 5   per sound, per 5-second stretch: 6 frames spanning [start-1s, end+1s] -> Qwen3.8-27B,
              asked three questions (name / a-b / describe), majority decides "is the source visible"
    stage 6   composites the panel

That is the whole of it.

## 2. What it does NOT do

  * **No temporal encoding.** `reason._ask` builds `[{"type": "image"}, ...]` -- one entry per
    frame. Qwen2.5/3-VL align their position ids to real timestamps, but only for
    `{"type": "video"}` input with an fps. As built, the model cannot know the six pictures are
    consecutive moments, how far apart they are, or which one is the instant the sound began.
  * **No motion.** Frames are independent JPEGs; nothing computes change between them.
  * **No audio-visual synchrony.** Nothing asks whether the picture changes AT the onset.
  * **Sparse sampling.** A 5-second stretch gets 6 frames -- one every 1.4 s. A car passing behind
    the camera is on screen for about half a second.
  * **`visible_entities` is computed, read, and then deliberately overridden.** An earlier draft of
    this review said "read by nothing"; that was wrong. `plan_augmentations` does read it -- a sound
    whose label appears in the list is marked redundant. But in the shipping configuration
    `defer = redundant and VLM_VISIBILITY and DEPICTION_REASONING` is true, which collapses the
    decision back to salience alone and hands the final word to the per-sound VLM check. The code
    says why, in a comment: stage 2 is a whole-clip object pass, and it once silenced a fire alarm
    because the pull station was on screen somewhere. So the verdict is not ignored by oversight, it
    is deferred by design -- and the consequence is the same, the gate's verdict is the VLM's.
  * **SAM 3 is written and unused.** `src/stage2_video_understanding/sam3.py` exists and was listed
    in the v4 plan as OWLv2's successor; it never entered the adopted row.

## 3. What is measured, not guessed

**Sampling density is a small win.** For the 11 gate leaks where the VLM saw nothing and the
annotator saw the source plainly, OWLv2 finds the source in **4 of 11** at our current sampling and
**7 of 11** at 10 fps.

**The VLM's blindness is perception, not prompting.** What it actually answered on those stretches:

    Thunder  named='nothing','nothing','nothing'      Bell   named='nothing','nothing'
    Water    named='nothing','nothing'                Vehicle named='nothing'
    Alarm    named='nothing','fire alarm'   <- the only one that ever named it

Ten of eleven said `nothing` in EVERY stretch. There is no wording fix hiding here. On
`as_fire_alarm` OWLv2 scores **0.71** while the VLM says nothing -- the object detector sees what
the VLM does not.

**Four of the eleven are definition edges, not model failures**: Thunder x4 (the source is the sky)
and Water x2 (a whole aquarium tank). No detector "sees" thunder.

## 4. The ceiling, stated before any proposal

Gate leaks after both detector vetoes are 18 pictures on TEST, of which 11 are "VLM saw nothing".
A perfect additional visibility signal removes at most those 11: at beta = 2 that is **-0.37 cost
per clip at the absolute ceiling**, realistically half. The two detector vetoes moved cost by
**1.14 per clip**.

**So a video improvement that SUPPLEMENTS the gate is worth about a third of what the audio lever
gave.** A video improvement that REPLACES the VLM's visibility vote has a higher ceiling, because a
gate that can be trusted is what allows `DISPLAY_THRESHOLD` to come down and recall to rise -- the
same trade the per-family bars offered, but with the false-alarm side handled. Those are two
different experiments with two different ceilings and they should not be conflated.

## 5. What the literature offers

**Open-vocabulary detection.** OWLv2 (2023) is no longer the strongest option:

    SAM 3 (Meta, ICLR 2026)   promptable concept segmentation; on the open-vocabulary SA-Co/Gold
                              benchmark it scores MORE THAN DOUBLE the cgF1 of OWLv2, which the
                              paper uses as its strongest baseline, and users prefer it about 3:1.
                              It also tracks a concept THROUGH frames rather than judging each
                              alone. We already have the backend written.
    Grounding DINO            the accuracy leader among zero-shot detectors on COCO/LVIS/ODinW
    YOLO-World                ~20x faster, lower accuracy -- irrelevant for us, we are not real-time
    RF-DETR                   strongest on domain transfer, but closed-vocabulary

**Audio-visual sound source localization (VSSL)** is Adam's synchrony idea as a research field, and
2025 work targets exactly our case -- sounds whose source is NOT on screen. Reported directly:
*"SSL-TIE is good at discriminating silence and noise but less good for off-screen sources, while
SSL-Align is good at discriminating noise and off-screen."* DCASE 2025 Task 3 is literally stereo
SELD **with onscreen/offscreen classification**.

The caveat that decides whether it fits: a VSSL model answers *"where is the sound coming from?"*
from the audio. Our gate asks *"is THIS DETECTED LABEL's source on screen?"* A parked ambulance with
its siren off is present and is not the source -- which is the wrong-family problem relocated into
the visual channel. Worth an evaluation on the 11 leaks plus the 12 hits, not a week of integration.

## 6. Experiments run, and their status

  * **frames as VIDEO rather than images** -- the same model, the same frames, the same prompt, only
    the content type changed. A first attempt reported FAIL; that was void, because every video call
    had raised `fps expected float, got list`. Re-run after the fix. It tests PERCEPTION (the name
    question), not the gate verdict, which is a majority of three votes.
  * **per-stretch OWLv2 vote** -- the 2x2 of (VLM says visible) x (detector sees it in this
    stretch). The cell that matters is VLM=no, detector=yes: each one is either a wrong picture
    removed or a needed sound lost, and the rule clears beta = 2 only above 2:1. The earlier
    rejection of OWLv2 as a silencing vote was CLIP-level, with no time alignment -- a parked car
    silenced a later ambulance. Running.
  * **SAM 3 in the same harness** -- `benchmark/gold/owl_per_stretch.py --backend sam3`, ready.
    Only worth the GPU if OWLv2 clears or nearly clears 2:1, since SAM 3 changes the model, not
    the ceiling.

## 7. Not yet tried, in order of cost

1. **Local synchrony.** Frame-difference energy INSIDE the detected object's box or mask versus
   outside, at the onset. Global frame difference is confounded by camera motion; the local version
   is not, and it reuses a detection that already runs. It is the cheap form of Adam's audio-sync
   idea and it discriminates exactly the "visible but VLM-blind" cases.
2. **SAM 3 REPLACING the VLM's visibility vote**, with `DISPLAY_THRESHOLD` then lowered. This is the
   version with the real ceiling and it is a proper amendment: DEV selection, both arms, one TEST
   look.
3. **A VSSL model** as a third opinion, evaluated on the 23 stretches before any integration.

## 8. The honest summary

The pipeline uses the video as a **bag of six photographs per five seconds**, with no sense of time,
no motion, no synchrony, and with its object detector's verdict discarded. Every one of Adam's three
instincts points at a real gap. But the gate's remaining errors are small enough that fixing all of
them is worth about a third of what the detector work delivered -- unless the video signal becomes
good enough to REPLACE the VLM's judgement, at which point it buys recall rather than precision, and
that is the version worth building.
