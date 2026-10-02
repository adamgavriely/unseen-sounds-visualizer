"""Writes the SAMPLE data.js for the Decision Inspector (layout only; meta.sample = true).

The step catalogue is the shipped pipeline's real decision list (D', 2 Oct 2026). The cases are taken from the 1 Oct
error dissection (docs/review/panel_2026-10-01/dissection.md, Round 60 raw answers, the cache-fill log); answers marked
"illustrative" are made up to show the layout. export.py replaces this file with real data after the freeze.
"""
import json
from pathlib import Path

LISTEN = "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
STEPS = [
 # stage 4: hearing
 ("beats_extract", 4, "BEATs detection", "BEATs scores 2-s windows; a sound type must peak above the bar for at least 0.3 s.", "peak ≥ 0.175, length ≥ 0.3 s", "BEATs", None),
 ("flexsed_extract", 4, "FlexSED detection", "FlexSED scores 215 sound types frame by frame; a run above the bar becomes a span.", "peak ≥ 0.8, length ≥ 0.3 s", "FlexSED", None),
 ("twin_union", 4, "Twin join", "A FlexSED span and a BEATs span of the same type within 1 s become one span with the earlier start.", "same family, ≤ 1 s apart", None, None),
 ("mirror_veto", 4, "Mirror veto", "Drops a BEATs-only span when FlexSED clearly hears a different sound at that moment and not this one.", "other type ≥ 0.7 and own type < 0.4", "FlexSED", None),
 ("mirror_keep", 4, "Mirror keep (listener)", "A mirror-vetoed span comes back if the listener confirms it and an open list names it.", "Qwen V1+V2 accept, and a V4 list names it", "Qwen3-Omni", "Is the sound of {family} present in this recording? Answer yes or no."),
 ("masked_weak", 4, "Masked weak veto", "A weak BEATs-only span under speech or music is dropped unless a listener confirms it.", "conf < 0.5 and speech/music ≥ 0.3", "Qwen3-Omni / Audio Flamingo", None),
 ("dasm_clip_veto", 4, "DASM clip veto", "Drops a sound type DASM never hears anywhere in the clip, unless a listener keeps it.", "DASM clip max ≥ 0.084", "DASM", None),
 ("dasm_local_veto", 4, "Two witnesses", "A span needs DASM around it, or both listeners must name it.", "DASM max (span ± 0.5 s) ≥ 0.35, or Qwen and Audio Flamingo both name it", "DASM + Qwen3-Omni + Audio Flamingo", LISTEN),
 ("scene_margin", 4, "Scene check", "If only one listener names the sound, the vision model judges if it is credible in this scene (read from yes/no probabilities).", "yes-margin minus twin-question margin > 0, majority of stretches", "Qwen3.8-27B",
  "Could the sound of {label} plausibly be heard in this scene? Answer yes or no.  (twin: Could the sound of {label} NOT plausibly be heard in this scene?)"),
 ("k4a_inventory", 4, "Listener agreement", "Drops a picture that neither listener's open list names, unless DASM hears it clearly.", "a list names it, or DASM ≥ 0.575", "Qwen3-Omni + Audio Flamingo + DASM", LISTEN),
 ("band_twin_pull", 4, "Earlier start", "Moves a late start back to an earlier FlexSED run of the same sound.", "run ≥ 0.5 ending ≤ 1 s before", "FlexSED", None),
 ("continuation_veto", 4, "Continuation veto", "Drops a span that is only a later piece of a sound already going on.", "FlexSED run ≥ 0.5 began ≥ 1.5 s earlier and is still going", "FlexSED", None),
 ("flexsed_cross_veto", 4, "FlexSED clip veto", "Drops a sound type FlexSED never hears in the whole clip.", "FlexSED clip max ≥ 0.3", "FlexSED", None),
 ("panns_clip_veto", 4, "PANNs clip veto", "Drops a FlexSED-only span PANNs never hears in the clip, unless the listeners keep it.", "PANNs clip max ≥ 0.05", "PANNs", None),
 ("listener_keep", 4, "Listener keep", "A span the PANNs veto removed comes back if the listeners accept it.", "peak ≥ 0.6: Qwen names it; below: both name it", "Qwen3-Omni + Audio Flamingo", LISTEN),
 ("band_rescue", 4, "Listener rescue (weak FlexSED)", "A sound FlexSED heard just below its bar is added if the listeners confirm it.", "FlexSED peak 0.5–0.8; peak ≥ 0.6: Qwen names it; below: both", "Qwen3-Omni + Audio Flamingo", LISTEN),
 ("dasm_rescue", 4, "DASM rescue", "Adds a sound only DASM found when both listeners name it.", "both lists name it; type not already found", "DASM + Qwen3-Omni + Audio Flamingo", LISTEN),
 ("onset_refine", 4, "Start refinement", "Sharpens a BEATs start by masking the window's opening; can only move later.", "evidence drop ≥ 10 %", "BEATs", None),
 ("finelap_veto", 4, "FineLAP check", "A rescued sound is dropped if FineLAP does not support it.", "FineLAP ≥ 0.329", "FineLAP", None),
 ("dasm_vote", 4, "DASM vote on rescues", "A rescued sound is kept only if DASM also hears it around it.", "DASM ≥ 0.575 (span ± 0.5 s)", "DASM", None),
 ("rescue_once", 4, "One rescue per type", "Only the earliest rescued span of a sound type is kept.", "earliest per family", None, None),
 # stage 5: is it needed
 ("label_filter", 5, "Drawable sound type", "Speech, music, textures, wind and generic names are not drawn.", "depictable list", None, None),
 ("family_merge", 5, "Group by sound type", "Spans of one sound type within 1 s join; weak spans of a type that also has strong ones are dropped.", "join gap 1.0 s; weak = conf < 0.35", None, None),
 ("display_bar", 5, "Display bar", "A sound below the confidence bar is planned but not shown.", "conf ≥ 0.35", None, None),
 ("speech_rescue", 5, "Talked-about rescue", "A weak sound people react to in speech is shown.", "both letter orders say 'reacting'", "Qwen3.8-27B (text)", "A sound of {label} was heard in a video. Around that moment, someone said: \"{speech}\". Are they (a) reacting to that sound or talking about it, or (b) not referring to that sound? Answer with the letter only."),
 ("gate", 5, "Visibility gate", "The vision model judges, per ≤ 5-s stretch, if the sound's source is visible. The sound is silenced only if every stretch is seen.", "majority of 3 questions per stretch", "Qwen3.8-27B",
  "Q-name: These frames are from the moment a sound of {label} was heard. Name the thing in these frames that is making that sound… If nothing … is visible, answer exactly: nothing.\nQ-ab: (a) you can SEE {label} happening on screen … (b) {label} is not visibly happening in these frames. Answer with the letter only. (both orders)\nQ-desc: Describe what is happening in these frames in one sentence."),
 ("family_rule", 5, "Family rule", "A sound is silenced when a more specific related sound at the same time was silenced as visible.", "visible specific silences general, never the reverse", None, None),
 ("disambiguate", 5, "Same event, two names", "Two labels on one event: the frames pick one.", "both orders agree", "Qwen3.8-27B", None),
 ("dedup", 5, "Duplicate pictures", "Sounds that would give the same picture are merged.", "SigLIP similarity ≥ 0.80", "SigLIP", None),
 # stage 6: display
 ("depict_event", 6, "Event on screen", "Drops a picture whose event is visibly happening and whose visible maker could make that sound.", "event yes/no both ways, and look-alike yes twice", "Qwen3.8-27B", None),
 ("display_join", 6, "Repeat merge", "Two pictures of the same sound closer than the gap become one.", "gap ≤ 2.5 s", None, None),
 ("group", 6, "Smart grouping", "Two same-type pictures ≤ 8 s apart merge if the second is the same continuing sound.", "both orders answer 'same'", "Qwen3-Omni",
  "You hear a {lab} sound near the start and again near the end of this recording. Is the {lab} near the end the same continuing sound as at the start, or a new, separate event? Answer with exactly one word."),
 ("max_slots", 6, "At most 3 at once", "More than 3 pictures at the same moment: the weakest waits.", "≤ 3", None, None),
 ("scorer", 7, "Scoring", "A picture is a hit if it is the right sound type and starts 0.5 s before to 1.0 s after the needed sound.", "−0.5 … +1.0 s", None, None),
 ("never_heard", 4, "Never heard", "No detector produced any span of this sound type near its start.", "—", None, None),
]
steps = [dict(id=a, stage=b, name=c, plain=d, bar=e, model=f, question=g) for a, b, c, d, e, f, g in STEPS]


# ---- the current build in plain words (shown on the Build tab); stage numbers match STEPS
BUILD = [
 {"part": "1 · Hear", "stages": [4], "what": "Find every non-speech sound and when it starts.",
  "how": "Two sound detectors scan the audio: BEATs (2-s windows) and FlexSED (frame by frame, 215 sound types). A candidate must be heard strongly enough and last at least 0.3 s. Speech is transcribed by Whisper and goes to subtitles, not pictures.",
  "models": "BEATs, FlexSED, Whisper"},
 {"part": "2 · Confirm", "stages": [4], "what": "Throw out false alarms; win back real sounds the detectors heard only weakly.",
  "how": "Two audio-language models (Qwen3-Omni and Audio Flamingo) listen and list the sounds they hear. Two more detectors (DASM, FineLAP) double-check. A sound needs two witnesses: DASM, or both listeners. If only one listener names it, the vision model checks that the sound is credible in the scene.",
  "models": "Qwen3-Omni, Audio Flamingo Next, DASM, FineLAP, PANNs"},
 {"part": "3 · Is it needed?", "stages": [5], "what": "Keep only sounds a deaf viewer cannot see the source of.",
  "how": "Speech, music and background textures are skipped. Weak sounds (confidence < 0.35) are skipped. A vision model looks at frames around the sound and asks three questions (name the source; is it visibly happening; describe the scene). If the source is visible in every stretch, the sound is silenced.",
  "models": "Qwen3.8-27B (vision)"},
 {"part": "4 · Draw", "stages": [5, 6], "what": "Choose what to draw and draw it clearly.",
  "how": "The thing to draw comes from fixed word lists (so no invented objects). Qwen-Image draws it. The vision model checks the picture with a multiple-choice question; a wrong picture is redrawn up to 5 times, then replaced by a word card (e.g. WHOOSH).",
  "models": "Qwen-Image-2512, Qwen3.8-27B"},
 {"part": "5 · Show", "stages": [6], "what": "Show each picture once, at the right moment.",
  "how": "Repeats of one sound within 2.5 s become one picture. Qwen3-Omni decides if two pictures up to 8 s apart are one continuing sound. A picture is dropped if its event is visibly happening on screen. At most 3 pictures at once; each stays 1.5 s, at most 1 s past the sound.",
  "models": "Qwen3-Omni, Qwen3.8-27B"},
]
BUILD_STEPS = [
 ("Old pipeline (28 Sept)", "18 / 51 / 3.690", "21 / 40 / 2.909", "before this month's detector work"),
 ("+ listeners rescue weak sounds", "26 / 45 / 3.070", "22 / 38 / 2.818", ""),
 ("+ listener and DASM checks, continuation rule, FineLAP", "28 / 24 / 2.366", "24 / 31 / 2.568", ""),
 ("+ listener agreement", "28 / 21 / 2.282", "23 / 29 / 2.568", ""),
 ("+ repeat merge 2.5 s, smart grouping, 0.3-s minimum", "29 / 18 / 2.141", "22 / 26 / 2.545", "version B"),
 ("+ two witnesses + scene check", "29 / 14 / 2.028", "24 / 23 / 2.386", "version D"),
 ("scene check read from probabilities (shipped)", "29 / 15 / 2.056", "24 / 24 / 2.409", "version D′, frozen 2 Oct"),
]
LESSON = ("The remaining errors are where the models disagree at about chance level, and where 'the object is on screen' is not "
          "'you can see the sound happening' (church tower vs ringing bell). That is the honest limit of today's open models.")
# ---- what was tried and why it failed (Tried tab). result = DEV hits / wrong / cost where known
TRIED = [
 ("Hear", "Newer sound taggers (EAT-large, Dasheng)", "found more sounds, more false alarms; failed the held-out check", "no"),
 ("Hear", "Meta PE-A-Frame detector", "near chance on held-out clips (AUROC 0.53)", "no"),
 ("Hear", "DASM as the main detector", "its confidence is flat, so it cannot time a sound", "no"),
 ("Hear", "Lower FlexSED bar", "+3 hits but +7 wrong pictures", "no"),
 ("Hear", "Sound separation first (SAM-Audio, Demucs)", "music was not removed (quality check failed)", "no"),
 ("Hear", "Onset clean-up (SEBB, 2024)", "lost 7 hits", "no"),
 ("Hear", "Short-sound path (Round 56)", "+1 hit (a gasp) but +18 wrong (27 / 36 / 2.761)", "no"),
 ("Hear", "Hear more weak sounds (band runs, tagger ensemble, rooster rule, tighter listener cut; Rounds 58, 63)", "the extra sounds are mostly wrong: a sound no listener names is right only 22 % of the time (held-out)", "no"),
 ("Confirm", "Listen with ±5 s of audio and video context (Round 59)", "Omni names what it SEES, not what it hears (3 of 9 right on held-out)", "no"),
 ("Confirm", "Newer audio LLMs as listeners (Kimi-Audio, Step-Audio-2, MOSS-Audio, SpotSound)", "3 to 10 wrong pictures per extra hit", "no"),
 ("Confirm", "Listener lists the whole clip's sounds (EXPECT)", "DEV +3 hits, but on TEST +2 hits and +6 wrong; the gate cannot refuse a wrong off-screen name", "no"),
 ("Confirm", "Second listener must agree (Round 45)", "the second listener hears the same wrong names", "no"),
 ("Confirm", "Two witnesses alone (Round 53)", "−5 wrong but −2 hits; fixed later by the scene check (shipped as D)", "partly"),
 ("Confirm", "FlexSED as a witness, tight re-ask (Rounds 58, 58b)", "stopped at the held-out check", "no"),
 ("Is it needed?", "Object detectors for visibility (CLIP, SigLIP, OWLv2, SAM 3)", "about chance: seeing an object is not seeing a sound's source", "no"),
 ("Is it needed?", "Other vision models (Gemma-4-31B, Qwen2.5-VL)", "lower agreement with the human labels", "no"),
 ("Is it needed?", "Box the source and ask about the crop (BOX, BOX-2)", "+1 hit (church bell) but +5 wrong", "no"),
 ("Is it needed?", "Motion-sound sync (Synchformer)", "kept 3 needed sounds but lost 5 correct 'visible' calls", "no"),
 ("Is it needed?", "Ask 'is the source acting right now?' (HUMAN-2)", "one sound short of the bar set in advance", "no"),
 ("Is it needed?", "Look-alike reasoning (Round 61)", "it also explained away a real crying sound by a visible parrot", "no"),
 ("Is it needed?", "Name the top 3 candidates, zoom, ask 'making the sound now?' (Round 66)", "26 / 12 / 2.141 vs D′ 29 / 15: −3 wrong but −3 hits; the bell was not rescued", "no"),
 ("Is it needed?", "Bell rule: sound from a hidden part of a visible thing (Round 64)", "passed DEV (30 / 14 / 1.972) but in practice covers one sound type; only general rules are allowed", "not shipped"),
 ("Is it needed?", "Does the visible effect show the sound? (Rounds 54, 62)", "the vision model answered by letter position or always 'no'; read as a probability it was just under the bar (0.647 vs 0.65)", "no"),
 ("Is it needed?", "Multi-step: name it, find it (box), zoom, 'making the sound now?' (Rounds 50, 50L)", "the zoom works when a source is named, but naming and zoomed answers are too noisy", "no"),
 ("Is it needed?", "Bigger or other vision models (model research panel)", "every model moves along the same trade-off; the question (object present vs object making the sound) is the limit. A 235B model is future work (no disk)", "no"),
 ("Show", "Stricter repeat rules (RPT-S, DBR)", "no net gain", "no"),
 ("Show", "Detect pauses inside a sound (Rounds 51, 55)", "the models cannot time a 2-s pause", "no"),
 ("Show", "Move late pictures to the first faint trace (GBTP)", "lost a hit", "no"),
 ("Show", "Rename a picture to what the listener hears (AVNAME)", "the namer names the loudest thing: −13 hits", "no"),
]

P = lambda step, res, value=None, bar=None, note=None, asks=None: {k: v for k, v in dict(step=step, res=res, value=value, bar=bar, note=note, asks=asks).items() if v is not None}
ILL = "illustrative answer (sample)"
clips = [
 {"clip": "bell_miami", "split": "DEV", "dur": 15.0, "video": "media/DEV/bell_miami.mp4",
  "gold": [{"id": "g1", "label": "Bell", "start": 0.2, "end": 14.5, "needed": True, "outcome": "miss", "lost_at": "gate",
            "why": "Gate: 3 of 3 questions said the source is on screen (a bell tower is visible; the ringing bell is a different one).", "cands": ["c1"]}],
  "pictures": [],
  "cands": [{"id": "c1", "label": "Church bell", "start": 0.22, "end": 3.2, "origin": "beats + flexsed", "fate": "dropped", "at": "gate", "trail": [
     P("beats_extract", "pass", "0.70", "≥ 0.175"), P("flexsed_extract", "pass", "0.98", "≥ 0.8"), P("twin_union", "pass", note="BEATs and FlexSED agree"),
     P("dasm_local_veto", "pass", "DASM 0.92", "≥ 0.35"), P("display_bar", "pass", "0.70", "≥ 0.35"),
     P("gate", "drop", "3 of 3 seen", "silenced only if every stretch is seen", asks=[
        {"who": "Q-name · stretch 1", "q": "Name the thing in these frames that is making that sound…", "a": "church bell", "vote": "seen"},
        {"who": "Q-ab · both orders", "q": "(a) you can SEE Church bell happening on screen … (b) not visibly happening", "a": "(a), (a)", "vote": "seen"},
        {"who": "Q-desc", "q": "Describe what is happening in these frames in one sentence.", "a": "A white church bell tower stands against a blue sky.", "vote": "seen"}],
      note="Q-name is real (the gate named 'church bell'); the other two answers are " + ILL + ".")]}]},
 {"clip": "tg_d029", "split": "DEV", "dur": 15.0, "video": "media/DEV/tg_d029.mp4",
  "gold": [{"id": "g1", "label": "Chicken, rooster", "start": 6.9, "end": 14.8, "needed": True, "outcome": "miss", "lost_at": "mirror_veto",
            "why": "Mirror veto: FlexSED heard a different sound strongly at that moment, and the rooster only at 0.26 (< 0.4).", "cands": ["c1"]}],
  "pictures": [],
  "cands": [{"id": "c1", "label": "Crowing, cock-a-doodle-doo", "start": 6.9, "end": 8.4, "origin": "beats", "fate": "dropped", "at": "mirror_veto", "trail": [
     P("beats_extract", "pass", "0.81", "≥ 0.175"), P("flexsed_extract", "skip", "0.26", "≥ 0.8", "FlexSED alone did not hear it"),
     P("mirror_veto", "drop", "own type 0.26; other type above 0.7", "drop if other ≥ 0.7 and own < 0.4", "the other type's name and score will come from the log"),
     P("mirror_keep", "skip", note="the listener lists did not name it, so it stays dropped (from the dissection: Qwen list no, Audio Flamingo list yes)")]}]},
 {"clip": "ambient_citywalk_nyc_1689", "split": "DEV", "dur": 18.0, "video": "media/DEV/ambient_citywalk_nyc_1689.mp4",
  "gold": [{"id": "g1", "label": "Vehicle (car horn)", "start": 3.8, "end": 4.3, "needed": True, "outcome": "miss", "lost_at": "band_rescue",
            "why": "FlexSED heard it at 0.71 (below 0.8). Both listeners were asked about that stretch and neither listed a vehicle.", "cands": ["c1"]}],
  "pictures": [],
  "cands": [{"id": "c1", "label": "Vehicle", "start": 0.0, "end": 10.0, "origin": "flexsed band run", "fate": "dropped", "at": "band_rescue", "trail": [
     P("beats_extract", "skip", "0.10", "≥ 0.175", "BEATs too weak"), P("flexsed_extract", "skip", "0.71", "≥ 0.8", "below FlexSED's own bar, so it goes to the listeners"),
     P("band_rescue", "drop", "peak 0.56 → Qwen and Audio Flamingo must both name it", "0.5–0.8", asks=[
        {"who": "Qwen3-Omni V4 on 0.0–10.0 s", "q": LISTEN, "a": "(no vehicle in the list)", "vote": "no"},
        {"who": "Audio Flamingo V4 on 0.0–10.0 s", "q": LISTEN, "a": "(no vehicle in the list)", "vote": "no"}],
      note="Real outcome from the 1 Oct cache fill (TIER no: Qwen V4 no, AF V4 no); the list texts will come from the log.")]}]},
 {"clip": "as_explosion_XJ8lc3I6", "split": "DEV", "dur": 12.0, "video": "media/DEV/as_explosion_XJ8lc3I6.mp4",
  "gold": [{"id": "g1", "label": "Gasp", "start": 6.7, "end": 7.0, "needed": True, "outcome": "miss", "lost_at": "beats_extract",
            "why": "BEATs heard it (0.42) but only for 0.25 s, shorter than the 0.3-s minimum.", "cands": []}],
  "pictures": [], "cands": []},
 {"clip": "tg_d107", "split": "DEV", "dur": 12.0, "video": "media/DEV/tg_d107.mp4",
  "gold": [{"id": "g1", "label": "Bird call", "start": 6.0, "end": 8.9, "needed": False, "visible": True},
           {"id": "g2", "label": "Laughter", "start": 8.2, "end": 9.0, "needed": True, "outcome": "hit", "cands": []}],
  "pictures": [{"id": "p1", "label": "Screaming", "start": 6.52, "end": 9.0, "verdict": "cross", "gold": "g1", "gold_text": "Bird call 6.0–8.9 (visible)", "cand": "c1"}],
  "cands": [{"id": "c1", "label": "Screaming", "start": 6.52, "end": 9.0, "origin": "beats", "fate": "drawn", "trail": [
     P("beats_extract", "pass", "0.4", "≥ 0.175", ILL), P("dasm_local_veto", "pass", "DASM below 0.35; one listener names it", "≥ 0.35 or both name it", "goes to the scene check"),
     P("scene_margin", "pass", "credible", "margin > 0", asks=[
        {"who": "Qwen3.8-27B, stretch 7.2–9.0 s", "q": "Could the sound of screaming plausibly be heard in this scene? Answer yes or no.", "a": "yes", "vote": "yes"}],
      note="Real answer from Round 60. The bird making the sound is on screen, and screaming is credible there, so the picture survives."),
     P("display_bar", "pass", "≥ 0.35", "≥ 0.35"),
     P("gate", "pass", "not seen", "silenced only if every stretch is seen", note="No screaming person is visible, so the gate keeps it (" + ILL + ")."),
     P("scorer", "cross", "cross", "−0.5 … +1.0 s", "No needed screaming here; the sound is a visible bird. Counted as a wrong picture.")]}]},
 {"clip": "ambient_weather_storm_7200", "split": "DEV", "dur": 16.0, "video": "media/DEV/ambient_weather_storm_7200.mp4",
  "gold": [{"id": "g1", "label": "Thunder", "start": 0.0, "end": 6.0, "needed": False, "visible": True}],
  "pictures": [{"id": "p1", "label": "Thunder", "start": 0.06, "end": 3.0, "verdict": "visible", "gold": "g1", "cand": "c1"}],
  "cands": [{"id": "c1", "label": "Thunder", "start": 0.06, "end": 3.0, "origin": "beats + flexsed", "fate": "drawn", "trail": [
     P("beats_extract", "pass", "0.6", "≥ 0.175", ILL), P("dasm_local_veto", "pass", "0.8", "≥ 0.35", ILL), P("display_bar", "pass", "0.6", "≥ 0.35"),
     P("gate", "pass", "1 of 3 seen", "silenced only if every stretch is seen", asks=[
        {"who": "Q-name", "q": "Name the thing in these frames that is making that sound…", "a": "nothing", "vote": "not seen"},
        {"who": "Q-ab · both orders", "q": "(a) you can SEE Thunder happening … (b) not visibly happening", "a": "(b), (a)", "vote": "split"},
        {"who": "Q-desc", "q": "Describe what is happening in these frames in one sentence.", "a": "Dark storm clouds over a flat field.", "vote": "not seen"}],
      note="All three answers are " + ILL + ". The annotator saw lightning at that moment; the frames sampled may miss it."),
     P("scorer", "visible", "visible", note="The gold marks this thunder as visible, so the picture counts as wrong.")]}]},
 {"clip": "tg_d032", "split": "DEV", "dur": 16.0, "video": "media/DEV/tg_d032.mp4",
  "gold": [{"id": "g1", "label": "Thunder", "start": 13.6, "end": 15.5, "needed": True, "outcome": "hit", "cands": ["c1"]}],
  "pictures": [{"id": "p1", "label": "Thunder", "start": 13.75, "end": 15.5, "verdict": "hit", "gold": "g1", "cand": "c1"}],
  "cands": [{"id": "c1", "label": "Thunder", "start": 13.75, "end": 14.75, "origin": "beats", "fate": "drawn", "trail": [
     P("beats_extract", "pass"), P("dasm_local_veto", "pass", "DASM 0.01; only Qwen names it", "≥ 0.35 or both name it", "goes to the scene check"),
     P("scene_margin", "pass", "credible", "margin > 0", asks=[
        {"who": "Qwen3.8-27B, 13.75–14.75 s", "q": "Could the sound of thunder plausibly be heard in this scene? Answer yes or no.", "a": "yes", "vote": "yes"}],
      note="Real answer from Round 60: this is the hit the scene check won back."),
     P("gate", "pass", "not seen"), P("scorer", "hit", "hit", "−0.5 … +1.0 s")]}]},
]
data = {"meta": {"version": "D′", "arm": "SHIP8+MD3+WW5+SL", "built": "sample, 2 Oct 2026", "sample": True, "window": [-0.5, 1.0],
                 "cost": "(4·miss + 2·wrong)/clips"},
        "sets": {"DEV": {"clips": 71, "needed": 58, "hits": 29, "wrong": 15, "visible": 6, "cross": 7, "phantom": 2, "cost": 2.056},
                 "TEST": {"clips": 88, "needed": 65, "hits": 24, "wrong": 24, "visible": 4, "cross": 15, "phantom": 5, "cost": 2.409}},
        "steps": steps, "clips": clips,
        "build": BUILD, "lesson": LESSON, "build_steps": [dict(zip(("change", "dev", "test", "note"), r)) for r in BUILD_STEPS],
        "tried": [dict(zip(("part", "idea", "why", "shipped"), r)) for r in TRIED]}
content = {"build": data.pop("build"), "lesson": data.pop("lesson"), "build_steps": data.pop("build_steps"), "tried": data.pop("tried")}
for name, var, obj in (("content.js", "INSPECTOR2_CONTENT", content), ("data_sample.js", "INSPECTOR2", data)):
    out = Path(__file__).with_name(name)
    out.write_text("window." + var + " = " + json.dumps(obj, ensure_ascii=False, indent=1) + ";\n", encoding="utf-8")
    print("wrote", out)
