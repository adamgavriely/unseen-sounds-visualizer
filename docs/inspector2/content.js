window.INSPECTOR2_CONTENT = {
 "build": [
  {
   "part": "1 · Hear",
   "stages": [
    4
   ],
   "what": "Find every non-speech sound and when it starts.",
   "how": "Two sound detectors scan the audio: BEATs (2-s windows) and FlexSED (frame by frame, 215 sound types). A candidate must be heard strongly enough and last at least 0.3 s. Speech is transcribed by Whisper and goes to subtitles, not pictures.",
   "models": "BEATs, FlexSED, Whisper"
  },
  {
   "part": "2 · Confirm",
   "stages": [
    4
   ],
   "what": "Throw out false alarms; win back real sounds the detectors heard only weakly.",
   "how": "Two audio-language models (Qwen3-Omni and Audio Flamingo) listen and list the sounds they hear. Two more detectors (DASM, FineLAP) double-check. A sound needs two witnesses: DASM, or both listeners. If only one listener names it, the vision model checks that the sound is credible in the scene.",
   "models": "Qwen3-Omni, Audio Flamingo Next, DASM, FineLAP, PANNs"
  },
  {
   "part": "3 · Is it needed?",
   "stages": [
    5
   ],
   "what": "Keep only sounds a deaf viewer cannot see the source of.",
   "how": "Speech, music and background textures are skipped. Weak sounds (confidence < 0.35) are skipped. A vision model looks at frames around the sound and asks three questions (name the source; is it visibly happening; describe the scene). If the source is visible in every stretch, the sound is silenced.",
   "models": "Qwen3.8-27B (vision)"
  },
  {
   "part": "4 · Draw",
   "stages": [
    5,
    6
   ],
   "what": "Choose what to draw and draw it clearly.",
   "how": "The thing to draw comes from fixed word lists (so no invented objects). Qwen-Image draws it. The vision model checks the picture with a multiple-choice question; a wrong picture is redrawn up to 5 times, then replaced by a word card (e.g. WHOOSH).",
   "models": "Qwen-Image-2512, Qwen3.8-27B"
  },
  {
   "part": "5 · Show",
   "stages": [
    6
   ],
   "what": "Show each picture once, at the right moment.",
   "how": "Repeats of one sound within 2.5 s become one picture. Qwen3-Omni decides if two pictures up to 8 s apart are one continuing sound. A picture is dropped if its event is visibly happening on screen. At most 3 pictures at once; each stays 1.5 s, at most 1 s past the sound.",
   "models": "Qwen3-Omni, Qwen3.8-27B"
  }
 ],
 "lesson": "The remaining errors are where the models disagree at about chance level, and where 'the object is on screen' is not 'you can see the sound happening' (church tower vs ringing bell). That is the honest limit of today's open models.",
 "build_steps": [
  {
   "change": "Old pipeline (28 Sept)",
   "dev": "18 / 51 / 3.690",
   "test": "21 / 40 / 2.909",
   "note": "before this month's detector work"
  },
  {
   "change": "+ listeners rescue weak sounds",
   "dev": "26 / 45 / 3.070",
   "test": "22 / 38 / 2.818",
   "note": ""
  },
  {
   "change": "+ listener and DASM checks, continuation rule, FineLAP",
   "dev": "28 / 24 / 2.366",
   "test": "24 / 31 / 2.568",
   "note": ""
  },
  {
   "change": "+ listener agreement",
   "dev": "28 / 21 / 2.282",
   "test": "23 / 29 / 2.568",
   "note": ""
  },
  {
   "change": "+ repeat merge 2.5 s, smart grouping, 0.3-s minimum",
   "dev": "29 / 18 / 2.141",
   "test": "22 / 26 / 2.545",
   "note": "version B"
  },
  {
   "change": "+ two witnesses + scene check",
   "dev": "29 / 14 / 2.028",
   "test": "24 / 23 / 2.386",
   "note": "version D"
  },
  {
   "change": "scene check read from probabilities (shipped)",
   "dev": "29 / 15 / 2.056",
   "test": "24 / 24 / 2.409",
   "note": "version D′, frozen 2 Oct"
  }
 ],
 "tried": [
  {
   "part": "Hear",
   "idea": "Newer sound taggers (EAT-large, Dasheng)",
   "why": "found more sounds, more false alarms; failed the held-out check",
   "shipped": "no"
  },
  {
   "part": "Hear",
   "idea": "Meta PE-A-Frame detector",
   "why": "near chance on held-out clips (AUROC 0.53)",
   "shipped": "no"
  },
  {
   "part": "Hear",
   "idea": "DASM as the main detector",
   "why": "its confidence is flat, so it cannot time a sound",
   "shipped": "no"
  },
  {
   "part": "Hear",
   "idea": "Lower FlexSED bar",
   "why": "+3 hits but +7 wrong pictures",
   "shipped": "no"
  },
  {
   "part": "Hear",
   "idea": "Sound separation first (SAM-Audio, Demucs)",
   "why": "music was not removed (quality check failed)",
   "shipped": "no"
  },
  {
   "part": "Hear",
   "idea": "Onset clean-up (SEBB, 2024)",
   "why": "lost 7 hits",
   "shipped": "no"
  },
  {
   "part": "Hear",
   "idea": "Short-sound path (Round 56)",
   "why": "+1 hit (a gasp) but +18 wrong (27 / 36 / 2.761)",
   "shipped": "no"
  },
  {
   "part": "Hear",
   "idea": "Hear more weak sounds (band runs, tagger ensemble, rooster rule, tighter listener cut; Rounds 58, 63)",
   "why": "the extra sounds are mostly wrong: a sound no listener names is right only 22 % of the time (held-out)",
   "shipped": "no"
  },
  {
   "part": "Confirm",
   "idea": "Listen with ±5 s of audio and video context (Round 59)",
   "why": "Omni names what it SEES, not what it hears (3 of 9 right on held-out)",
   "shipped": "no"
  },
  {
   "part": "Confirm",
   "idea": "Newer audio LLMs as listeners (Kimi-Audio, Step-Audio-2, MOSS-Audio, SpotSound)",
   "why": "3 to 10 wrong pictures per extra hit",
   "shipped": "no"
  },
  {
   "part": "Confirm",
   "idea": "Listener lists the whole clip's sounds (EXPECT)",
   "why": "DEV +3 hits, but on TEST +2 hits and +6 wrong; the gate cannot refuse a wrong off-screen name",
   "shipped": "no"
  },
  {
   "part": "Confirm",
   "idea": "Second listener must agree (Round 45)",
   "why": "the second listener hears the same wrong names",
   "shipped": "no"
  },
  {
   "part": "Confirm",
   "idea": "Two witnesses alone (Round 53)",
   "why": "−5 wrong but −2 hits; fixed later by the scene check (shipped as D)",
   "shipped": "partly"
  },
  {
   "part": "Confirm",
   "idea": "FlexSED as a witness, tight re-ask (Rounds 58, 58b)",
   "why": "stopped at the held-out check",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Object detectors for visibility (CLIP, SigLIP, OWLv2, SAM 3)",
   "why": "about chance: seeing an object is not seeing a sound's source",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Other vision models (Gemma-4-31B, Qwen2.5-VL)",
   "why": "lower agreement with the human labels",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Box the source and ask about the crop (BOX, BOX-2)",
   "why": "+1 hit (church bell) but +5 wrong",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Motion-sound sync (Synchformer)",
   "why": "kept 3 needed sounds but lost 5 correct 'visible' calls",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Ask 'is the source acting right now?' (HUMAN-2)",
   "why": "one sound short of the bar set in advance",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Look-alike reasoning (Round 61)",
   "why": "it also explained away a real crying sound by a visible parrot",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Name the top 3 candidates, zoom, ask 'making the sound now?' (Round 66)",
   "why": "26 / 12 / 2.141 vs D′ 29 / 15: −3 wrong but −3 hits; the bell was not rescued",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Bell rule: sound from a hidden part of a visible thing (Round 64)",
   "why": "passed DEV (30 / 14 / 1.972) but in practice covers one sound type; only general rules are allowed",
   "shipped": "not shipped"
  },
  {
   "part": "Is it needed?",
   "idea": "Does the visible effect show the sound? (Rounds 54, 62)",
   "why": "the vision model answered by letter position or always 'no'; read as a probability it was just under the bar (0.647 vs 0.65)",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Multi-step: name it, find it (box), zoom, 'making the sound now?' (Rounds 50, 50L)",
   "why": "the zoom works when a source is named, but naming and zoomed answers are too noisy",
   "shipped": "no"
  },
  {
   "part": "Is it needed?",
   "idea": "Bigger or other vision models (model research panel)",
   "why": "every model moves along the same trade-off; the question (object present vs object making the sound) is the limit. A 235B model is future work (no disk)",
   "shipped": "no"
  },
  {
   "part": "Show",
   "idea": "Stricter repeat rules (RPT-S, DBR)",
   "why": "no net gain",
   "shipped": "no"
  },
  {
   "part": "Show",
   "idea": "Detect pauses inside a sound (Rounds 51, 55)",
   "why": "the models cannot time a 2-s pause",
   "shipped": "no"
  },
  {
   "part": "Show",
   "idea": "Move late pictures to the first faint trace (GBTP)",
   "why": "lost a hit",
   "shipped": "no"
  },
  {
   "part": "Show",
   "idea": "Rename a picture to what the listener hears (AVNAME)",
   "why": "the namer names the loudest thing: −13 hits",
   "shipped": "no"
  }
 ]
};
