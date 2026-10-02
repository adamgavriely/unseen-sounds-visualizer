window.INSPECTOR2 = {
 "meta": {
  "version": "D′",
  "arm": "SHIP8+MD3+WW5+SL",
  "built": "sample, 2 Oct 2026",
  "sample": true,
  "window": [
   -0.5,
   1.0
  ],
  "cost": "(4·miss + 2·wrong)/clips"
 },
 "sets": {
  "DEV": {
   "clips": 71,
   "needed": 58,
   "hits": 29,
   "wrong": 15,
   "visible": 6,
   "cross": 7,
   "phantom": 2,
   "cost": 2.056
  },
  "TEST": {
   "clips": 88,
   "needed": 65,
   "hits": 24,
   "wrong": 24,
   "visible": 4,
   "cross": 15,
   "phantom": 5,
   "cost": 2.409
  }
 },
 "steps": [
  {
   "id": "beats_extract",
   "stage": 4,
   "name": "BEATs detection",
   "plain": "BEATs scores 2-s windows; a sound type must peak above the bar for at least 0.3 s.",
   "bar": "peak ≥ 0.175, length ≥ 0.3 s",
   "model": "BEATs",
   "question": null
  },
  {
   "id": "flexsed_extract",
   "stage": 4,
   "name": "FlexSED detection",
   "plain": "FlexSED scores 215 sound types frame by frame; a run above the bar becomes a span.",
   "bar": "peak ≥ 0.8, length ≥ 0.3 s",
   "model": "FlexSED",
   "question": null
  },
  {
   "id": "twin_union",
   "stage": 4,
   "name": "Twin join",
   "plain": "A FlexSED span and a BEATs span of the same type within 1 s become one span with the earlier start.",
   "bar": "same family, ≤ 1 s apart",
   "model": null,
   "question": null
  },
  {
   "id": "mirror_veto",
   "stage": 4,
   "name": "Mirror veto",
   "plain": "Drops a BEATs-only span when FlexSED clearly hears a different sound at that moment and not this one.",
   "bar": "other type ≥ 0.7 and own type < 0.4",
   "model": "FlexSED",
   "question": null
  },
  {
   "id": "mirror_keep",
   "stage": 4,
   "name": "Mirror keep (listener)",
   "plain": "A mirror-vetoed span comes back if the listener confirms it and an open list names it.",
   "bar": "Qwen V1+V2 accept, and a V4 list names it",
   "model": "Qwen3-Omni",
   "question": "Is the sound of {family} present in this recording? Answer yes or no."
  },
  {
   "id": "masked_weak",
   "stage": 4,
   "name": "Masked weak veto",
   "plain": "A weak BEATs-only span under speech or music is dropped unless a listener confirms it.",
   "bar": "conf < 0.5 and speech/music ≥ 0.3",
   "model": "Qwen3-Omni / Audio Flamingo",
   "question": null
  },
  {
   "id": "dasm_clip_veto",
   "stage": 4,
   "name": "DASM clip veto",
   "plain": "Drops a sound type DASM never hears anywhere in the clip, unless a listener keeps it.",
   "bar": "DASM clip max ≥ 0.084",
   "model": "DASM",
   "question": null
  },
  {
   "id": "dasm_local_veto",
   "stage": 4,
   "name": "Two witnesses",
   "plain": "A span needs DASM around it, or both listeners must name it.",
   "bar": "DASM max (span ± 0.5 s) ≥ 0.35, or Qwen and Audio Flamingo both name it",
   "model": "DASM + Qwen3-Omni + Audio Flamingo",
   "question": "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
  },
  {
   "id": "scene_margin",
   "stage": 4,
   "name": "Scene check",
   "plain": "If only one listener names the sound, the vision model judges if it is credible in this scene (read from yes/no probabilities).",
   "bar": "yes-margin minus twin-question margin > 0, majority of stretches",
   "model": "Qwen3.8-27B",
   "question": "Could the sound of {label} plausibly be heard in this scene? Answer yes or no.  (twin: Could the sound of {label} NOT plausibly be heard in this scene?)"
  },
  {
   "id": "k4a_inventory",
   "stage": 4,
   "name": "Listener agreement",
   "plain": "Drops a picture that neither listener's open list names, unless DASM hears it clearly.",
   "bar": "a list names it, or DASM ≥ 0.575",
   "model": "Qwen3-Omni + Audio Flamingo + DASM",
   "question": "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
  },
  {
   "id": "band_twin_pull",
   "stage": 4,
   "name": "Earlier start",
   "plain": "Moves a late start back to an earlier FlexSED run of the same sound.",
   "bar": "run ≥ 0.5 ending ≤ 1 s before",
   "model": "FlexSED",
   "question": null
  },
  {
   "id": "continuation_veto",
   "stage": 4,
   "name": "Continuation veto",
   "plain": "Drops a span that is only a later piece of a sound already going on.",
   "bar": "FlexSED run ≥ 0.5 began ≥ 1.5 s earlier and is still going",
   "model": "FlexSED",
   "question": null
  },
  {
   "id": "flexsed_cross_veto",
   "stage": 4,
   "name": "FlexSED clip veto",
   "plain": "Drops a sound type FlexSED never hears in the whole clip.",
   "bar": "FlexSED clip max ≥ 0.3",
   "model": "FlexSED",
   "question": null
  },
  {
   "id": "panns_clip_veto",
   "stage": 4,
   "name": "PANNs clip veto",
   "plain": "Drops a FlexSED-only span PANNs never hears in the clip, unless the listeners keep it.",
   "bar": "PANNs clip max ≥ 0.05",
   "model": "PANNs",
   "question": null
  },
  {
   "id": "listener_keep",
   "stage": 4,
   "name": "Listener keep",
   "plain": "A span the PANNs veto removed comes back if the listeners accept it.",
   "bar": "peak ≥ 0.6: Qwen names it; below: both name it",
   "model": "Qwen3-Omni + Audio Flamingo",
   "question": "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
  },
  {
   "id": "band_rescue",
   "stage": 4,
   "name": "Listener rescue (weak FlexSED)",
   "plain": "A sound FlexSED heard just below its bar is added if the listeners confirm it.",
   "bar": "FlexSED peak 0.5–0.8; peak ≥ 0.6: Qwen names it; below: both",
   "model": "Qwen3-Omni + Audio Flamingo",
   "question": "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
  },
  {
   "id": "dasm_rescue",
   "stage": 4,
   "name": "DASM rescue",
   "plain": "Adds a sound only DASM found when both listeners name it.",
   "bar": "both lists name it; type not already found",
   "model": "DASM + Qwen3-Omni + Audio Flamingo",
   "question": "List every distinct non-speech sound you hear in this recording, one per line, most prominent first."
  },
  {
   "id": "onset_refine",
   "stage": 4,
   "name": "Start refinement",
   "plain": "Sharpens a BEATs start by masking the window's opening; can only move later.",
   "bar": "evidence drop ≥ 10 %",
   "model": "BEATs",
   "question": null
  },
  {
   "id": "finelap_veto",
   "stage": 4,
   "name": "FineLAP check",
   "plain": "A rescued sound is dropped if FineLAP does not support it.",
   "bar": "FineLAP ≥ 0.329",
   "model": "FineLAP",
   "question": null
  },
  {
   "id": "dasm_vote",
   "stage": 4,
   "name": "DASM vote on rescues",
   "plain": "A rescued sound is kept only if DASM also hears it around it.",
   "bar": "DASM ≥ 0.575 (span ± 0.5 s)",
   "model": "DASM",
   "question": null
  },
  {
   "id": "rescue_once",
   "stage": 4,
   "name": "One rescue per type",
   "plain": "Only the earliest rescued span of a sound type is kept.",
   "bar": "earliest per family",
   "model": null,
   "question": null
  },
  {
   "id": "label_filter",
   "stage": 5,
   "name": "Drawable sound type",
   "plain": "Speech, music, textures, wind and generic names are not drawn.",
   "bar": "depictable list",
   "model": null,
   "question": null
  },
  {
   "id": "family_merge",
   "stage": 5,
   "name": "Group by sound type",
   "plain": "Spans of one sound type within 1 s join; weak spans of a type that also has strong ones are dropped.",
   "bar": "join gap 1.0 s; weak = conf < 0.35",
   "model": null,
   "question": null
  },
  {
   "id": "display_bar",
   "stage": 5,
   "name": "Display bar",
   "plain": "A sound below the confidence bar is planned but not shown.",
   "bar": "conf ≥ 0.35",
   "model": null,
   "question": null
  },
  {
   "id": "speech_rescue",
   "stage": 5,
   "name": "Talked-about rescue",
   "plain": "A weak sound people react to in speech is shown.",
   "bar": "both letter orders say 'reacting'",
   "model": "Qwen3.8-27B (text)",
   "question": "A sound of {label} was heard in a video. Around that moment, someone said: \"{speech}\". Are they (a) reacting to that sound or talking about it, or (b) not referring to that sound? Answer with the letter only."
  },
  {
   "id": "gate",
   "stage": 5,
   "name": "Visibility gate",
   "plain": "The vision model judges, per ≤ 5-s stretch, if the sound's source is visible. The sound is silenced only if every stretch is seen.",
   "bar": "majority of 3 questions per stretch",
   "model": "Qwen3.8-27B",
   "question": "Q-name: These frames are from the moment a sound of {label} was heard. Name the thing in these frames that is making that sound… If nothing … is visible, answer exactly: nothing.\nQ-ab: (a) you can SEE {label} happening on screen … (b) {label} is not visibly happening in these frames. Answer with the letter only. (both orders)\nQ-desc: Describe what is happening in these frames in one sentence."
  },
  {
   "id": "family_rule",
   "stage": 5,
   "name": "Family rule",
   "plain": "A sound is silenced when a more specific related sound at the same time was silenced as visible.",
   "bar": "visible specific silences general, never the reverse",
   "model": null,
   "question": null
  },
  {
   "id": "disambiguate",
   "stage": 5,
   "name": "Same event, two names",
   "plain": "Two labels on one event: the frames pick one.",
   "bar": "both orders agree",
   "model": "Qwen3.8-27B",
   "question": null
  },
  {
   "id": "dedup",
   "stage": 5,
   "name": "Duplicate pictures",
   "plain": "Sounds that would give the same picture are merged.",
   "bar": "SigLIP similarity ≥ 0.80",
   "model": "SigLIP",
   "question": null
  },
  {
   "id": "depict_event",
   "stage": 6,
   "name": "Event on screen",
   "plain": "Drops a picture whose event is visibly happening and whose visible maker could make that sound.",
   "bar": "event yes/no both ways, and look-alike yes twice",
   "model": "Qwen3.8-27B",
   "question": null
  },
  {
   "id": "display_join",
   "stage": 6,
   "name": "Repeat merge",
   "plain": "Two pictures of the same sound closer than the gap become one.",
   "bar": "gap ≤ 2.5 s",
   "model": null,
   "question": null
  },
  {
   "id": "group",
   "stage": 6,
   "name": "Smart grouping",
   "plain": "Two same-type pictures ≤ 8 s apart merge if the second is the same continuing sound.",
   "bar": "both orders answer 'same'",
   "model": "Qwen3-Omni",
   "question": "You hear a {lab} sound near the start and again near the end of this recording. Is the {lab} near the end the same continuing sound as at the start, or a new, separate event? Answer with exactly one word."
  },
  {
   "id": "max_slots",
   "stage": 6,
   "name": "At most 3 at once",
   "plain": "More than 3 pictures at the same moment: the weakest waits.",
   "bar": "≤ 3",
   "model": null,
   "question": null
  },
  {
   "id": "scorer",
   "stage": 7,
   "name": "Scoring",
   "plain": "A picture is a hit if it is the right sound type and starts 0.5 s before to 1.0 s after the needed sound.",
   "bar": "−0.5 … +1.0 s",
   "model": null,
   "question": null
  },
  {
   "id": "never_heard",
   "stage": 4,
   "name": "Never heard",
   "plain": "No detector produced any span of this sound type near its start.",
   "bar": "—",
   "model": null,
   "question": null
  }
 ],
 "clips": [
  {
   "clip": "bell_miami",
   "split": "DEV",
   "dur": 15.0,
   "video": "media/DEV/bell_miami.mp4",
   "gold": [
    {
     "id": "g1",
     "label": "Bell",
     "start": 0.2,
     "end": 14.5,
     "needed": true,
     "outcome": "miss",
     "lost_at": "gate",
     "why": "Gate: 3 of 3 questions said the source is on screen (a bell tower is visible; the ringing bell is a different one).",
     "cands": [
      "c1"
     ]
    }
   ],
   "pictures": [],
   "cands": [
    {
     "id": "c1",
     "label": "Church bell",
     "start": 0.22,
     "end": 3.2,
     "origin": "beats + flexsed",
     "fate": "dropped",
     "at": "gate",
     "trail": [
      {
       "step": "beats_extract",
       "res": "pass",
       "value": "0.70",
       "bar": "≥ 0.175"
      },
      {
       "step": "flexsed_extract",
       "res": "pass",
       "value": "0.98",
       "bar": "≥ 0.8"
      },
      {
       "step": "twin_union",
       "res": "pass",
       "note": "BEATs and FlexSED agree"
      },
      {
       "step": "dasm_local_veto",
       "res": "pass",
       "value": "DASM 0.92",
       "bar": "≥ 0.35"
      },
      {
       "step": "display_bar",
       "res": "pass",
       "value": "0.70",
       "bar": "≥ 0.35"
      },
      {
       "step": "gate",
       "res": "drop",
       "value": "3 of 3 seen",
       "bar": "silenced only if every stretch is seen",
       "note": "Q-name is real (the gate named 'church bell'); the other two answers are illustrative answer (sample).",
       "asks": [
        {
         "who": "Q-name · stretch 1",
         "q": "Name the thing in these frames that is making that sound…",
         "a": "church bell",
         "vote": "seen"
        },
        {
         "who": "Q-ab · both orders",
         "q": "(a) you can SEE Church bell happening on screen … (b) not visibly happening",
         "a": "(a), (a)",
         "vote": "seen"
        },
        {
         "who": "Q-desc",
         "q": "Describe what is happening in these frames in one sentence.",
         "a": "A white church bell tower stands against a blue sky.",
         "vote": "seen"
        }
       ]
      }
     ]
    }
   ]
  },
  {
   "clip": "tg_d029",
   "split": "DEV",
   "dur": 15.0,
   "video": "media/DEV/tg_d029.mp4",
   "gold": [
    {
     "id": "g1",
     "label": "Chicken, rooster",
     "start": 6.9,
     "end": 14.8,
     "needed": true,
     "outcome": "miss",
     "lost_at": "mirror_veto",
     "why": "Mirror veto: FlexSED heard a different sound strongly at that moment, and the rooster only at 0.26 (< 0.4).",
     "cands": [
      "c1"
     ]
    }
   ],
   "pictures": [],
   "cands": [
    {
     "id": "c1",
     "label": "Crowing, cock-a-doodle-doo",
     "start": 6.9,
     "end": 8.4,
     "origin": "beats",
     "fate": "dropped",
     "at": "mirror_veto",
     "trail": [
      {
       "step": "beats_extract",
       "res": "pass",
       "value": "0.81",
       "bar": "≥ 0.175"
      },
      {
       "step": "flexsed_extract",
       "res": "skip",
       "value": "0.26",
       "bar": "≥ 0.8",
       "note": "FlexSED alone did not hear it"
      },
      {
       "step": "mirror_veto",
       "res": "drop",
       "value": "own type 0.26; other type above 0.7",
       "bar": "drop if other ≥ 0.7 and own < 0.4",
       "note": "the other type's name and score will come from the log"
      },
      {
       "step": "mirror_keep",
       "res": "skip",
       "note": "the listener lists did not name it, so it stays dropped (from the dissection: Qwen list no, Audio Flamingo list yes)"
      }
     ]
    }
   ]
  },
  {
   "clip": "ambient_citywalk_nyc_1689",
   "split": "DEV",
   "dur": 18.0,
   "video": "media/DEV/ambient_citywalk_nyc_1689.mp4",
   "gold": [
    {
     "id": "g1",
     "label": "Vehicle (car horn)",
     "start": 3.8,
     "end": 4.3,
     "needed": true,
     "outcome": "miss",
     "lost_at": "band_rescue",
     "why": "FlexSED heard it at 0.71 (below 0.8). Both listeners were asked about that stretch and neither listed a vehicle.",
     "cands": [
      "c1"
     ]
    }
   ],
   "pictures": [],
   "cands": [
    {
     "id": "c1",
     "label": "Vehicle",
     "start": 0.0,
     "end": 10.0,
     "origin": "flexsed band run",
     "fate": "dropped",
     "at": "band_rescue",
     "trail": [
      {
       "step": "beats_extract",
       "res": "skip",
       "value": "0.10",
       "bar": "≥ 0.175",
       "note": "BEATs too weak"
      },
      {
       "step": "flexsed_extract",
       "res": "skip",
       "value": "0.71",
       "bar": "≥ 0.8",
       "note": "below FlexSED's own bar, so it goes to the listeners"
      },
      {
       "step": "band_rescue",
       "res": "drop",
       "value": "peak 0.56 → Qwen and Audio Flamingo must both name it",
       "bar": "0.5–0.8",
       "note": "Real outcome from the 1 Oct cache fill (TIER no: Qwen V4 no, AF V4 no); the list texts will come from the log.",
       "asks": [
        {
         "who": "Qwen3-Omni V4 on 0.0–10.0 s",
         "q": "List every distinct non-speech sound you hear in this recording, one per line, most prominent first.",
         "a": "(no vehicle in the list)",
         "vote": "no"
        },
        {
         "who": "Audio Flamingo V4 on 0.0–10.0 s",
         "q": "List every distinct non-speech sound you hear in this recording, one per line, most prominent first.",
         "a": "(no vehicle in the list)",
         "vote": "no"
        }
       ]
      }
     ]
    }
   ]
  },
  {
   "clip": "as_explosion_XJ8lc3I6",
   "split": "DEV",
   "dur": 12.0,
   "video": "media/DEV/as_explosion_XJ8lc3I6.mp4",
   "gold": [
    {
     "id": "g1",
     "label": "Gasp",
     "start": 6.7,
     "end": 7.0,
     "needed": true,
     "outcome": "miss",
     "lost_at": "beats_extract",
     "why": "BEATs heard it (0.42) but only for 0.25 s, shorter than the 0.3-s minimum.",
     "cands": []
    }
   ],
   "pictures": [],
   "cands": []
  },
  {
   "clip": "tg_d107",
   "split": "DEV",
   "dur": 12.0,
   "video": "media/DEV/tg_d107.mp4",
   "gold": [
    {
     "id": "g1",
     "label": "Bird call",
     "start": 6.0,
     "end": 8.9,
     "needed": false,
     "visible": true
    },
    {
     "id": "g2",
     "label": "Laughter",
     "start": 8.2,
     "end": 9.0,
     "needed": true,
     "outcome": "hit",
     "cands": []
    }
   ],
   "pictures": [
    {
     "id": "p1",
     "label": "Screaming",
     "start": 6.52,
     "end": 9.0,
     "verdict": "cross",
     "gold": "g1",
     "gold_text": "Bird call 6.0–8.9 (visible)",
     "cand": "c1"
    }
   ],
   "cands": [
    {
     "id": "c1",
     "label": "Screaming",
     "start": 6.52,
     "end": 9.0,
     "origin": "beats",
     "fate": "drawn",
     "trail": [
      {
       "step": "beats_extract",
       "res": "pass",
       "value": "0.4",
       "bar": "≥ 0.175",
       "note": "illustrative answer (sample)"
      },
      {
       "step": "dasm_local_veto",
       "res": "pass",
       "value": "DASM below 0.35; one listener names it",
       "bar": "≥ 0.35 or both name it",
       "note": "goes to the scene check"
      },
      {
       "step": "scene_margin",
       "res": "pass",
       "value": "credible",
       "bar": "margin > 0",
       "note": "Real answer from Round 60. The bird making the sound is on screen, and screaming is credible there, so the picture survives.",
       "asks": [
        {
         "who": "Qwen3.8-27B, stretch 7.2–9.0 s",
         "q": "Could the sound of screaming plausibly be heard in this scene? Answer yes or no.",
         "a": "yes",
         "vote": "yes"
        }
       ]
      },
      {
       "step": "display_bar",
       "res": "pass",
       "value": "≥ 0.35",
       "bar": "≥ 0.35"
      },
      {
       "step": "gate",
       "res": "pass",
       "value": "not seen",
       "bar": "silenced only if every stretch is seen",
       "note": "No screaming person is visible, so the gate keeps it (illustrative answer (sample))."
      },
      {
       "step": "scorer",
       "res": "cross",
       "value": "cross",
       "bar": "−0.5 … +1.0 s",
       "note": "No needed screaming here; the sound is a visible bird. Counted as a wrong picture."
      }
     ]
    }
   ]
  },
  {
   "clip": "ambient_weather_storm_7200",
   "split": "DEV",
   "dur": 16.0,
   "video": "media/DEV/ambient_weather_storm_7200.mp4",
   "gold": [
    {
     "id": "g1",
     "label": "Thunder",
     "start": 0.0,
     "end": 6.0,
     "needed": false,
     "visible": true
    }
   ],
   "pictures": [
    {
     "id": "p1",
     "label": "Thunder",
     "start": 0.06,
     "end": 3.0,
     "verdict": "visible",
     "gold": "g1",
     "cand": "c1"
    }
   ],
   "cands": [
    {
     "id": "c1",
     "label": "Thunder",
     "start": 0.06,
     "end": 3.0,
     "origin": "beats + flexsed",
     "fate": "drawn",
     "trail": [
      {
       "step": "beats_extract",
       "res": "pass",
       "value": "0.6",
       "bar": "≥ 0.175",
       "note": "illustrative answer (sample)"
      },
      {
       "step": "dasm_local_veto",
       "res": "pass",
       "value": "0.8",
       "bar": "≥ 0.35",
       "note": "illustrative answer (sample)"
      },
      {
       "step": "display_bar",
       "res": "pass",
       "value": "0.6",
       "bar": "≥ 0.35"
      },
      {
       "step": "gate",
       "res": "pass",
       "value": "1 of 3 seen",
       "bar": "silenced only if every stretch is seen",
       "note": "All three answers are illustrative answer (sample). The annotator saw lightning at that moment; the frames sampled may miss it.",
       "asks": [
        {
         "who": "Q-name",
         "q": "Name the thing in these frames that is making that sound…",
         "a": "nothing",
         "vote": "not seen"
        },
        {
         "who": "Q-ab · both orders",
         "q": "(a) you can SEE Thunder happening … (b) not visibly happening",
         "a": "(b), (a)",
         "vote": "split"
        },
        {
         "who": "Q-desc",
         "q": "Describe what is happening in these frames in one sentence.",
         "a": "Dark storm clouds over a flat field.",
         "vote": "not seen"
        }
       ]
      },
      {
       "step": "scorer",
       "res": "visible",
       "value": "visible",
       "note": "The gold marks this thunder as visible, so the picture counts as wrong."
      }
     ]
    }
   ]
  },
  {
   "clip": "tg_d032",
   "split": "DEV",
   "dur": 16.0,
   "video": "media/DEV/tg_d032.mp4",
   "gold": [
    {
     "id": "g1",
     "label": "Thunder",
     "start": 13.6,
     "end": 15.5,
     "needed": true,
     "outcome": "hit",
     "cands": [
      "c1"
     ]
    }
   ],
   "pictures": [
    {
     "id": "p1",
     "label": "Thunder",
     "start": 13.75,
     "end": 15.5,
     "verdict": "hit",
     "gold": "g1",
     "cand": "c1"
    }
   ],
   "cands": [
    {
     "id": "c1",
     "label": "Thunder",
     "start": 13.75,
     "end": 14.75,
     "origin": "beats",
     "fate": "drawn",
     "trail": [
      {
       "step": "beats_extract",
       "res": "pass"
      },
      {
       "step": "dasm_local_veto",
       "res": "pass",
       "value": "DASM 0.01; only Qwen names it",
       "bar": "≥ 0.35 or both name it",
       "note": "goes to the scene check"
      },
      {
       "step": "scene_margin",
       "res": "pass",
       "value": "credible",
       "bar": "margin > 0",
       "note": "Real answer from Round 60: this is the hit the scene check won back.",
       "asks": [
        {
         "who": "Qwen3.8-27B, 13.75–14.75 s",
         "q": "Could the sound of thunder plausibly be heard in this scene? Answer yes or no.",
         "a": "yes",
         "vote": "yes"
        }
       ]
      },
      {
       "step": "gate",
       "res": "pass",
       "value": "not seen"
      },
      {
       "step": "scorer",
       "res": "hit",
       "value": "hit",
       "bar": "−0.5 … +1.0 s"
      }
     ]
    }
   ]
  }
 ],
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
