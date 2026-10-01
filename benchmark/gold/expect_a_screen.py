"""Round 40b EXPECT-A (docs/prereg_round13_detector_push.md "Round 40b EXPECT-A"): no VLM proposal. Candidates = depictable
families named by the whole-clip Qwen3-Omni answer to a list question with a cleaner decode (LIST_Q, 96 new tokens,
repetition_penalty 1.2, greedy), matched by the shipped V4 matcher (listener_afnext.v4_match) PLUS the frozen word map below
(Birdsong -> Bird, Hammering -> Hammer, ...), not already drawn by SHIP8; onset = earliest weak-bar run (as Round 40); shipped
gate at the onset; 2-s picture; scored as Round 40 (expect_screen.cmd_gate / cmd_score re-used on this round's folder).

    TG_ARMS=SHIP8 python benchmark/gold/expect_a_screen.py listen   # GPU: Omni -> expect_a/listen/
    TG_ARMS=SHIP8 python benchmark/gold/expect_a_screen.py cands    # CPU -> expect_a/cands.json
    TG_ARMS=SHIP8 python benchmark/gold/expect_a_screen.py gate     # GPU: Qwen3.8-27B -> expect_a/gate/
    TG_ARMS=SHIP8 python benchmark/gold/expect_a_screen.py score    # CPU -> expect_a_screen.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
os.environ.setdefault("TG_ARMS", "SHIP8")
from benchmark.gold import expect_screen as E

E.DIR = _ROOT / "benchmark" / "gold" / "expect_a"
E.OUT = _ROOT / "benchmark" / "gold" / "expect_a_screen.json"
DIR = E.DIR
LIST_Q = ("List every distinct non-speech sound you hear in this recording as a comma-separated list of short sound names, "
          "most prominent first. Name each sound once. Answer with the list only.")
MAX_NEW, REP_PEN = 96, 1.2
# Frozen word map (written before any Round 40b number): a whole normalised item, or one of its words, -> depictable family.
MAP = {
    "birdsong": "Bird", "bird song": "Bird", "birds": "Bird", "bird": "Bird", "chirping": "Bird", "chirp": "Bird",
    "tweeting": "Bird", "crowing": "Bird", "rooster": "Bird", "chicken": "Bird", "hen": "Bird", "seagull": "Gull, seagull",
    "gull": "Gull, seagull", "crow": "Crow", "owl": "Owl", "duck": "Duck", "goose": "Goose", "honking": "Honk", "honk": "Honk",
    "hammering": "Hammer", "hammer": "Hammer", "drilling": "Drill", "drill": "Drill", "sawing": "Sawing", "saw": "Sawing",
    "barking": "Dog", "bark": "Dog", "dog": "Dog", "howling": "Dog", "meowing": "Cat", "meow": "Cat", "cat": "Cat",
    "purring": "Cat", "mooing": "Cattle", "cow": "Cattle", "cows": "Cattle", "cattle": "Cattle", "horse": "Horse",
    "neighing": "Horse", "hooves": "Horse", "sheep": "Sheep", "bleating": "Sheep", "pig": "Pig", "oinking": "Pig",
    "insects": "Insect", "insect": "Insect", "buzzing": "Buzz", "crickets": "Cricket", "cricket": "Cricket", "frogs": "Frog",
    "frog": "Frog", "croaking": "Frog",
    "laughing": "Laughter", "laughter": "Laughter", "laugh": "Laughter", "giggling": "Giggle", "clapping": "Clapping",
    "applause": "Clapping", "cheering": "Crowd", "crowd": "Crowd", "crowd noise": "Crowd", "chatter": "Crowd",
    "footsteps": "Footsteps", "footstep": "Footsteps", "walking": "Footsteps", "steps": "Footsteps", "running": "Footsteps",
    "breathing": None, "coughing": "Cough", "cough": "Cough", "sneezing": "Sneeze", "screaming": "Screaming",
    "scream": "Screaming", "shouting": "Shout", "shout": "Shout", "yelling": "Yell", "crying": "Crying, sobbing",
    "baby crying": "Baby cry, infant cry", "snoring": "Snoring", "whistling": "Whistle", "whistle": "Whistle",
    "thunder": "Thunder", "rain": "Rain", "raining": "Rain", "rainfall": "Rain", "wind": None, "water": "Water",
    "stream": "Water", "river": "Water", "waves": "Water", "ocean": "Water", "sea": "Water", "waterfall": "Water",
    "water running": "Water", "running water": "Water", "dripping": "Water", "splashing": "Splash, splatter",
    "splash": "Splash, splatter", "fire": "Fire", "crackling": "Crackle", "fireworks": "Fireworks", "explosion": "Explosion",
    "explosions": "Explosion", "blast": "Explosion", "gunshot": "Gunshot", "gunshots": "Gunshot", "gunfire": "Gunshot",
    "shooting": "Gunshot",
    "train": "Train", "trains": "Train", "subway": "Train", "railway": "Train", "tracks": "Train", "whistle train": "Train",
    "car": "Vehicle", "cars": "Vehicle", "traffic": "Vehicle", "vehicle": "Vehicle", "vehicles": "Vehicle", "engine": "Vehicle",
    "truck": "Vehicle", "bus": "Vehicle", "motorcycle": "Vehicle", "revving": "Vehicle", "car horn": "Honk",
    "horn": "Honk", "honking horn": "Honk", "siren": "Siren", "sirens": "Siren", "ambulance": "Siren", "police siren": "Siren",
    "aircraft": "Aircraft", "airplane": "Aircraft", "plane": "Aircraft", "jet": "Aircraft", "helicopter": "Helicopter",
    "boat": "Boat", "ship": "Boat", "bicycle": "Bicycle", "bike bell": "Bell",
    "bell": "Bell", "bells": "Bell", "church bell": "Bell", "church bells": "Bell", "ringing": "Bell", "chime": "Bell",
    "doorbell": "Doorbell", "alarm": "Alarm", "clock": "Clock", "ticking": "Tick", "telephone": "Telephone",
    "phone": "Telephone", "phone ringing": "Telephone", "beeping": None, "beep": None,
    "door": "Door", "door closing": "Door", "door slam": "Door", "knocking": "Knock", "knock": "Knock",
    "glass": "Glass", "glass breaking": "Glass", "dishes": "Dishes", "plates": "Dishes", "cutlery": "Chink, clink",
    "clinking": "Chink, clink", "pots": "Dishes", "pans": "Dishes", "cooking": "Cooking", "frying": "Cooking", "sizzling": "Cooking",
    "boiling": "Boiling", "kettle": "Kettle whistle", "toilet flush": "Toilet flush", "flushing": "Toilet flush",
    "typing": "Typing", "keyboard": "Computer keyboard", "camera": "Camera", "camera shutter": "Camera",
    "chainsaw": "Chainsaw", "lawn mower": "Lawn mower", "vacuum": "Vacuum cleaner", "blender": "Blender",
    "hair dryer": "Hair dryer", "fan": "Mechanical fan", "machinery": None, "music": None, "speech": None, "talking": None,
    "voices": None, "singing": None, "piano": None, "drum": None, "drums": None, "guitar": None,
}
_LOW = {f.lower(): f for f in E.FAMILIES}
_LABELS = {l.lower(): l for l in E.VOCAB["labels"]}
assert all(v is None or v in _LOW.values() for v in MAP.values()), [v for v in MAP.values() if v and v not in _LOW.values()]


def items_of(text: str):
    parts = re.split(r"[,\n;]+", text)
    out = []
    for p in parts:
        p = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", p).strip().strip(".").strip().lower()
        p = re.sub(r"^(a|an|the|some)\s+", "", p)
        if p and p not in out:
            out.append(p)
    return out


def map_item(item: str):
    """-> family or None by the frozen map: whole item, then (item without a trailing 'sound(s)'/'noise'), then its words (each
    also singularised: -es / -s), then a depictable label / family name as a whole item (canonical family)."""
    from src.labels import canonical
    cands = [item, re.sub(r"\s+(sounds?|noises?)$", "", item)]
    for c in cands:
        if c in MAP:
            return MAP[c]
        if c in _LOW:
            return _LOW[c]
        if c in _LABELS:
            f = canonical(_LABELS[c]); return f if f in _LOW.values() else None
    for w in item.split():
        for ww in (w, w[:-2] if w.endswith("es") else w, w[:-1] if w.endswith("s") else w):
            if ww in MAP:
                return MAP[ww]
    return None


def cmd_listen():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    from benchmark.gold import listener_variants as LV
    d = DIR / "listen"; d.mkdir(parents=True, exist_ok=True)
    todo = [(pt, st) for pt, st in E.stems_only() if not (d / f"{st}.json").exists()]
    LV.S.load_gold = LV._no_gold
    print(f"{len(todo)} clips to listen", flush=True)
    if not todo:
        return
    proc = Qwen3OmniMoeProcessor.from_pretrained(LV.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(LV.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    for pt, st in todo:
        w, sr = sf.read(str(E.WAV[pt] / f"{st}.wav"), dtype="float32")
        assert sr == LV.SR, (st, sr)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": LIST_Q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=MAX_NEW, do_sample=False, repetition_penalty=REP_PEN)
        txt = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        rec = {"part": pt, "clip": st, "seconds": round(len(w) / sr, 2), "prompt": LIST_Q, "max_new": MAX_NEW,
               "repetition_penalty": REP_PEN, "text": txt, "items": items_of(txt)}
        (d / f"{st}.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"{pt:4s} {st} -> {txt!r}", flush=True)


def cmd_cands():
    from sentence_transformers import SentenceTransformer
    from benchmark.gold import listener_variants as LV
    from benchmark.gold import listener_afnext as AF
    from benchmark.gold import gbtp_screen as G
    from benchmark.gold import detached_add_screen as D
    from benchmark.gold.score_per_sound import same_family
    from src.labels import canonical
    assert D.BARS == {"flex": 0.3, "beats": 0.175, "dasm": 0.575}, D.BARS
    O = LV.Onto()
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    rows = []; missing = {m: 0 for m in D.BARS}
    tally = {"items": 0, "named": 0, "by_map": 0, "by_matcher": 0, "already_drawn": 0, "no_run": 0, "candidate": 0}
    for pt, st, g, pics in E.parts():
        lf = DIR / "listen" / f"{st}.json"
        if not lf.exists():
            print("missing listen output", pt, st); continue
        L = json.loads(lf.read_text(encoding="utf-8"))
        items = items_of(L["text"]); tally["items"] += len(items)
        fams, how = [], {}
        for it in items:
            f = map_item(it)
            if f:
                how.setdefault(f, f"map:{it}")
                if f not in fams:
                    fams.append(f)
        text_lines = "\n".join(items)
        for F in E.FAMILIES:
            if F in fams:
                continue
            m, ok = AF.v4_match(O, emb, text_lines, F)
            if ok:
                fams.append(F); how[F] = "matcher:" + "|".join(m["matched_word"] + m["matched_cos"])
        tally["named"] += len(fams); tally["by_map"] += sum(v.startswith("map") for v in how.values())
        tally["by_matcher"] += sum(v.startswith("matcher") for v in how.values())
        cs = G.caches(pt, st)
        for m in D.BARS:
            missing[m] += cs[m] is None
        drawn = [l for l, a, b, r in pics]
        for F in fams:
            row = {"part": pt, "clip": st, "family": F, "how": how[F], "omni_match": True}
            already = [l for l in drawn if canonical(l) == F or same_family(l, F)]
            if already:
                tally["already_drawn"] += 1; row["outcome"] = "already drawn"; row["drawn_as"] = already
            else:
                evs = {mm: D.evidence(fr, F) for mm, fr in cs.items()}
                found = {}
                for mm in D.BARS:
                    rr = D.runs_of(evs[mm], D.BARS[mm])
                    if rr:
                        found[mm] = min(s for s, e in rr)
                if not found:
                    tally["no_run"] += 1; row["outcome"] = "no run at weak bars"
                    row["no_evidence"] = [mm for mm in D.BARS if evs[mm] is None]
                else:
                    tally["candidate"] += 1
                    src = min(found, key=found.get)
                    row.update({"outcome": "candidate", "onset": round(found[src], 2), "from": src,
                                "onsets": {k: round(v, 2) for k, v in found.items()}})
            rows.append(row)
    DIR.mkdir(parents=True, exist_ok=True)
    (DIR / "cands.json").write_text(json.dumps({"tally": tally, "caches_missing": missing, "rows": rows}, indent=1), encoding="utf-8")
    print(tally, "caches missing", missing)
    for r in rows:
        if r["outcome"] == "candidate":
            print(f"   {r['part']:4s} {r['clip']} {r['family']} ({r['how']}) onset {r['onset']} ({r['from']}, {r['onsets']})")


if __name__ == "__main__":
    {"listen": cmd_listen, "cands": cmd_cands, "gate": E.cmd_gate, "score": E.cmd_score}[sys.argv[1]]()
