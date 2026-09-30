"""Round 29 GA (docs/prereg_round13_detector_push.md): audio-identity veto of the gate's named visible thing, DEV judge clips.
For each cached Qwen3.8-27B gate stretch (benchmark/gold/gate_gold/Qwen38-27B) with a "seen" majority and a named object,
Qwen3-Omni hears the stretch audio (+-1 s, no frames) and answers an a/b in both orderings; a firm "not this thing" in both
vetoes the stretch's "seen". Writes gate_gold/aveto_Qwen38-27B/<stem>.json and scores seen_silenced / needed_kept.

    python benchmark/gold/gate_audio_veto.py run      # GPU
    python benchmark/gold/gate_audio_veto.py score
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import gate_gold as G
from benchmark.gold import detector_dry as DD

SRC = G.OUT_DIR / "Qwen38-27B"
OUT = G.OUT_DIR / "aveto_Qwen38-27B"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SR = 16000
Q = ("Listen to this recording. Which is true? (a) this is the sound of {a} (b) {b}. Answer with the letter only.")


def run():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    judge = set(G.JUDGE100.read_text().split())
    OUT.mkdir(parents=True, exist_ok=True)
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    ida = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("a", "A", " a", " A", "(a", "(A")})
    idb = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("b", "B", " b", " B", "(b", "(B")})

    def pick(a, opt1, opt2):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": a}, {"type": "text", "text": Q.format(a=opt1, b=opt2)}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[a], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            lg = model.thinker(**inp).logits[0, -1].float()
        return "a" if float(lg[ida].max()) >= float(lg[idb].max()) else "b"
    for f in sorted(SRC.glob("*.json")):
        if f.stem not in judge or (OUT / f.name).exists():
            continue
        d = json.loads(f.read_text(encoding="utf-8"))
        w, sr = sf.read(str(DD.wav_for(DD.clip_path(d["clip"]))), dtype="float32")
        assert sr == SR, sr
        for s in d["sounds"]:
            fam = s["label"].lower()
            for st in s["stretches"]:
                named = (st.get("named") or "").strip()
                st["aveto"] = None
                if not st.get("seen_majority") or not named or named.lower() == "nothing":
                    continue
                a = w[int(max(0.0, st["start"] - 1.0) * SR):int((st["end"] + 1.0) * SR)]
                if len(a) < SR:
                    continue
                this = f"a {named}" if not named.lower().startswith(("a ", "an ", "the ")) else named
                other = f"this is a different {fam} sound, or something else"
                r1 = pick(a, this, other)                               # the named thing as (a)
                conv2 = pick(a, other.replace("this is ", ""), f"this is the sound of {this}")   # swapped order
                st["aveto_answers"] = [r1, conv2]
                st["aveto"] = (r1 == "b") and (conv2 == "a")            # firm "not the named thing" in both orderings
        (OUT / f.name).write_text(json.dumps(d, indent=1), encoding="utf-8")
        print(f.stem, sum(st.get("aveto") is True for s in d["sounds"] for st in s["stretches"]), "vetoes", flush=True)


def seen_all(sts, veto):
    """the shipped majority rule (gate_gold.decide); a vetoed stretch counts as not seen"""
    return G.decide(sts, "majority") and not (veto and any(st.get("aveto") is True for st in sts))


def score():
    res = {}
    for rule, veto in (("majority", False), ("GA", True)):
        c = {"seen": 0, "seen_sil": 0, "needed": 0, "needed_kept": 0, "flips": []}
        for f in sorted(OUT.glob("*.json")):
            for s in json.loads(f.read_text(encoding="utf-8"))["sounds"]:
                if s["importance"] < 2:
                    continue
                pred = seen_all(s["stretches"], veto)
                base = seen_all(s["stretches"], False)
                if s["seen"]:
                    c["seen"] += 1; c["seen_sil"] += pred
                else:
                    c["needed"] += 1; c["needed_kept"] += not pred
                if pred != base:
                    c["flips"].append([f.stem, s["label"], s["start"], "seen" if s["seen"] else "NEEDED", "silenced" if pred else "kept"])
        res[rule] = c
        print(f"[{rule}] seen silenced {c['seen_sil']}/{c['seen']} | needed kept {c['needed_kept']}/{c['needed']}")
        for x in c["flips"]:
            print("    ", x)
    m, b = res["GA"], res["majority"]
    res["go"] = (m["needed_kept"] - b["needed_kept"] >= 3) and (b["seen_sil"] - m["seen_sil"] <= 1)
    print("GA screen:", "GO" if res["go"] else "STOP")
    (G.OUT_DIR / "aveto_summary.json").write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    run() if sys.argv[1] == "run" else score()
