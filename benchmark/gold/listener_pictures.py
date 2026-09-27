"""Report-only listener job for the pictures (docs/freeze_picture_setup_2026-09-25.md, "Report-only listener job"): what an
audio LLM hears on the 54 fresh-picture sounds, and whether a closed question over the fired labels would change the drawn
noun. Nothing here enters the frozen setup; the sealed confirmation set is not touched.

    python benchmark/gold/listener_pictures.py            # GPU (Qwen3-Omni-30B-A3B-Instruct), writes listener_pictures.json
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from src.labels import ancestors, is_descendant, label_names, _parents

MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
BENCH = _ROOT / "data" / "work" / "picture_bench_fresh"
VIDEOS = _ROOT / "data" / "input" / "pic_fresh"
OUT = _ROOT / "benchmark" / "gold" / "pictures" / "listener_pictures.json"
Q_A = "What is making this sound? Answer with a short noun phrase."
Q_B = "Which of these best names the sound in this recording? {opts} Answer with one letter only."


def all_labels():
    par = _parents()
    return sorted(set(par) | set(par.values()))


def mentions(text, label):
    t = " " + re.sub(r"[^a-z ]", " ", text.lower()) + " "
    return any(re.search(r"\b" + re.escape(n) + r"s?\b", t) for n in label_names(label) if len(n) > 2)


def classify_a(ans, source, fired, labels):
    anc = set(ancestors(source))
    if mentions(ans, source) or any(mentions(ans, l) for l in labels if is_descendant(l, source)):
        return "source"
    if any(mentions(ans, a) for a in anc):
        return "family"
    if any(mentions(ans, f) for f in fired if f != source and f not in anc):
        return "fired sibling"
    if any(mentions(ans, l) for l in labels):
        return "other ontology"
    return "off-ontology"


def main():
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    specs = json.loads((BENCH / "specs.json").read_text(encoding="utf-8"))
    subj = json.loads((BENCH / "subjects_V31G.json").read_text(encoding="utf-8"))
    labels = all_labels()
    items = []
    for s in specs:
        sj = subj[str(s["i"])]
        src, fired = sj["source"], sj.get("fired", [])
        alts = [f for f in fired if f != src and f not in ancestors(src)]
        items.append({"i": s["i"], "clip": s["clip"], "start": s["start"], "end": s["end"], "source": src, "fired": fired,
                      "alternatives": alts})
    pop = sum(bool(it["alternatives"]) for it in items)
    print(f"[pictures] 54 sounds; prompt-B population (a real fired alternative): {pop}", flush=True)

    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass

    def run(w, q, n):
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=n, do_sample=False)
        return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

    for it in items:
        raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-i", str(VIDEOS / f"{it['clip']}.mp4"), "-vn", "-ac", "1",
                              "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
        w = np.frombuffer(raw, np.float32)
        a = int(max(0.0, it["start"] - 1.0) * 16000)
        b = int(min(len(w) / 16000, min(it["end"], it["start"] + 7.0) + 1.0) * 16000)
        seg = w[a:max(b, a + 16000)].copy()
        it["answer_a"] = run(seg, Q_A, 16)
        it["class_a"] = classify_a(it["answer_a"], it["source"], it["fired"], labels)
        if it["alternatives"]:
            opts = [it["source"]] + [f for f in it["fired"] if f != it["source"]] + ["unsure"]
            letters = "ABCDEFGHIJ"[:len(opts)]
            q = Q_B.format(opts=" ".join(f"({L}) {o}" for L, o in zip(letters, opts)))
            ans = run(seg, q, 4)
            m = re.search(r"[A-J]", ans.upper())
            pick = opts[letters.index(m.group(0))] if m and m.group(0) in letters else "unparsed"
            it["answer_b"], it["pick_b"] = ans, pick
            it["noun_change_b"] = pick not in (it["source"], "unsure", "unparsed") and pick not in ancestors(it["source"])
        print(f"   {it['i']:2d} {it['source'][:22]:22s} A: {it['answer_a'][:30]:30s} [{it['class_a']}]"
              + (f"  B: {it.get('pick_b')}" if it["alternatives"] else ""), flush=True)

    counts_a = {k: sum(it["class_a"] == k for it in items) for k in ("source", "family", "fired sibling", "other ontology", "off-ontology")}
    changes = sum(bool(it.get("noun_change_b")) for it in items)
    res = {"population_b": pop, "counts_a": counts_a, "noun_changes_b": changes, "changes_to_unfired": 0,
           "worth_micro_sitting": pop >= 5 and changes >= 5, "items": items}
    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[pictures] A: {counts_a}; B: {changes} noun changes of {pop} eligible -> worth a micro-sitting: {res['worth_micro_sitting']}")
    print("->", OUT)


if __name__ == "__main__":
    main()
