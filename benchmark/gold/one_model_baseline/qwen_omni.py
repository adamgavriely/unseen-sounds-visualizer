"""Outside one-model baseline 2: Qwen3-Omni-30B-A3B-Instruct (Apache-2.0) hears the whole clip and lists every
non-speech, non-music sound with its start second. One picture per listed (sound, start). No threshold, nothing tuned.

Fixed before any reply is read:
  * audio: the whole clip, 16 kHz mono (data/work/gold_wav_flat/<stem>.wav);
  * vocabulary: the 447 AudioSet-Strong class names (as M2D uses them, ontology spelling), speech and music removed,
    given in the prompt so every answer has a scoreable name;
  * thinker only (talker disabled); max 512 new tokens; sampling as the Qwen3 model card advises (temperature 0.6,
    top_p 0.95, top_k 20, repetition penalty 1.05), torch seed 0 per clip. (First try, greedy with no cap on the
    number of lines, looped on 4 of 4 clips -- the vocabulary recited, "Hammer" x20, junk tokens -- and was stopped
    before any score was read; the cap of 20 lines and the sampling were added then. The second try showed the real
    cause: generate() did not stop at the end-of-turn token <|im_end|> and ran on into new "assistant" turns
    ("istant", "none" x50). Fixed by passing <|im_end|> and <|endoftext|> as eos_token_id; the reply is also cut at
    the first end-of-turn marker.)
  * parse: lines "<seconds> | <name>"; the name must match a vocabulary entry (case-insensitive, then the longest
    entry contained in the reply's name); unmatched lines are dropped and counted; a start outside the clip is clipped.

    python benchmark/gold/one_model_baseline/qwen_omni.py          # GPU (msproj), stems from stems.json -> qwen_replies.json
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
_ROOT = HERE.parent.parent.parent
sys.path.insert(0, str(_ROOT))
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
WAV = Path(os.environ.get("MSCPROJ_ROOT", Path(__file__).resolve().parents[3])) / "data" / "work" / "gold_wav_flat"
OUT = HERE / "qwen_replies.json"
MAX_NEW = 512
GEN = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, repetition_penalty=1.05)


def vocabulary():
    from src.labels import SPEECH_LABELS, is_music, is_descendant
    z = np.load(_ROOT / "data" / "work" / "psed_ens_cache" / "M2D" / "ambient_citywalk_cairo_834.npz")
    names = [str(x) for x in z["labels"]]
    keep = [n for n in names if not (n in SPEECH_LABELS or is_descendant(n, "Speech") or is_music(n))]
    return sorted(set(keep))


def prompt(vocab, dur):
    return ("You are listening to a video's soundtrack of %.1f seconds. List every sound you hear that is NOT speech and "
            "NOT music. For each sound give the second at which it starts. If the same sound starts again later "
            "(after a pause), list it again with the new start; do not list a steady sound more than once. At most 20 lines. "
            "Use ONLY names from this list, written exactly:\n%s\n\n"
            "Answer with one line per sound, in the form: <start in seconds> | <name>\n"
            "If you hear no such sound, answer: none" % (dur, "; ".join(vocab)))


def parse(txt, vocab, dur):
    low = {v.lower(): v for v in vocab}
    pics, dropped = [], 0
    for line in txt.splitlines():
        m = re.match(r"\s*[-*]?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*[|:\-]\s*(.+?)\s*$", line)
        if not m:
            continue
        t, name = float(m.group(1)), m.group(2).strip().strip(".").lower()
        lab = low.get(name)
        if lab is None:
            inside = [v for k, v in low.items() if k in name]
            lab = max(inside, key=len) if inside else None
        if lab is None:
            dropped += 1; continue
        t = min(max(t, 0.0), dur)
        pics.append((lab, t, min(t + 1.0, dur)))
    return pics, dropped


def main():
    import soundfile as sf
    import torch
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    stems = json.loads((HERE / "stems.json").read_text(encoding="utf-8"))
    stems = stems["dev"] + stems["test"]
    out_path = OUT
    if len(sys.argv) > 1:                                  # trial: first N clips into qwen_trial.json
        stems, out_path = stems[: int(sys.argv[1])], HERE / "qwen_trial.json"
    vocab = vocabulary()
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    eos = [i for i in (tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")) if i is not None]
    print("eos ids", eos, flush=True)
    print(f"loaded in {time.time() - t0:.0f} s; vocabulary {len(vocab)}; {len(stems)} clips, {len(done)} done", flush=True)
    for i, st in enumerate(stems):
        if st in done:
            continue
        w, sr = sf.read(str(WAV / f"{st}.wav"), dtype="float32")
        assert sr == 16000, (st, sr)
        if w.ndim > 1:
            w = w.mean(axis=1)
        dur = len(w) / sr
        q = prompt(vocab, dur)
        conv = [{"role": "user", "content": [{"type": "audio", "audio": w}, {"type": "text", "text": q}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inp = proc(text=text, audio=[w], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        torch.manual_seed(0)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=MAX_NEW, eos_token_id=eos, pad_token_id=eos[-1], **GEN)
        raw = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        reply = raw.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        done[st] = {"duration": dur, "reply": reply, "new_tokens": int(out.shape[1] - inp["input_ids"].shape[1])}
        out_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(i, st, repr(reply[:120]), flush=True)
    print("DONE", len(done))


if __name__ == "__main__":
    main()
