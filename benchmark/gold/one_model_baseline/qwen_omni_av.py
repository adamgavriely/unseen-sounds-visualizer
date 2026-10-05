"""Outside one-model baseline 3: Qwen3-Omni-30B-A3B-Instruct (Apache-2.0) WATCHES the clip (video + its audio, in one
pass, use_audio_in_video=True) and lists every non-speech, non-music sound whose source it cannot see, with its start
and end second (Adam, 5 Oct: "let him do the full pipeline besides the drawing"). One picture per listed (sound, start, end): the one model hears, checks the screen and picks. Nothing tuned.

Same vocabulary, parse, sampling, end-of-turn stop and 20-line cap as qwen_omni.py (fixed there before any score was
read). Video read by the official qwen-omni-utils helper (process_mm_info, default 2 frames/s and pixel budget), env
~/venvs/qomni_av (msproj + qwen-omni-utils).

    python benchmark/gold/one_model_baseline/qwen_omni_av.py [N]     # GPU; stems.json -> qwen_av_replies.json (N: trial)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import qwen_omni as Q

_ROOT = HERE.parent.parent.parent
OUT = HERE / "qwen_av_replies.json"


def prompt(vocab, dur):
    return ("You are watching a video with its sound, %.1f seconds long. List every sound you hear that is NOT speech, "
            "NOT music, and whose source you CANNOT see in the picture when the sound happens (it is off screen, hidden "
            "or too small to see). Do not list sounds whose source is visible. For each sound give the second at which "
            "it starts and the second at which it ends. If the same sound starts again later (after a pause), list it again with the new start; do not "
            "list a steady sound more than once. At most 20 lines. Use ONLY names from this list, written exactly:\n%s\n\n"
            "Answer with one line per sound, in the form: <start in seconds> | <end in seconds> | <name>\n"
            "If there is no such sound, answer: none" % (dur, "; ".join(vocab)))


def parse(txt, vocab, dur):
    """lines "<start> | <end> | <name>" (a line without an end is kept with end = start + 1 s); name matched as in
    qwen_omni.parse; unmatched lines dropped and counted"""
    import re
    low = {v.lower(): v for v in vocab}
    pics, dropped = [], 0
    for line in txt.splitlines():
        m = re.match(r"\s*[-*]?\s*(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*\|\s*(?:(\d+(?:\.\d+)?)\s*(?:s|sec|seconds)?\s*\|\s*)?(.+?)\s*$", line)
        if not m:
            continue
        a = min(max(float(m.group(1)), 0.0), dur)
        b = min(max(float(m.group(2)), a), dur) if m.group(2) else min(a + 1.0, dur)
        name = m.group(3).strip().strip(".").lower()
        lab = low.get(name)
        if lab is None:
            inside = [v for k, v in low.items() if k in name]
            lab = max(inside, key=len) if inside else None
        if lab is None:
            dropped += 1; continue
        pics.append((lab, a, b))
    return pics, dropped


def video_of(stem):
    """videos.json: every stem -> its video file (paths relative to the repository root, under data/input)"""
    if not hasattr(video_of, "m"):
        video_of.m = json.loads((HERE / "videos.json").read_text(encoding="utf-8"))
    return HERE.parents[2] / video_of.m[stem]


def main():
    import soundfile as sf
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    stems = json.loads((HERE / "stems.json").read_text(encoding="utf-8"))
    stems = stems["dev"] + stems["test"]
    out_path = OUT
    if len(sys.argv) > 1:
        stems, out_path = stems[: int(sys.argv[1])], HERE / "qwen_av_trial.json"
    vocab = Q.vocabulary()
    done = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {}
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(Q.MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(Q.MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    print(f"loaded in {time.time() - t0:.0f} s; eos {eos}; {len(stems)} clips, {len(done)} done", flush=True)
    for i, st in enumerate(stems):
        if st in done:
            continue
        dur = float(sf.info(str(Q.WAV / f"{st}.wav")).duration)
        vid = str(video_of(st))
        conv = [{"role": "user", "content": [{"type": "video", "video": vid}, {"type": "text", "text": prompt(vocab, dur)}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
        inp = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True,
                   use_audio_in_video=True)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        torch.manual_seed(0)
        with torch.inference_mode():
            out = model.thinker.generate(**inp, max_new_tokens=Q.MAX_NEW, eos_token_id=eos, pad_token_id=eos[-1],
                                         use_audio_in_video=True, **Q.GEN)
        raw = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=False)[0]
        reply = raw.split("<|im_end|>")[0].split("<|endoftext|>")[0].strip()
        done[st] = {"duration": dur, "reply": reply, "video": vid, "new_tokens": int(out.shape[1] - inp["input_ids"].shape[1])}
        out_path.write_text(json.dumps(done, indent=1), encoding="utf-8")
        print(i, st, repr(reply[:120]), flush=True)
    print("DONE", len(done))


if __name__ == "__main__":
    main()
