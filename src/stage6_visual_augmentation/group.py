"""Smart grouping of repeated pictures (Round 47 GROUP, Adam's idea, shipped 1 Oct 2026).

Two pictures of the same sound family that are close in time (picture gap in (0, GROUP_MAX_GAP] s) may be one sound
with a pause (an alarm, a shaver) or two events (two barks). Qwen3-Omni hears the stretch between them and answers "same"
or "new", in both option orders; only same + same merges them into one picture (first start, second end). Limitation: no chaining (after a+b merge, the pair b->c is not
re-asked as a->c); no DEV/TEST clip had a chain. Merging only
removes pictures, never adds one. The answers are computed once per clip (`ask`) and cached; the display step
(`_display_spans`) reads the cache, so the scorer and the renderer see the same pictures.

GROUP_MAX_GAP = 4.0 s (Adam: merge only when close enough; twice the gold's 2-s "sound starts again" rule).
"""
from __future__ import annotations

import json
from pathlib import Path

import config

PRE, POST = 1.0, 1.5
A_Q = ("You hear a {lab} sound near the start and again near the end of this recording. Is the {lab} near the end {o1}, or "
       "{o2}? Answer with exactly one word: {w1} or {w2}.")
A_OPT = {"same": "the same continuing sound as at the start (for example one alarm or engine with a short pause)",
         "new": "a new, separate event (for example a second bark, knock or shot)"}
ORDERS = (("same", "new"), ("new", "same"))
_CACHE = {}


def _family(a: str, b: str) -> bool:
    from src.labels import canonical, is_descendant
    return a == b or canonical(a) == canonical(b) or is_descendant(a, b) or is_descendant(b, a)


def key(label: str, a_start: float) -> str:
    return f"{label}|{a_start:.2f}"


def pairs(spans, max_gap: float):
    """consecutive same-family pictures (label, start, end, ...) with 0 < next start - this end <= max_gap"""
    ps = sorted(spans, key=lambda p: p[1])
    out = []
    for i, p in enumerate(ps):
        nxt = [q for q in ps[i + 1:] if _family(q[0], p[0])]
        if nxt and 0 < nxt[0][1] - p[2] <= max_gap:
            out.append((p, nxt[0]))
    return out


def _answers(clip: str):
    path = getattr(config, "GROUP_CACHE", None)
    if not path or clip is None:
        return {}
    if path not in _CACHE:
        p = Path(path)
        _CACHE[path] = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    return _CACHE[path].get(clip, {})


def apply(spans, clip: str):
    """merge the pairs Omni heard as one sound (both orders 'same'); spans are mutable [label, start, end, spec] lists"""
    if not getattr(config, "GROUP_ASK", False):
        return spans
    ans = _answers(clip or getattr(config, "GROUP_CLIP", None))
    out = sorted(spans, key=lambda p: p[1])
    for a, b in pairs(out, float(getattr(config, "GROUP_MAX_GAP", 4.0))):
        rep = ans.get(key(a[0], a[1]), [])
        if rep and all(str(x).lower().startswith("same") for x in rep) and b in out and a in out:
            a[2] = max(a[2], b[2])
            out.remove(b)
    return out


def ask(wav, sr: int, spans, gen, max_gap: float = None):
    """answers for every candidate pair of one clip: {key: [answer order 1, answer order 2]}.
    `gen(content, audio)` is a Qwen3-Omni thinker call (greedy, 4 new tokens), as benchmark/gold/grp_screen.py."""
    max_gap = float(getattr(config, "GROUP_MAX_GAP", 4.0)) if max_gap is None else max_gap
    res = {}
    for a, b in pairs(spans, max_gap):
        s, e = max(0.0, a[2] - PRE), min(len(wav) / sr, b[1] + POST)
        cut = wav[int(s * sr):int(e * sr)]
        rep = []
        for w1, w2 in ORDERS:
            q = A_Q.format(lab=a[0].lower(), o1=A_OPT[w1], o2=A_OPT[w2], w1=w1, w2=w2)
            rep.append(gen([{"type": "audio", "audio": cut}, {"type": "text", "text": q}], cut))
        res[key(a[0], a[1])] = rep
    return res


def ensure(clip: str, wav_path: Path, specs, duration: float) -> None:
    """live path for a new video: if GROUP_ASK is on and the clip has no cached answers, ask Qwen3-Omni now and cache them
    (config.GROUP_CACHE, default data/work/group_answers.json). Only candidate pairs are asked; most clips have none."""
    if not getattr(config, "GROUP_ASK", False):
        return
    path = Path(getattr(config, "GROUP_CACHE", None) or (Path(config.WORK_DIR) / "group_answers.json"))
    config.GROUP_CACHE = str(path)
    allc = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    if clip in allc:
        return
    from src.stage6_visual_augmentation import _display_spans
    config.GROUP_ASK = False
    try:
        spans = [list(s) for s in _display_spans(specs, duration)]
    finally:
        config.GROUP_ASK = True
    res = {}
    if pairs(spans, float(getattr(config, "GROUP_MAX_GAP", 4.0))):
        import librosa
        import torch
        from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
        wav, sr = librosa.load(str(wav_path), sr=16000, mono=True)
        name = getattr(config, "GROUP_MODEL", "Qwen/Qwen3-Omni-30B-A3B-Instruct")
        proc = Qwen3OmniMoeProcessor.from_pretrained(name)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(name, dtype=torch.bfloat16, device_map="auto").eval()
        try:
            model.disable_talker()
        except Exception:
            pass

        def gen(content, audio):
            conv = [{"role": "user", "content": content}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inp = proc(text=text, audio=[audio], return_tensors="pt", padding=True, use_audio_in_video=False)
            inp = inp.to(model.thinker.device).to(torch.bfloat16)
            with torch.inference_mode():
                out = model.thinker.generate(**inp, max_new_tokens=4, do_sample=False)
            return proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip().lower()

        res = ask(wav, sr, spans, gen)
        del model
        torch.cuda.empty_cache()
    allc[clip] = res
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(allc, indent=1), encoding="utf-8")
    _CACHE.pop(str(path), None)


def main(argv=None):
    """separate-process step for a rendered arm folder (one model on the GPU at a time, as listener_prep):
        python -m src.stage6_visual_augmentation.group <arm work root> <wav16 dir> [cache.json]"""
    import sys
    from src.types import AugmentationSpec
    a = list(sys.argv[1:] if argv is None else argv)
    root, wavd = Path(a[0]), Path(a[1])
    config.use_shipped()                          # the shipped display flags (MERGE_GAP 2.5, MIN_DWELL, ...)
    config.GROUP_ASK = True
    config.GROUP_CACHE = a[2] if len(a) > 2 else config.GROUP_CACHE    # default: the shipped WORK_DIR/group_answers.json
    for d in sorted(p for p in root.iterdir() if (p / "augmentations.json").exists()):
        specs = [AugmentationSpec(index=s.get("index", 0), event_label=s["event_label"], start=float(s["start"]),
                                  end=float(s["end"]), augment=bool(s.get("augment")), confidence=float(s.get("confidence", 0)),
                                  image_path=s.get("image_path"), talked_about=bool(s.get("talked_about")),
                                  spans=[tuple(x) for x in s.get("spans", [])], breaks=[tuple(x) for x in s.get("breaks", [])])
                 for s in json.loads((d / "augmentations.json").read_text(encoding="utf-8"))]
        dur = float(json.loads((d / "media.json").read_text(encoding="utf-8"))["duration"])
        ensure(d.name, wavd / f"{d.name}.wav", specs, dur)
        print("group", d.name, json.loads(Path(config.GROUP_CACHE).read_text(encoding="utf-8")).get(d.name), flush=True)


if __name__ == "__main__":
    main()
