"""Round 66 NAME-ALL (docs/prereg_round13_detector_push.md): per gate stretch, name 3 candidate makers (no "nothing" escape) plus
noun phrases of the gate's describe answer, ground each (Round 50 prompt), crop it, and read the crop question as a
null-calibrated logit margin m = [s(Q+) - s0(Q+)] - [s(Q-) - s0(Q-)] (Q- = the NOT twin; s0 = the same prompts on a grey crop with
phrase "this thing", per label). Where nothing grounds, one ViCrop relative-attention crop of the middle frame. The stretch margin
is the max over its crops. Used only when config.NAME_ALL = (t_lo, t_hi) (default None); see reason.decide_subjects.
Benchmark twin: benchmark/gold/nameall.py (same prompts and parsing, gate-gold).
"""
from __future__ import annotations

import re

N_PROMPT = ("A {label} sound is heard while these frames are on screen. List the 3 visible things most likely to be making it, "
            "most likely first, one short noun phrase per line. Always give 3, even if none fits well.")
GROUND_Q = ("These frames (numbered 1..{n}) are from the moment a sound of {label} was heard. Reply ONLY with JSON, no "
            "other text, no tools: {{\"frame\": k, \"bbox_2d\": [x1, y1, x2, y2]}} around {phrase} on a 0-1000 grid of "
            "frame k (1000 = full width or height), or {{\"bbox_2d\": null}} if {phrase} is not visible.")
CROP_Q = ("This is a close-up cut from a video frame. Is this {phrase} making the {label} sound right now? "
          "Answer yes or no.")
CROP_NQ = ("This is a close-up cut from a video frame. Is this {phrase} NOT making the {label} sound right now? "
           "Answer yes or no.")
GENERIC = "Describe this image briefly."
DETS = set("a an the two three four five several some many his her their its one another this that these those few".split())
STOPW = set(("in on with at of to from by near under over behind through down up into onto across along around while and or but "
             "as is are was were be being been has have had which who whom whose there for against beside next during between "
             "inside outside above below beneath toward towards out off it they he she").split())
_S0: dict = {}


def parse_n(reply):
    from src.stage5_cross_modal_analysis.reason import _clean_phrase
    lines = [l.strip() for l in (reply or "").splitlines() if l.strip()]
    if len(lines) < 3 and lines and re.search(r"[,;]", lines[0]):
        lines = [x.strip() for x in re.split(r"[,;]", lines[0]) if x.strip()] + lines[1:]
    out = []
    for l in lines[:3]:
        l = re.sub(r"^\s*(\d+\s*[\.\):]|[-*•])\s*", "", l).strip().strip("*").strip()
        p = _clean_phrase(l, max_words=5)
        if p:
            out.append(p)
    return out


def noun_phrases(text):
    toks = re.findall(r"[a-z][a-z'-]*|[,.;:!?]", (text or "").lower())
    out, i = [], 0
    while i < len(toks):
        if toks[i] in DETS:
            j, ph = i + 1, []
            while j < len(toks) and len(ph) < 4:
                w = toks[j]
                if not re.match(r"[a-z]", w) or w in STOPW or w in DETS or (ph and w.endswith("ing")):
                    break
                ph.append(w); j += 1
            if ph:
                out.append(" ".join(ph))
            i = j
        else:
            i += 1
    return out


def candidates(n_reply, desc_reply):
    norm = lambda p: re.sub(r"^(a|an|the)\s+", "", p.lower().strip())
    seen, out = set(), []
    for src, ps in (("N", parse_n(n_reply)), ("desc", noun_phrases(desc_reply))):
        for p in ps:
            k = norm(p)
            if k and k not in seen:
                seen.add(k); out.append({"phrase": p, "src": src})
    return out[:4]


def _s0(label, mdl, proc):
    from PIL import Image
    from src.stage5_cross_modal_analysis.reason import _yes_no_margin
    if label not in _S0:
        grey = Image.new("RGB", (224, 224), (128, 128, 128))
        _S0[label] = (_yes_no_margin(mdl, proc, CROP_Q.format(phrase="this thing", label=label), [grey]),
                      _yes_no_margin(mdl, proc, CROP_NQ.format(phrase="this thing", label=label), [grey]))
    return _S0[label]


def _margin(label, phrase, crop, mdl, proc):
    from src.stage5_cross_modal_analysis.reason import _yes_no_margin
    s0q, s0n = _s0(label, mdl, proc)
    sq = _yes_no_margin(mdl, proc, CROP_Q.format(phrase=phrase, label=label), [crop])
    sn = _yes_no_margin(mdl, proc, CROP_NQ.format(phrase=phrase, label=label), [crop])
    return (sq - s0q) - (sn - s0n)


def _vicrop_box(img, label, mdl, proc):
    """ViCrop rel-att (2502.17422): last-token attention to the image tokens under Q+ ("this thing") / under a generic prompt,
    mean over heads and the middle third of the decoder layers, 3x3-smoothed; a 40 % x 40 % box around the arg-max cell."""
    import torch
    import torch.nn.functional as F
    pad = proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    merge = int(getattr(proc.image_processor, "merge_size", 2))
    prev = getattr(mdl.config, "_attn_implementation", "sdpa")

    def att(prompt):
        content = [{"type": "image"}, {"type": "text", "text": prompt}]
        text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                        add_generation_prompt=True, enable_thinking=False)
        inp = proc(text=[text], images=[img], return_tensors="pt").to(mdl.device)
        pos = (inp["input_ids"][0] == pad).nonzero().flatten()
        t, gh, gw = [int(v) for v in inp["image_grid_thw"][0]]
        h, w = gh // merge, gw // merge
        assert len(pos) == t * h * w, (len(pos), t, h, w)
        with torch.inference_mode():
            out = mdl(**inp, output_attentions=True)
        A = out.attentions
        sel = A[len(A) // 3: 2 * len(A) // 3]
        a = torch.stack([x[0, :, -1, pos].float().mean(0) for x in sel]).mean(0).reshape(t, h, w).mean(0)
        del out
        return a, h, w

    try:
        mdl.set_attn_implementation("eager")
        aq, h, w = att(CROP_Q.format(phrase="thing", label=label))
        ag, _, _ = att(GENERIC)
    finally:
        mdl.set_attn_implementation(prev)
    rel = aq / (ag + 1e-6)
    rel = F.avg_pool2d(rel[None, None], 3, stride=1, padding=1, count_include_pad=False)[0, 0]
    i, j = divmod(int(rel.argmax()), w)
    W, H = img.size
    cx, cy = (j + 0.5) / w * W, (i + 0.5) / h * H
    bw, bh = 0.4 * W, 0.4 * H
    x0, y0 = max(0.0, min(W - bw, cx - bw / 2)), max(0.0, min(H - bh, cy - bh / 2))
    return [x0 / W * 1000, y0 / H * 1000, (x0 + bw) / W * 1000, (y0 + bh) / H * 1000]


def stretch_margin(label, frames, mdl, proc):
    """-> {"m": max margin over the stretch's crops or None, "cands": [...], "vicrop": ..., "gen": n, "prefill": n}"""
    from src.stage5_cross_modal_analysis import reason as R
    rec = {"m": None, "cands": [], "vicrop": None, "gen": 0, "prefill": 0}
    if not frames or len(frames) < 2:
        return rec
    n_reply = R._ask(mdl, proc, N_PROMPT.format(label=label), images=frames, max_new=40)
    rec["gen"] += 1
    desc = R._ask(mdl, proc, R.DESCRIBE_PROMPT, images=frames, max_new=48)      # the gate's own describe call
    ms = []
    for c in candidates(n_reply, desc):
        reply = R._ask(mdl, proc, GROUND_Q.format(n=len(frames), label=label, phrase=c["phrase"]), images=frames, max_new=64)
        rec["gen"] += 1
        status, parsed = R._box_parse(reply, len(frames), strict=True)
        c["status"] = status
        if status == "box":
            k, bb = parsed
            cr = R._box_crop(frames[k - 1], bb)
            if cr is None:
                c["status"] = "degenerate"
            else:
                c["m"] = round(_margin(label, c["phrase"], cr, mdl, proc), 4)
                rec["prefill"] += 2
                ms.append(c["m"])
        rec["cands"].append(c)
    if not ms:
        try:
            img = frames[min(2, len(frames) - 1)]
            bb = _vicrop_box(img, label, mdl, proc)
            rec["prefill"] += 2
            cr = R._box_crop(img, bb)
            if cr is not None:
                m = round(_margin(label, "thing", cr, mdl, proc), 4)
                rec["prefill"] += 2
                rec["vicrop"] = m
                ms.append(m)
        except Exception as e:                                                 # no crop (counted)
            rec["vicrop"] = "error: " + repr(e)[:120]
    rec["m"] = max(ms) if ms else None
    return rec


def sound_score(margins):
    """the floor(n/2)+1-th largest stretch margin (None = -inf): A > t <=> more than half the stretches have m > t"""
    v = sorted([m if m is not None else float("-inf") for m in margins], reverse=True)
    return v[len(v) // 2] if v else float("-inf")
