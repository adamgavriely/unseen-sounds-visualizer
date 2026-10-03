"""Stage 6 - Visual Augmentation Generation + alongside-video compositing.

v1 prototype backend = RETRIEVE: fetch a free Creative-Commons image (Openverse,
no key) for each augmented sound, cover-crop it, and record attribution. v2 will
add a 'diffusion' backend (SDXL/FLUX). 'placeholder' draws a labelled panel and
needs nothing.

Compositing shows the augmentation image for the currently-active sound event
*alongside* the original video (side-by-side), time-aligned, with the original
audio kept. See docs/history/earlier_drafts/project_notes.tex sec:placement (release v1.2.0) for the placement rationale.
"""
from __future__ import annotations

import io
import json
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image, ImageDraw

from src.types import AugmentationSpec
import config
from src.labels import search_query

_OPENVERSE = "https://api.openverse.org/v1/images/"
_HEADERS = {"User-Agent": "MscFinalProject/0.1 (academic research)"}


# ----------------------------------------------------------------------
# image helpers
# ----------------------------------------------------------------------
_FONT_CACHE = {}
# PIL's built-in bitmap font is ~11 px and unreadable on a 720 px panel, and the two
# platforms have different fonts: Windows ships Arial, while the cluster gets DejaVu
# from the conda-forge fontconfig packages. Try both, then fall back.
_FONT_PATHS = [
    "C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
]


def _font(size: int):
    """A scalable font at `size`, or PIL's default if no TrueType file is found."""
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    from PIL import ImageFont
    import glob
    candidates = list(_FONT_PATHS)
    # conda envs install fonts under <prefix>/fonts or <prefix>/share/fonts
    import sys
    candidates += glob.glob(f"{sys.prefix}/fonts/**/*.ttf", recursive=True)
    candidates += glob.glob(f"{sys.prefix}/share/fonts/**/*.ttf", recursive=True)
    for path in candidates:
        try:
            f = ImageFont.truetype(path, size)
            _FONT_CACHE[size] = f
            return f
        except Exception:
            continue
    _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]


def _cover_crop(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    tw, th = size
    w, h = img.size
    scale = max(tw / w, th / h)
    nw, nh = max(tw, int(w * scale)), max(th, int(h * scale))
    img = img.resize((nw, nh), Image.LANCZOS)
    left, top = (nw - tw) // 2, (nh - th) // 2
    return img.crop((left, top, left + tw, top + th))


def _caption(img: Image.Image, text: str) -> Image.Image:
    if not text:
        return img
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    d.rectangle([0, h - 58, w, h], fill=(0, 0, 0, 175))
    d.text((16, h - 44), text, font=_font(max(18, h // 26)), fill=(240, 240, 245))
    return img


def _prompt_caption(img: Image.Image, label: str, prompt: str) -> Image.Image:
    """The generator's prompt, printed under the picture. A debugging aid, not a
    viewer feature: it is how a developer sees WHY a picture looks the way it does."""
    if not prompt:
        return img
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    fnt = _font(max(14, h // 34))
    # wrap to the cell width
    words, lines, cur = prompt.split(), [], ""
    for wd in words:
        trial = (cur + " " + wd).strip()
        if d.textlength(trial, font=fnt) > w - 24 and cur:
            lines.append(cur); cur = wd
        else:
            cur = trial
    if cur:
        lines.append(cur)
    lines = [label.upper()] + lines[:3]
    lh = int(fnt.size * 1.35)
    top = h - (lh * len(lines) + 14)
    d.rectangle([0, top, w, h], fill=(0, 0, 0, 190))
    for i, line in enumerate(lines):
        col = (255, 220, 120) if i == 0 else (240, 240, 245)
        d.text((12, top + 7 + i * lh), line, font=fnt, fill=col)
    return img


def _fit_on_white(img: Image.Image, size: Tuple[int, int]) -> Image.Image:
    """Letterbox onto white instead of cropping to fill.

    _cover_crop turns a 768x768 generation into a 16:9 panel by cropping top and bottom
    -- which removes exactly the empty margins the pictogram prompt asked for, leaving
    the subject spanning the full height with no edge to put the sound mark against.
    Fitting keeps the whole drawing and pads with white, which is also the ground the
    prompt asks for, so the padding is invisible.
    """
    out = Image.new("RGB", size, (255, 255, 255))
    w, h = img.size
    scale = min(size[0] / w, size[1] / h)
    resized = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    out.paste(resized, ((size[0] - resized.size[0]) // 2,
                        (size[1] - resized.size[1]) // 2))
    return out


def _sound_glyph(img: Image.Image, strength: float = 1.0) -> Image.Image:
    """Stamp a sound indicator onto the image, so the panel reads without decoding.

    A photograph of a fire engine says "there is a fire engine"; it does not say "you
    are HEARING one". The viewer has to infer that the panel is about sound at all,
    which is exactly the decoding step the design review asked to remove -- and the DHH
    visualization literature is consistent that the cue should be immediate rather than
    interpreted (sec:litreview). So the source and the fact-of-sound are combined in one
    simple picture: the retrieved photograph of the source, plus the standard radiating
    arcs, plus the source named in words.

    Deliberately drawn rather than generated. It has to be identical every time, legible
    over an arbitrary photograph, and cheap; a diffusion model gives none of those.
    """
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    r = max(7, int(min(w, h) * 0.038))
    # Put the mark against the thing that is making the sound -- beside the bird, not
    # parked in the corner of the frame. The design review's example was sound drawn at the bird's
    # mouth; the subject's silhouette is findable on a plain ground, so the arcs go just
    # outside its upper edge. Photographs fill the frame and fall back to the corner.
    box = _subject_bbox(img)
    if box:
        # emitter just past the subject's edge, arcs opening AWAY from it: sound comes
        # out of the bird, so the rings must expand outward, not wrap back around it
        cx = min(w - int(r * 4.6), box[2] + int(r * 1.1))
        cy = max(int(r * 2.2), box[1] + (box[3] - box[1]) // 4)
    else:
        cx, cy = w - int(r * 5.2), int(r * 4.2)

    # On a pictogram the ground is plain and light, so dark arcs read best and a disc
    # behind them would be visual noise. Over a photograph nothing can be assumed, so
    # white arcs on a dark disc are the only reliable option.
    r_, g_, b_ = img.convert("RGB").getpixel((min(cx, w - 1), min(cy, h - 1)))
    on_light = (r_ + g_ + b_) / 3 > 140 and box is not None
    ink = (28, 32, 44, 255) if on_light else (255, 255, 255, 235)
    if not on_light:
        pad = int(r * 3.6)
        d.ellipse([cx - pad, cy - pad, cx + pad, cy + pad], fill=(0, 0, 0, 120))
    # louder sounds get one more arc: the panel shows intensity without a number
    for i in range(3 if strength >= 0.45 else 2):
        rr = r * (1.3 + 0.78 * i)
        d.arc([cx - rr, cy - rr, cx + rr, cy + rr], start=-52, end=52,
              fill=ink, width=max(2, r // 3))
    d.ellipse([cx - r * 0.38, cy - r * 0.38, cx + r * 0.38, cy + r * 0.38], fill=ink)
    return img


_PIPE = None


def unload_generator():
    """Free the image generator.

    Stage 5's reasoner and Stage 6's generator each want most of a 24 GB card. The
    generator is cached across clips on purpose -- reloading it per clip dominates the
    cost -- but that cache is exactly what makes the second clip fail: the VLM loads
    on top of a generator that is still resident. Whoever needs the GPU next asks for
    it, rather than hoping the allocator sorts it out.
    """
    global _PIPE
    if _PIPE is None:
        return
    del _PIPE
    _PIPE = None
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# "line art" is taken literally: the first attempt produced a clean bird silhouette on a
# background of black stripes, and a good fire engine boxed in by spurious bars. The
# words that actually work are the stock-photo ones -- isolated, white background,
# sticker -- which name the RESULT rather than the drawing technique.
# "sticker style" produced a genuinely good flat drawing -- sitting on a coloured badge
# disc on a grey ground, which leaves no white margin for the sound mark and no clean
# edge to find. The style words are worth keeping; the badge is not.
ICON_STYLE = ("minimalist pictogram of one {subject}, single simple silhouette, two "
              "colours only, thick clean outline, plain empty white background, app icon, "
              "instantly recognisable, no text")
ICON_NEGATIVE = ("photograph, photorealistic, realistic, 3d render, text, letters, words, "
                 "watermark, logo, caption, stripes, lines, bars, grid, frame, border, "
                 "pattern, background decoration, scenery, clutter, multiple objects, "
                 "small details, blurry, circle background, badge, sticker outline, "
                 "coloured background, grey background, gradient, shadow, vignette, "
                 "ornate, decorative, folk art, papercut, floral, leaves, intricate, "
                 "symmetrical pattern, engraving, woodcut, tattoo, mandala")


# "single subject" used to be here and is gone: the depiction is now an EVENT
# ("a crowd applauding", "rain falling on a street"), and an event is not always
# one thing. The plain background still does the job the constraint was for.
PLAIN_TAIL = ", plain white background, clearly visible"


def plain_prompt(subject: str) -> str:
    """State the thing, and nothing about how to draw it.

    Six rounds were spent art-directing this ("minimal line art", "3d icon", "sticker",
    "flat cartoon") and every directive made the output worse: literal black stripes, a
    badge on grey, papercut folk art. The proposal asks for "static storyboard-style
    images", which is a plain depiction of the event, and a model with real prompt
    adherence produces exactly that when simply asked. The only additions earn their
    place: an isolated subject on white keeps the panel readable at a glance AND stops
    the background inventing things -- a bare "fire engine with its siren on" put the
    engine in FLAMES, which would tell a deaf viewer "fire" when the sound is a siren.
    """
    return f"{subject}{PLAIN_TAIL}"


def icon_prompt(subject: str) -> str:
    """Ask for a pictogram rather than a picture.

    The generator ablation had SDXL rendering "a clear, simple illustration of: Siren"
    and losing to a stock photograph, because it produced a plausible SCENE whose
    subject the viewer -- and the describing VLM -- had to work out. A panel beside a
    video gets a glance, not a study; what it needs is the least ambiguous possible
    depiction of one thing.

    So the prompt asks for the thing diffusion is reliably good at and photographs are
    not: one large object, flat, few shapes, plain background, nothing else in frame.
    That also makes the subject's silhouette findable in code, which is what lets the
    sound indicator be placed against the object instead of parked in a corner.
    """
    return ICON_STYLE.format(subject=subject)


def _subject_bbox(img: Image.Image, tol: int = 28):
    """Where the drawn thing is, assuming a plain background.

    Icon-style output sits on a near-uniform ground, so anything differing from the
    corner colour is the subject. Returns None when that assumption does not hold (a
    photograph fills the frame), and the caller falls back to a fixed corner.
    """
    try:
        rgb = img.convert("RGB")
        w, h = rgb.size
        # The prompt asks for a white ground, so "not near-white" is the subject. This
        # survives the faint gradients diffusion leaves behind, which a corner-colour
        # comparison does not -- that version rejected 7 of 8 real generations.
        mask = rgb.convert("L").point(lambda v: 255 if v < 238 else 0)
        box = mask.getbbox()
        if not box:
            return None
        cover = ((box[2] - box[0]) * (box[3] - box[1])) / float(w * h)
        ink = mask.histogram()[255] / float(w * h)
        # A photograph covers the frame; so does a generation that filled the background
        # with decoration. Either way there is no clean edge to put the mark against.
        if cover > 0.92 or ink > 0.62:
            return None
        return box
    except Exception:
        return None


# PICTURE_V3: what the generator is told to leave out. Qwen-Image runs true classifier-free guidance
# and was being called with an empty negative, so scenery was free to appear (the palace crowd, the
# street firecracker, thunder as a full-frame sky). A diffusion negative has no clauses, so any word
# that names the source itself is removed from it per picture (a crowd must not be negated away).
NEGATIVE_V3 = ["scenery", "landscape", "background scene", "room interior", "street", "buildings",
               "sky background", "text", "letters", "watermark"]
# "people" was in this list and is not any more (P4, round 3): whenever a sound is made by people the
# subject does not always say so ("Someone coughing", "Runner's feet"), and negating people there
# pushes the generator away from the source -- the one route to a false message.


def negative_for(subject: str) -> str:
    words = {w.strip(",.").lower() for w in subject.split()}
    human = words & {"people", "crowd", "audience", "person", "man", "woman", "child", "children",
                     "baby", "hands", "clapping", "laughing", "cheering", "applause", "laughter"}
    weather = words & {"cloud", "clouds", "lightning", "rain", "raining", "storm", "thunder",
                       "thunderstorm", "wind", "snow", "hail"}
    keep = [n for n in NEGATIVE_V3 if not (set(n.split()) & words)
            and not (weather and n in ("sky background", "landscape"))]
    return ", ".join(keep)


def _diffusion_image(path: Path, prompt: str, size=(1024, 1024),
                     model: str = "stabilityai/stable-diffusion-xl-base-1.0",
                     device: str = "cuda", seed: Optional[int] = None,
                     negative: Optional[str] = None) -> bool:
    """v2-b backend: generate the augmentation with SDXL (GPU). One pipeline is
    kept loaded across calls -- model load dominates cost, generation is ~2 s."""
    global _PIPE
    try:
        import torch
        from diffusers import AutoPipelineForText2Image
        is_flux = "flux" in model.lower()
        # Qwen-Image-2512 (v4, docs/history/preregistrations/prereg_v4.md, release v1.2.0): 20B MMDiT + a Qwen2.5-VL text encoder,
        # ~57 GB in bf16 -- an A100-80 / RTX Pro 6000 job, no offload. Its guidance is
        # "true" classifier-free guidance (true_cfg_scale), the card's recommended 50/4.0.
        is_qwen = "qwen-image" in model.lower()
        if _PIPE is None:
            from diffusers import DiffusionPipeline
            loader = DiffusionPipeline if is_qwen else AutoPipelineForText2Image
            _PIPE = loader.from_pretrained(
                model, torch_dtype=(torch.bfloat16 if (is_flux or is_qwen) else torch.float16)
                if device == 'cuda' else torch.float32,
                use_safetensors=True)
            if is_flux and device == "cuda" and torch.cuda.get_device_properties(0).total_memory < 40e9:
                # FLUX is ~24 GB in bf16 and the L4 exposes 22. Whole-component
                # offload still OOMs; sequential offload moves one module at a time,
                # ~25 s per 512 px image instead of ~2 s, and it fits.
                _PIPE.enable_sequential_cpu_offload()
            else:
                _PIPE = _PIPE.to(device)
            _PIPE.set_progress_bar_config(disable=True)
        kw = dict(prompt=prompt, width=size[0], height=size[1])
        if is_qwen:
            kw.update(num_inference_steps=50, true_cfg_scale=4.0, negative_prompt=negative or " ")
        elif 'turbo' in model or (is_flux and "schnell" in model.lower()):
            # Distilled: trained for very few steps and ignores classifier-free
            # guidance, so a negative prompt does nothing here and a high guidance
            # scale degrades it. Also ~7x cheaper per image, which matters at 300.
            kw.update(num_inference_steps=4, guidance_scale=0.0)
        else:
            kw.update(num_inference_steps=25, guidance_scale=4.5)
        if is_flux:
            kw["max_sequence_length"] = 256
        if seed is not None:
            # unseeded by default (as shipped); a seed makes two variants of the same picture
            # comparable, which the picture bench needs and the thesis will want
            kw["generator"] = torch.Generator(device=device).manual_seed(int(seed))
        img = _PIPE(**kw).images[0]
        img.save(path)
        return True
    except Exception as e:
        print(f"       [stage6] diffusion failed ({type(e).__name__}: {e}); "
              f"falling back to placeholder")
        return False


def _placeholder_image(path: Path, caption: str, size=(1024, 1024)) -> None:
    img = Image.new("RGB", size, color=(20, 22, 30))
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, size[0] - 8, size[1] - 8], outline=(90, 100, 140), width=3)
    d.text((28, 28), "[augmentation placeholder]", fill=(150, 160, 190))
    _caption(img, caption)
    img.save(path)


# ----------------------------------------------------------------------
# Openverse retrieval (salvaged from the legacy pipeline)
# ----------------------------------------------------------------------
def _http_get(url: str, timeout: int = 25) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _search(query: str, page_size: int = 8) -> list:
    qs = urllib.parse.urlencode({"q": query, "page_size": page_size, "mature": "false"})
    return json.loads(_http_get(f"{_OPENVERSE}?{qs}")).get("results", [])


def _query_variants(query: str) -> list:
    words = query.split()
    variants, seen = [], set()
    for n in (len(words), 6, 4, 3, 2, 1):
        v = " ".join(words[:n]).strip()
        if v and v.lower() not in seen:
            seen.add(v.lower())
            variants.append(v)
    return variants


def _retrieve(query: str, out_path: Path, size: Tuple[int, int]) -> Optional[dict]:
    """Download a relevant CC image to out_path; return attribution dict or None."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kw = query.split()[0].lower() if query else ""
    for q in _query_variants(query):
        try:
            results = _search(q)
        except Exception:
            continue
        # prefer results whose title actually mentions the keyword (mild on-topic bias)
        results.sort(key=lambda r: 0 if kw and kw in (r.get("title") or "").lower() else 1)
        for r in results:
            url = r.get("url") or r.get("thumbnail")
            if not url:
                continue
            try:
                img = Image.open(io.BytesIO(_http_get(url))).convert("RGB")
                _caption(_sound_glyph(_cover_crop(img, size)), query).save(out_path)
                return {"requested": query, "matched_query": q, "title": r.get("title"),
                        "creator": r.get("creator"), "license": r.get("license"),
                        "license_url": r.get("license_url"),
                        "source": r.get("foreign_landing_url") or url}
            except Exception:
                continue
    return None


# ----------------------------------------------------------------------
# Stage 6 entry: produce one image per augmented sound
# ----------------------------------------------------------------------
def _is_blank(path: Path) -> bool:
    """A generated picture with nothing on it: every pixel within a shade of every other."""
    try:
        import numpy as np
        from PIL import Image
        return float(np.asarray(Image.open(path).convert("L"), dtype="float32").std()) < 3.0
    except Exception:
        return False


def _final_picture(spec, path: Path, work_dir: Path, query: str, size, model: str, device: str) -> None:
    """One picture of the frozen final setup: a burst card for a sound with no maker, a fixed template subject for
    a sound with a canonical maker, otherwise the V3.1 subject; the rules tail; the template's extra negative words;
    seed_of + 1 as in the screening arms; one redraw when the picture is nearly blank (ink < 0.05)."""
    from PIL import Image
    from benchmark.gold.gen_screen import (TEMPLATES, TEMPLATE_NEG, CARDS, RULES_TAIL, burst_card, ink, seed_of,
                                           negative_for as screen_negative)
    source = getattr(spec, "source", "") or spec.event_label
    word = CARDS.get(source) or CARDS.get(spec.event_label)
    if word:
        burst_card(word).resize(size).save(path)
        spec.image_prompt, spec.image_path, spec.backend = "CARD:" + word, str(path), "card"
        return
    key = source if source in TEMPLATES else (spec.event_label if spec.event_label in TEMPLATES else None)
    subject = TEMPLATES[key] if key else (spec.subject or query)
    prompt = subject + RULES_TAIL
    neg = ", ".join(x for x in (screen_negative(subject), TEMPLATE_NEG.get(key or "", "")) if x) or " "
    seed = seed_of({"clip": work_dir.name, "label": spec.event_label, "start": float(spec.start)}) + 1
    import shutil
    verify = bool(getattr(config, "PICTURE_VERIFY", False))
    tries = int(getattr(config, "PICTURE_VERIFY_TRIES", 5)) if verify else 1
    log = {"clip": work_dir.name, "index": spec.index, "label": spec.event_label, "source": source,
           "subject": subject, "tries": []}
    ok = False
    learned, saw_text = [], False               # refinement carried from each refused try to the next
    if verify:
        from src.stage6_visual_augmentation.verify import rewrite_first
    # PICTURE_SENSE (src/stage6_visual_augmentation/sense.py, 28 Sept, test docs/history/analyses/picture_sense_test_2026-09-28.md, release v1.2.0): the
    # generic method instead of the AMBIGUOUS table -- the slot sentence and the mined negatives from try 1, sense.check;
    # templates are left as they are
    sense = bool(verify and getattr(config, "PICTURE_SENSE", False))
    sp = None
    if sense and not key:
        from src.stage6_visual_augmentation.sense import plan as sense_plan
        sp = sense_plan(spec, subject, model, device, size)
        log["sense"] = sp
    first = bool(verify and not sense and rewrite_first(spec, subject))
    for t in range(tries):
        # PICTURE_VERIFY (src/stage6_visual_augmentation/verify.py): try 1 is exactly the shipped picture (same seed);
        # each later try a new seed (stride 1000, clear of the blank guard's +1); from try 3 the clearer fixed rewrite
        # for an ambiguous word, "no text" in the prompt and text words in the negative. Refined each time
        # (28 Sept): every refused try adds what the VLM saw instead (verify.feedback_negative) to the next negative, and
        # text found by OCR switches "no text" on from the next try
        subj_t, prompt_t, neg_t, seed_t = subject, prompt, neg, seed + 1000 * t
        if sp:
            subj_t, prompt_t = sp["subject"], sp["subject"] + RULES_TAIL
            neg_t = ", ".join(x for x in (screen_negative(subj_t), sp["neg"]) if x) or " "
        # an entry flagged rewrite_first (smoke detector, 28 Sept) uses its clearer fixed wording from try 1: the
        # checker cannot tell its look-alike (a dome camera) from it, so only the wording keeps the look-alike out
        if t >= 2 or saw_text or first:
            from src.stage6_visual_augmentation.verify import rewrite_for
            rw = None if sense else rewrite_for(spec, subject)
            if rw:
                subj_t = rw["subject"]
                neg_t = ", ".join(x for x in (screen_negative(subj_t), TEMPLATE_NEG.get(key or "", ""), rw["neg"])
                                  if x)
            prompt_t = subj_t + RULES_TAIL + ", no text, no letters, no signs"
            neg_t = ", ".join(x for x in (neg_t.strip(), "text, letters, words, writing, sign, label, logo") if x)
        if learned:
            neg_t = ", ".join(x for x in (neg_t.strip(), ", ".join(learned)) if x)
        spec.image_prompt = prompt_t
        ok = _diffusion_image(path, prompt_t, size, model=model, device=device, seed=seed_t, negative=neg_t)
        if ok and ink(Image.open(path)) < 0.05:
            ok = _diffusion_image(path, prompt_t, size, model=model, device=device, seed=seed_t + 1, negative=neg_t)
            print(f"       [stage6] {spec.event_label}: the picture came out nearly blank; redrew it")
        if not ok or not verify:
            break
        if sense:
            from src.stage6_visual_augmentation.sense import check
        else:
            from src.stage6_visual_augmentation.verify import check
        res = check(path, spec, subject, salt=0, tag=work_dir.name)
        log["tries"].append({"try": t + 1, "seed": seed_t, "prompt": prompt_t, "ok": res["ok"],
                             "picked": res["mc"]["picked"], "intended": res["mc"]["intended"],
                             "options": res["mc"]["options"], "text": res["text"]["words"], "saw": res.get("saw", "")})
        print(f"       [verify] {spec.event_label} try {t + 1}: {'OK' if res['ok'] else 'REJECT'} "
              f"picked={res['mc']['picked']!r} text={res['text']['words']} saw={res.get('saw', '')!r}", flush=True)
        if res["ok"]:
            break
        from src.stage6_visual_augmentation.verify import feedback_negative
        fb = feedback_negative(res["mc"]["picked"], res["mc"]["intended"], subj_t if sp else subject, spec)
        for w in (x.strip() for x in fb.split(",")):
            if w and w not in learned:
                learned.append(w)
        saw_text = saw_text or bool(res["text"]["words"])
        log["tries"][-1]["learned"] = list(learned)
        if t < tries - 1:                        # keep each refused try for the audit trail
            shutil.copy2(path, path.with_name(f"{path.stem}_try{t + 1}.png"))
    else:
        if verify and ok:                        # every try refused: a word card with the sound's name
            from src.stage6_visual_augmentation.verify import card_word
            shutil.copy2(path, path.with_name(f"{path.stem}_try{tries}.png"))
            word = card_word(spec)
            burst_card(word, fit=True).resize(size).save(path)
            spec.image_prompt, spec.image_path, spec.backend = "VCARD:" + word, str(path), "card"
            log["final"] = "word card"
            VERIFY_LOG.append(log)
            return
    if verify:
        log["final"] = ("picture" if len(log["tries"]) <= 2 and not first else "rewritten") if ok else "placeholder"
        if ok and len(log["tries"]) > 2 and not rewrite_applies(spec, subject):
            log["final"] = "picture"             # later tries without a table rewrite: seed / no-text / feedback only
        VERIFY_LOG.append(log)
    if ok:
        spec.image_path, spec.backend = str(path), "diffusion"
    else:
        _placeholder_image(path, query, size)
        spec.image_path, spec.backend = str(path), "placeholder"


VERIFY_LOG: list = []          # one entry per verified picture (PICTURE_VERIFY); generate_augmentations writes it out


def rewrite_applies(spec, subject: str) -> bool:
    from src.stage6_visual_augmentation.verify import rewrite_for
    return rewrite_for(spec, subject) is not None


def generate_augmentations(specs: List[AugmentationSpec], work_dir: Path,
                           backend: str = "retrieve", size=(1024, 1024),
                           model: str = "", device: str = "cpu") -> List[AugmentationSpec]:
    out_dir = work_dir / "augmentations"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("aug_*.png"):
        old.unlink()

    credits, n = [], 0
    for spec in specs:
        if not spec.augment:
            continue
        path = out_dir / f"aug_{spec.index:03d}.png"
        query = search_query(spec.subject or spec.event_label)
        if backend == "retrieve":
            meta = _retrieve(query, path, size)
            if meta:
                spec.image_path = str(path)
                spec.backend = "retrieve"
                credits.append({"index": spec.index, "label": spec.event_label, **meta})
            else:                                   # graceful fallback
                _placeholder_image(path, query, size)
                spec.image_path = str(path)
                spec.backend = "placeholder"
        elif backend == "diffusion" and getattr(config, "PICTURE_FINAL", False):
            # The frozen final picture setup (docs/history/preregistrations/freeze_picture_setup_2026-09-25.md, release v1.2.0), for demo videos only
            # (week plan C.1, level 2: switched on only if the blind confirmation passes). One source of truth:
            # the templates, cards, rules tail and blank guard are imported from the screening code as frozen.
            _final_picture(spec, path, work_dir, query, size, model, device)
        elif backend == "diffusion":                # v2-b, university GPU
            # Always the pictogram prompt: what Stage 5 stored is the subject, and the
            # style is this backend's business, not the gate's.
            prompt = plain_prompt(spec.subject or query)
            spec.image_prompt = prompt          # the exact string the generator saw
            neg = negative_for(spec.subject or query) if getattr(config, "PICTURE_V3", False) else None
            if _diffusion_image(path, prompt, size, model=model, device=device, negative=neg):
                # Two of the 33 pictures in the final DEV render came out a plain white square
                # (2026-09-24). The panel changes, the viewer looks, and there is nothing there --
                # the full price of a picture for none of the information. One retry at a
                # different seed costs a few seconds and the failure is rare.
                if _is_blank(path) and _diffusion_image(path, prompt, size, model=model,
                                                        device=device, negative=neg):
                    print(f"       [stage6] {spec.event_label}: the picture came out blank; redrew it")
                spec.image_path = str(path)
                spec.backend = "diffusion"
            else:
                _placeholder_image(path, query, size)
                spec.image_path = str(path)
                spec.backend = "placeholder"
        else:                                       # 'placeholder'
            _placeholder_image(path, spec.image_prompt or query, size)
            spec.image_path = str(path)
            spec.backend = "placeholder"
        n += 1

    (work_dir / "credits.json").write_text(
        json.dumps(credits, indent=2, ensure_ascii=False), encoding="utf-8")
    mine = [r for r in VERIFY_LOG if r.get("clip") == work_dir.name]
    if mine:                                        # PICTURE_VERIFY: tries, what the VLM saw, final kind
        (work_dir / "picture_verify.json").write_text(json.dumps(mine, indent=1, ensure_ascii=False),
                                                      encoding="utf-8")
    print(f"       [stage6] {backend}: produced {n} augmentation image(s); "
          f"{len(credits)} retrieved (rest placeholder).")
    return specs


# ----------------------------------------------------------------------
# alongside-video compositor
#
# A picture appears when its sound is heard and goes away when the sound stops.
# That sounds obvious, and the first version did not do it: every augmented sound
# held a slot for the whole clip, dimmed when idle, so a viewer saw three pictures
# permanently parked beside the video and the panel read as a collage rather than as
# a system reacting to anything. It also broke the claim the panel is supposed to
# make -- a picture that is on screen when the sound is not is telling the viewer
# something false.
#
# What the DHH-visualization evidence actually asks for (notes, "Rendering
# improvements") is that visuals not MOVE, because moving and popping targets cost a
# viewer who is already splitting attention between the video and the panel. Position
# stability and permanent presence are different things. So a sound keeps a fixed
# position for as long as it is on screen, and when it is not heard, its cell is
# empty. Nothing jumps; nothing lingers.
#
# The number of rows is the maximum number of sounds heard AT ONCE, not the number of
# distinct sounds in the clip, so three sounds that never overlap share one full-size
# cell in turn instead of splitting the panel into three thin strips that are empty
# most of the time.
# ----------------------------------------------------------------------
MAX_SLOTS = 3   # granularity requirement (notes sec:granularity): >=3 simultaneous
                # sources never observed on the benchmark; more would split attention

# A detected span can be a fifth of a second (AED_MIN_DUR), and a picture flashed for
# 200 ms is a distraction rather than information -- the viewer is watching the video,
# and has to look across, recognise a picture, and look back. So a picture stays up for
# at least MIN_DWELL, and two bursts of the same sound closer than MERGE_GAP are one
# appearance rather than a flicker. Both are display decisions: they change how long a
# picture is shown, never whether the sound was detected or gated.
MIN_DWELL = 1.5
MERGE_GAP = 0.8


def _display_spans(specs: List[AugmentationSpec], duration: float, require_image: bool = True, clip: str = None):
    """(label, start, end, spec) intervals to actually put on screen.

    Merges repeats of the same sound that are closer together than MERGE_GAP, and gives
    every appearance at least MIN_DWELL on screen.
    """
    if getattr(config, "DEPICT_EVENT", False):       # Round 57 DEPICT-EVENT: drop pictures of a visibly happening, look-alike event
        from src.stage6_visual_augmentation.depict import filter_specs as _depict
        specs = _depict(specs, clip)
    dwell = float(getattr(config, "MIN_DWELL", MIN_DWELL))
    after = getattr(config, "MAX_AFTER_END", None)
    # gap < the stretched tail would put one label in two rows at once; with MAX_AFTER_END the tail is at most that long
    gap = max(float(getattr(config, "MERGE_GAP", MERGE_GAP)), dwell if after is None else min(dwell, float(after)))
    cap = getattr(config, "MAX_SPAN", None)      # picture-level cap (amendment 3)
    by_label = {}
    floor = getattr(config, "PICTURE_MIN_CONF", None)
    for s in sorted((s for s in specs if s.augment and (s.image_path or not require_image)
                     and (floor is None or s.confidence >= floor)),
                    key=lambda s: s.start):
        by_label.setdefault(s.event_label, []).append(s)
    spans = []
    cur_raw = {}                                   # span -> the real end of its last burst (before dwell)
    for label, group in by_label.items():
        cur = None; raw_end = None
        # every burst of the sound gets the picture; a spec with no burst list is an
        # older artefact and falls back to its single start/end
        bursts = [(s, a, b) for s in group
                  for a, b in (getattr(s, "spans", None) or [(s.start, s.end)])]
        # R13-6: a recorded break (the family's evidence gone for >= the retrigger gap, or a new sound) is never bridged;
        # no spec carries one unless config.RETRIGGER / RETRIGGER_RAW was on in stage 4, so nothing else changes
        from src.labels import crosses_break
        brk = sorted({tuple(x) for s in group for x in (getattr(s, "breaks", None) or [])})
        for s, a0, b0 in sorted(bursts, key=lambda t: t[1]):
            a, b = max(0.0, a0), min(float(duration), max(b0, a0 + dwell))
            # Amendment 3 (2026-09-21, bug fixes): the gap is measured from the sound's REAL
            # end, not from the stretched end (raw end + dwell), which chained repeats up to
            # gap + dwell apart into one picture and hid every later onset; and a chain never
            # runs past its first start + MAX_SPAN -- the stage-4 cap was undone here.
            if cur and a - raw_end <= gap and (not cap or a < cur[1] + cap) and not (brk and crosses_break(brk, raw_end, a0)):
                # same sound again, right away: extend rather than blink
                _trail_join(s, a0, b0, cur, a - raw_end, gap)
                cur[2] = max(cur[2], b)
                raw_end = max(raw_end, b0)
                if s.confidence > cur[3].confidence:
                    cur[3] = s
            else:
                cur = [label, a, b, s]; raw_end = b0
                spans.append(cur)
            cur_raw[id(cur)] = raw_end
    for sp in spans:                                 # MAX_AFTER_END (28 Sept): at most this long past the real end
        sp[2] = min(float(duration), max(sp[2], sp[1] + dwell))
        if cap:
            sp[2] = min(sp[2], sp[1] + float(cap))
        if after is not None:
            sp[2] = min(sp[2], max(cur_raw[id(sp)], sp[1]) + float(after))
    _trail_windows(spans, dwell, after, gap)
    if getattr(config, "GROUP_ASK", False):          # Round 47 GROUP (1 Oct): Omni-confirmed repeats -> one picture
        from src.stage6_visual_augmentation.group import apply as _group
        spans = _group(spans, clip)
    return [tuple(sp) for sp in sorted(spans, key=lambda sp: sp[1])]


def _assign_rows(spans):
    """Give each appearance a row, so overlapping sounds never share one.

    Standard interval colouring: reuse the first row whose previous occupant has
    finished. The number of rows that come out is the maximum number of sounds heard at
    the same time, which is what the panel should be divided into -- a clip with three
    sounds that never overlap gets one full-size cell, not three thin ones.
    """
    rows, placed = [], []
    for label, a, b, spec in spans:
        for r, free_at in enumerate(rows):
            if a >= free_at - 1e-6:
                rows[r] = b
                placed.append((r, label, a, b, spec))
                break
        else:
            rows.append(b)
            placed.append((len(rows) - 1, label, a, b, spec))
    limit = int(getattr(config, "MAX_SLOTS", MAX_SLOTS))
    if len(rows) > limit:
        # More simultaneous sounds than the panel can carry: keep the loudest, and drop
        # the rest rather than shrinking every cell past legibility.
        # Priority: a sound people are reacting to on the soundtrack first, then the
        # loudest. Confidence measures loudness; "did you hear that?" about a quiet
        # sound is better evidence that it matters than the decibel level is.
        rank = lambda p: (not getattr(p[4], "talked_about", False), -p[4].confidence)
        # Amendment 3 (2026-09-21, bug fix): the old re-pack walked the sounds in rank order
        # and compared a quiet sound's start with a LOUDER, LATER sound's end, so a quiet
        # sound that overlapped nothing was dropped. Now a sound is kept, in rank order, iff
        # adding it keeps the number of sounds on screen at any moment within the limit;
        # rows are then assigned in time order as usual.
        chosen = []
        for cand in sorted(placed, key=rank):
            trial = chosen + [cand]
            edges = sorted([(p[2], 1) for p in trial] + [(p[3], -1) for p in trial], key=lambda e: (e[0], e[1]))
            depth = peak = 0
            for _, d in edges:
                depth += d; peak = max(peak, depth)
            if peak <= limit:
                chosen.append(cand)
            else:
                _trail_slot(cand, limit)
        rows, placed = [], []
        for _, label, a, b, spec in sorted(chosen, key=lambda p: p[2]):
            for r, free_at in enumerate(rows):
                if a >= free_at - 1e-6:
                    rows[r] = b
                    placed.append((r, label, a, b, spec))
                    break
            else:
                rows.append(b)
                placed.append((len(rows) - 1, label, a, b, spec))
    return placed, max(1, len(rows))


def _timeline(specs: List[AugmentationSpec], duration: float, extra_bounds=()):
    """Contiguous segments over [0,duration]; each carries {row: (label, spec)}.

    ``extra_bounds`` adds cut points that do not come from the augmentations -- the
    debug strip uses them so the list of raw detections can change while no picture
    does."""
    placed, n_rows = _assign_rows(_display_spans(specs, duration))
    extra = {min(max(0.0, float(t)), float(duration)) for t in extra_bounds}
    if not placed:
        bounds = sorted({0.0, float(duration)} | extra)
        return [({}, b - a) for a, b in zip(bounds, bounds[1:]) if b - a >= 0.05] \
            or [({}, duration)], 0, []
    bounds = sorted({0.0, float(duration)} | {p[2] for p in placed}
                    | {min(p[3], duration) for p in placed} | extra)
    segs = []
    for a, b in zip(bounds, bounds[1:]):
        if b - a < 0.05:
            continue
        mid = (a + b) / 2
        active = {}
        for r, label, s0, s1, spec in placed:
            if s0 <= mid < s1:
                active[r] = (label, spec)
        segs.append((active, b - a))
    return segs or [({}, duration)], n_rows, placed


def _opacity(confidence: float) -> float:
    """Confidence -> visual weight: faint sounds render translucent, strong ones
    solid (evidence: SoundVizVR loudness encoding / Fortnite distance-as-opacity)."""
    if not getattr(config, "CONFIDENCE_FADE", True):
        return 1.0
    return 0.45 + 0.55 * max(0.0, min(1.0, confidence / 0.6))


def _chip_color(label: str) -> tuple:
    palette = [(214, 93, 76), (76, 145, 214), (98, 180, 106), (206, 164, 66),
               (160, 108, 208), (72, 180, 178)]
    return palette[hash(label) % len(palette)]


def _render_slot(canvas: Image.Image, box: tuple, spec: Optional[AugmentationSpec],
                 label: str, mode: str) -> None:
    """Draw one cell: the picture while its sound is heard, empty ground when it is not.

    An idle cell used to keep the picture at 19% opacity, on the reasoning that a
    ghosted image reads as "heard a moment ago" and a black rectangle reads as a broken
    player. It does not survive contact with a viewer: what it actually produced was
    several pictures permanently on screen, which is what a panel showing nothing should
    never look like. The panel says "this sound is happening now", so when no sound is
    happening it has to say nothing, and the empty cell is drawn in the same ground as
    the rest of the panel so that it reads as part of the panel rather than as a hole
    in it.
    """
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    d = ImageDraw.Draw(canvas, "RGBA")
    if spec is None or not spec.image_path:      # nothing sounding in this cell
        d.rectangle(box, fill=(16, 18, 24))
        return
    if mode == "minimal":                        # label chip, no imagery
        d.rectangle(box, fill=(24, 26, 34))
        r, g, b = _chip_color(label)
        a = int(255 * _opacity(spec.confidence))
        d.rectangle([x0, y0, x0 + 10, y1], fill=(r, g, b, a))
        d.text((x0 + 24, (y0 + y1) // 2 - 10), label, font=_font(max(16, h // 12)),
               fill=(230, 232, 240, a))
        return
    # Glyph BEFORE caption: the caption bar spans the full width, and drawing it first
    # makes the subject's bounding box the whole frame, which parks the sound mark in
    # the corner instead of against the thing making the sound. Both are opt-in
    # (config) and off by default. Fill the cell: letterboxing was for pictograms,
    # whose empty margins were the ground the sound symbol sat in.
    raw = Image.open(spec.image_path).convert("RGB")
    # Fit, do not crop. A square FLUX image in a 720x285 cell lost its top and bottom
    # -- the subject with them. The generator puts everything on plain white, so the
    # letterbox padding is invisible.
    img = _fit_on_white(raw, (w, h))
    if getattr(config, "SHOW_SOUND_GLYPH", False):
        img = _sound_glyph(img, spec.confidence)
    if getattr(config, "SHOW_LABELS", False):
        img = _caption(img, label)
    if getattr(config, "SHOW_PROMPT", False):
        img = _prompt_caption(img, label, spec.image_prompt or spec.subject)
    img = img.convert("RGBA")
    img.putalpha(int(255 * _opacity(spec.confidence)))
    base = Image.new("RGBA", (w, h), (16, 18, 24, 255))
    canvas.paste(Image.alpha_composite(base, img).convert("RGB"), (x0, y0))
    d.line([x0, y1 - 1, x1, y1 - 1], fill=(40, 44, 56))


DEBUG_STRIP = 260   # px under the panel for the detection list (debug only)


def _decision_for(label: str, specs: List[AugmentationSpec], t: float = None,
                  placed=None) -> str:
    """What the gate did with a raw detection AT THIS MOMENT, for the debug strip.

    "SHOWN" used to mean the sound had a picture somewhere in the clip, so a siren
    hidden for its last stretch (car on screen) still read SHOWN while nothing was on
    the panel. Now it means a picture is on the panel right now; a sound that has one
    elsewhere reads "not now"."""
    from src.labels import canonical, is_salient_nonspeech
    if not is_salient_nonspeech(label):
        return "speech/music"
    fam = canonical(label)
    for sp in specs:
        if sp.event_label == fam or sp.event_label == label:
            r = sp.reason.lower()
            if sp.augment:
                # SHOWN means a picture for it is on the panel at this instant -- read
                # from what the compositor actually placed, not from the sound's spans:
                # with three rows and four overlapping sounds the quietest is not drawn,
                # and the strip used to say SHOWN over an empty cell.
                if placed is not None:
                    if any(lab == sp.event_label and a <= t < b for _, lab, a, b, _ in placed):
                        return "SHOWN"
                    ever = any(lab == sp.event_label for _, lab, _, _, _ in placed)
                    return "not now" if ever else "no room (3 rows)"
                return "SHOWN"
            if "visible" in r:
                return "visible"
            if "same source" in r or "same picture" in r or "same sound" in r:
                return "merged"
            if "below" in r or "nothing backs" in r:
                return "faint"
            if "out of place" in r:
                return "vetoed: place"
            if "no depiction" in r:
                return "undrawable"
            return sp.reason[:18]
    return "dropped"


def _debug_strip(canvas: Image.Image, box: tuple, t: float, events,
                 specs: List[AugmentationSpec], top: int = 8, placed=None) -> None:
    """Every raw detection active at time t, loudest first, with the gate's verdict.

    Diagnosis, not presentation. Design review: "every sound recognized should be written, so I
    can see in the video what is heard and what it decided." When a picture is wrong
    this strip says whether the detector heard the wrong thing or the gate did the
    wrong thing with the right one -- which are different repairs."""
    x0, y0, x1, y1 = box
    d = ImageDraw.Draw(canvas, "RGBA")
    d.rectangle(box, fill=(10, 11, 15))
    d.line([x0, y0, x1, y0], fill=(60, 64, 80))
    # Speech and music are never candidates, and on a talky clip they took two of the six
    # rows while a half-second bark went unlisted. Design review: leave them out.
    from src.labels import is_salient_nonspeech, canonical
    gate = float(getattr(config, "DISPLAY_THRESHOLD", 0.0))
    # One row per FAMILY, like the panel: four siren sub-labels (police, fire engine,
    # ambulance, civil defence) are one Siren picture, so they are one row here, with
    # the loudest sub-label in brackets. Nothing below the gate is listed.
    fam = {}
    for e in events:
        if not (e.start <= t < e.end and is_salient_nonspeech(e.label)
                and e.confidence >= gate):
            continue
        f = canonical(e.label)
        if f not in fam or e.confidence > fam[f][0]:
            fam[f] = (e.confidence, e.label)
    active = sorted(fam.items(), key=lambda kv: -kv[1][0])[:top]
    fnt = _font(max(12, (y1 - y0) // 12))
    lh = int(fnt.size * 1.3)
    d.text((x0 + 10, y0 + 5), f"t = {t:4.1f}s   heard, above {gate:.2f}", font=fnt,
           fill=(120, 127, 143))
    if not active:
        d.text((x0 + 10, y0 + 5 + lh), "(nothing above the gate)", font=fnt,
               fill=(96, 103, 118))
        return
    colours = {"SHOWN": (120, 220, 140), "visible": (240, 200, 90),
               "merged": (150, 170, 240), "faint": (130, 130, 140),
               "speech/music": (110, 110, 120), "dropped": (200, 110, 110)}
    for i, (f, (conf, sub)) in enumerate(active):
        verdict = _decision_for(sub, specs, t, placed)
        col = colours.get(verdict, (200, 200, 210))
        line = f"{conf:.2f}  {f[:20]}" + (f"  ({sub[:22]})" if sub != f else "")
        d.text((x0 + 10, y0 + 5 + (i + 1) * lh), line, font=fnt, fill=(225, 228, 235))
        d.text((x1 - 10 - d.textlength(verdict, font=fnt), y0 + 5 + (i + 1) * lh),
               verdict, font=fnt, fill=col)


def composite_alongside(video_path: Path, specs: List[AugmentationSpec],
                        out_path: Path, duration: float,
                        panel: int = 720, fps: int = 25,
                        mode: str = "full", events=None) -> Path:
    """Side-by-side: original video (left) + time-aligned augmentation panel (right).

    mode: "full" (imagery in stable slots) | "minimal" (label chips) |
    "off" (no panel: re-encode the original as-is, the control condition)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if mode == "off":
        subprocess.run(["ffmpeg", "-y", "-i", str(video_path), "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", str(out_path)], check=True, capture_output=True)
        return out_path
    work = out_path.parent / f"_{out_path.stem}_panels"
    work.mkdir(parents=True, exist_ok=True)
    for old in work.glob("*.png"):
        old.unlink()

    # One frame of panel per segment of the timeline. A segment boundary is a sound
    # starting or stopping, so the panel only changes when something is actually heard.
    debug = bool(getattr(config, "SHOW_DEBUG_SOUNDS", False) and events)
    cuts = []
    if debug:
        # every half second, so the strip's time label is never more than 0.25 s off
        # the video; it used to carry a segment's midpoint for the segment's whole length
        cuts = [k * 0.5 for k in range(int(duration * 2) + 1)]
        cuts += [e.start for e in events] + [e.end for e in events]
    segs, n_rows, placed = _timeline(specs, duration, extra_bounds=cuts)
    slot_h = panel - (DEBUG_STRIP if debug else 0)
    lines = []
    t_cursor = 0.0
    for i, (active, dur) in enumerate(segs):
        p = work / f"p{i:04d}.png"
        canvas = Image.new("RGB", (panel, panel), (16, 18, 24))
        if debug:
            _debug_strip(canvas, (0, slot_h, panel, panel), t_cursor + dur / 2,
                         events, specs, placed=placed)
        t_cursor += dur
        if n_rows == 0:
            # No augmentation anywhere in this clip is a DECISION, not a failure: say
            # so at a readable size, or a blank panel looks like a broken player. Only
            # for clips with nothing at all -- a clip that has pictures elsewhere shows
            # an empty panel in its quiet moments, because a message appearing and
            # disappearing between sounds is the flicker this layout exists to avoid.
            d = ImageDraw.Draw(canvas)
            big, small = _font(max(22, panel // 22)), _font(max(16, panel // 32))
            l1, l2 = "no off-screen sound detected", "nothing to add for this scene"
            for text, fnt, dy, col in ((l1, big, -26, (185, 192, 208)),
                                       (l2, small, 18, (120, 127, 143))):
                w = d.textlength(text, font=fnt)
                d.text(((panel - w) / 2, slot_h / 2 + dy), text, font=fnt, fill=col)
        else:
            sh = slot_h // n_rows
            for r in range(n_rows):
                y1 = slot_h if r == n_rows - 1 else (r + 1) * sh
                label, spec = active.get(r, ("", None))
                _render_slot(canvas, (0, r * sh, panel, y1), spec, label, mode)
        canvas.save(p)
        # concat resolves 'file' paths relative to concat.txt's own dir -> use basenames
        lines += [f"file '{p.name}'", f"duration {dur:.3f}"]
    lines.append(f"file 'p{len(segs)-1:04d}.png'")
    concat = work / "concat.txt"
    concat.write_text("\n".join(lines), encoding="utf-8")

    panel_mp4 = work / "panel.mp4"
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-r", str(fps), "-c:v", "libx264", "-preset", "ultrafast",
                    "-pix_fmt", "yuv420p", str(panel_mp4)],
                   check=True, capture_output=True)

    # scale original to panel height and hstack the two; keep original audio.
    # normalise fps + SAR on BOTH inputs first, else the hstack can stall/balloon
    # on a framerate/aspect mismatch (source VP8 30fps vs panel 25fps).
    fc = (f"[0:v]scale=-2:{panel},fps={fps},setsar=1[l];"
          f"[1:v]scale={panel}:{panel},fps={fps},setsar=1[r];"
          f"[l][r]hstack=inputs=2[v]")
    subprocess.run([
        "ffmpeg", "-y", "-i", str(video_path), "-i", str(panel_mp4),
        "-filter_complex", fc,
        "-map", "[v]", "-map", "0:a?", "-c:v", "libx264", "-preset", "veryfast",
        "-crf", "28", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(out_path)],
        check=True, capture_output=True)
    return out_path


# ----------------------------------------------------------------------------- Decision Inspector hooks (logging only)
def _trail_join(s, a0, b0, cur, g, gap):
    try:
        from src import trail as _T
        _T.decide("display_join", s, "pass", value="gap " + format(g, ".2f") + " s after the real end of the picture from "
                  + format(cur[1], ".2f") + " s", bar="joined if the gap <= " + format(gap, ".2f") + " s",
                  note="this burst does not start a new picture: it extends the one already on screen",
                  burst=format(a0, ".2f") + "-" + format(b0, ".2f"), joined="yes", into=format(cur[1], ".2f"))
    except Exception:
        pass


def _trail_windows(spans, dwell, after, gap):
    try:
        from src import trail as _T
        for sp in spans:
            _T.decide("display_join", sp[3], "pass", value="picture " + format(sp[1], ".2f") + "-" + format(sp[2], ".2f") + " s",
                      bar="at least " + format(dwell, ".1f") + " s on screen, at most "
                          + ("-" if after is None else format(float(after), ".1f")) + " s past the real end; repeats within "
                          + format(gap, ".2f") + " s join",
                      note="a picture starts here", picture=format(sp[1], ".2f") + "-" + format(sp[2], ".2f"),
                      burst=format(sp[1], ".2f") + "-" + format(sp[2], ".2f"))
    except Exception:
        pass


def _trail_slot(cand, limit):
    try:
        from src import trail as _T
        _r, label, a, b, spec = cand
        _T.decide("max_slots", spec, "drop", value="more than " + str(limit) + " pictures at once",
                  bar="keep by (people react to it, then confidence " + format(float(spec.confidence), ".3f") + ")",
                  note="the weakest picture at a crowded moment is not shown", picture=format(a, ".2f") + "-" + format(b, ".2f"),
                  burst=format(a, ".2f") + "-" + format(b, ".2f"))
    except Exception:
        pass
