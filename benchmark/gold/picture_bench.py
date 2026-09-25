"""Picture quality bench: the same 33 sounds, drawn several ways, marked by two independent models.
(2026-09-24, docs/picture_quality_prereg.md)

Adam: the pictures are "not always very nice"; try scene context; ask a model what was drawn. Three
reviewers settled the design over two rounds and the pass marks were committed before this ran.

Phases, each its own process so one large model is resident at a time:

  specs      the 33 drawn sounds of a render, with their frames' times       (CPU)
  subjects   re-run ONLY the depiction step of stage 5 with the new rules    (Qwen3.8 VLM)
  draw       draw one arm's 33 pictures, seeded, with or without the guard   (FLUX or Qwen-Image)
  eval       mark an arm's pictures: Idefics3 forced choice + CLIP-L ranking (Idefics3, CLIP-L)
  report     calibration against the hand verdicts, flip tables, the gates   (CPU)

Arms (docs/picture_quality_prereg.md): A0 shipped subjects, seeded; A1 + guard; A2 + new subjects;
A3 the A2 subjects drawn by Qwen-Image-2512. "shipped" is the render's own pictures, used to
calibrate the evaluators against the hand verdicts.

    python benchmark/gold/picture_bench.py specs
    python benchmark/gold/picture_bench.py eval --arm shipped
    python benchmark/gold/picture_bench.py draw --arm A0
    ...
    python benchmark/gold/picture_bench.py report
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import zlib
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config

TAG = "dev_symgen_v30"
BENCH = _ROOT / "data" / "work" / "picture_bench"
HAND = _ROOT / "benchmark" / "gold" / "pictures" / "hand_labels_dev_symgen_v30.json"
RESULTS = _ROOT / "benchmark" / "gold" / "pictures"

# The guard: a picture whose non-near-white pixels cover less than this share of the frame is
# treated as blank. 5% is reviewer A's number from round 1, fixed before any picture was drawn.
INK_BAR = 0.05
NEAR_WHITE = 238                   # the same cut stage 6 uses to find the subject (_subject_bbox)
N_SIBLINGS, N_OTHERS = 3, 2        # decoys: 3 from the target's own top-level branch, 2 from others
CANNOT_TELL = "cannot tell"


# ------------------------------------------------------------------------------------ helpers
def clip_path(stem):
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / sub / f"{stem}.mp4"
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "gold139" / "all").glob(stem + ".*"):
        return p
    return None


def seed_of(item) -> int:
    return zlib.crc32(f"{item['clip']}|{item['label']}|{item['start']:.2f}".encode()) & 0x7FFFFFFF


def ink(path) -> float:
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
    return float((a.min(axis=2) < NEAR_WHITE).mean())


def load_specs():
    return json.loads((BENCH / "specs.json").read_text(encoding="utf-8"))


def top_branch(label):
    from src.labels import ancestors, canonical
    anc = ancestors(canonical(label))
    return anc[-1] if anc else canonical(label)


def decoys_for(label, pool):
    """3 siblings from the target's own top-level branch, 2 from elsewhere, fixed per label.

    Siblings make the question hard; the two from other branches are what let a picture that says
    something FALSE be caught -- a pick from another branch is a false message, not a near miss.
    Anything that is the target, its ancestor or its descendant is excluded: those are the same
    sound, and offering one would mark a correct picture wrong.
    """
    from src.labels import canonical, is_descendant
    t = canonical(label)
    ok = [p for p in pool if p != t and not is_descendant(p, t) and not is_descendant(t, p)]
    rng = random.Random(zlib.crc32(t.encode()))
    same = sorted(p for p in ok if top_branch(p) == top_branch(t))
    other = sorted(p for p in ok if top_branch(p) != top_branch(t))
    pick = rng.sample(same, min(N_SIBLINGS, len(same)))
    pick += rng.sample(other, min(N_OTHERS + N_SIBLINGS - len(pick), len(other)))
    return pick


# ------------------------------------------------------------------------------------- specs
def phase_specs(a):
    from src.labels import canonical
    root = _ROOT / "data" / "work" / f"protocol_proposed_{TAG}"
    items = []
    frozen = None
    if getattr(a, "frozen", ""):                 # GP-4 confirmation: only the frozen clips and sounds
        fz = json.loads((_ROOT / a.frozen).read_text(encoding="utf-8"))
        frozen = {(s["clip"], s["label"], s["start"]) for s in fz["sounds"]}
    for f in sorted(root.glob("*/augmentations.json")):
        stem = f.parent.name
        for sp in json.loads(f.read_text(encoding="utf-8")):
            if not (sp.get("augment") and sp.get("image_path")):
                continue
            spans = sp.get("spans") or [[sp["start"], sp["end"]]]
            if frozen is not None and (stem, sp["event_label"], round(float(sp["start"]), 2)) not in frozen:
                continue
            items.append({"i": len(items), "clip": stem, "label": sp["event_label"],
                          "detail": sp.get("detail", ""), "subject": sp.get("subject", ""),
                          "start": float(sp["start"]), "end": float(sp["end"]),
                          "first_burst": [float(spans[0][0]), float(spans[0][1])],
                          "shipped_image": sp["image_path"]})
    # the decoy pool: every family the annotator used on these clips, plus the drawn labels
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.error_taxonomy import GOLD
    gold = S.load_gold([GOLD])
    pool = sorted({canonical(s["label"]) for v in gold.values() for s in v}
                  | {canonical(it["label"]) for it in items})
    for it in items:
        it["decoys"] = decoys_for(it["label"], pool)
    # join the hand verdicts by content, not by position
    hand = json.loads(HAND.read_text(encoding="utf-8"))["pictures"] if TAG == "dev_symgen_v30" else []
    key = {(h["clip"], h["label"], h["subject"]): h for h in hand}
    for it in items:
        h = key.get((it["clip"], it["label"], it["subject"]))
        it["hand"] = h["verdict"] if h else None
        it["hand_readable"] = h["readable_as_the_sound"] if h else None
    if frozen is not None:
        assert len(items) == len(frozen), f"{len(items)} specs vs {len(frozen)} frozen sounds"
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "specs.json").write_text(json.dumps(items, indent=1), encoding="utf-8")
    print(f"{len(items)} drawn sounds; {sum(1 for i in items if i['hand'])} joined to a hand verdict; "
          f"decoy pool {len(pool)} families")


# ---------------------------------------------------------------------------------- subjects
def phase_subjects_v3(a):
    """PICTURE_V3 subjects for the bench's sounds: the specific source from the render's own raw
    detector events (labels.choose_source on the drawn burst), then reason._depict_v3. The gate is not
    re-run, so both arms draw exactly the same sounds."""
    config.use_v4("590")
    config.DEVICE = "cuda"
    config.PICTURE_V3 = True
    from src.types import AudioEvent
    from src.labels import choose_source, canonical
    from src.stage2_video_understanding import _sample_frames
    from src.stage5_cross_modal_analysis import reason as R
    items = load_specs()
    work = _ROOT / "data" / "work" / f"protocol_proposed_{TAG}"
    mdl, proc = R._load(config.VLM_MODEL, "cuda")
    out, places = {}, {}
    for it in items:
        ev = [AudioEvent(e["label"], e["start"], e["end"], e["confidence"])
              for e in json.loads((work / it["clip"] / "events.json").read_text(encoding="utf-8"))]
        src, cands = choose_source(ev, canonical(it["label"]), (it["start"], it["end"]), explain=True)
        if it["clip"] not in places:            # only for the strip backstops; v3 never sees it
            frames = _sample_frames(clip_path(it["clip"]), 4)
            places[it["clip"]] = R._clean_phrase(R._ask(mdl, proc, R.PLACE_PROMPT, images=frames,
                                                        max_new=16), max_words=4) or "an unknown place"

        class _Spec:                            # the fields _depict_v3 reads
            event_label, detail, source = it["label"], it["detail"], src
        phrase = R._depict_v3(_Spec, places[it["clip"]], mdl, proc) or (src + " making its sound")
        out[str(it["i"])] = {"subject": phrase, "source": src, "place": places[it["clip"]],
                             "candidates": cands}
        print(f"   {it['i']:2d} {it['label'][:14]:14s} [{src[:24]:24s}] {it['subject'][:28]:28s} -> {phrase}",
              flush=True)
    (BENCH / "subjects_V3.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", BENCH / "subjects_V3.json")


def phase_subjects_v31(a):
    """V3.1 (PICTURE_SCENE) subjects: V3's source, then RESOLVE on six frames over the drawn burst (the
    burst the source was chosen on, P1 round 2) with the clip's place, then the V3.1 depiction and the
    list guard. `fired` = every raw firing of the family in the burst at the detector's bar."""
    config.use_v4("590")
    config.DEVICE = "cuda"
    config.PICTURE_V3 = True
    config.PICTURE_SCENE = True
    name = a.arm or "V31"
    config.PICTURE_SCENE_GUARD2 = name == "V31G"      # GP-4 3(b): the other-branch / head-word guard
    from src.types import AudioEvent
    from src.labels import choose_source, canonical
    from src.stage2_video_understanding import _sample_frames, _sample_frames_at
    from src.stage5_cross_modal_analysis import reason as R
    items = load_specs()
    work = _ROOT / "data" / "work" / f"protocol_proposed_{TAG}"
    mdl, proc = R._load(config.VLM_MODEL, "cuda")
    out, places = {}, {}
    bar = float(getattr(config, "AED_THRESHOLD", 0.175))
    for it in items:
        ev = [AudioEvent(e["label"], e["start"], e["end"], e["confidence"])
              for e in json.loads((work / it["clip"] / "events.json").read_text(encoding="utf-8"))]
        fam = canonical(it["label"])
        a0, b0 = it["start"], it["end"]
        src, cands = choose_source(ev, fam, (a0, b0), explain=True)
        fired = sorted({e.label for e in ev if canonical(e.label) == fam and e.start <= b0 and e.end >= a0
                        and e.confidence >= bar})
        vp = clip_path(it["clip"])
        if it["clip"] not in places:
            places[it["clip"]] = R._clean_phrase(R._ask(mdl, proc, R.PLACE_PROMPT, images=_sample_frames(vp, 4),
                                                        max_new=16), max_words=4) or "an unknown place"
        lo, hi = a0 - 1.0, min(b0, a0 + 5.0) + 1.0
        frames = _sample_frames_at(vp, [lo + (hi - lo) * k / 5 for k in range(6)])

        class _Spec:
            event_label, detail, source = it["label"], it["detail"], src
        phrase = R._depict_v31(_Spec, places[it["clip"]], frames, fired, mdl, proc)
        out[str(it["i"])] = {"subject": phrase, "source": src, "place": places[it["clip"]],
                             "candidates": cands, "fired": fired}
        print(f"   {it['i']:2d} {it['label'][:14]:14s} [{src[:24]:24s}] {it['subject'][:28]:28s} -> {phrase}",
              flush=True)
    (BENCH / f"subjects_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", BENCH / f"subjects_{name}.json")


def phase_expand(a):
    """GP-4 3(b): the text-only two-step expansion of an existing subjects file (--arm names it, e.g. V31)
    into subjects_<arm>X.json, with the refused words and the fallback rate logged."""
    config.use_v4("590")
    config.DEVICE = "cuda"
    from src.labels import canonical
    from src.stage5_cross_modal_analysis import reason as R
    items = load_specs()
    subj = json.loads((BENCH / f"subjects_{a.arm}.json").read_text(encoding="utf-8"))
    mdl, proc = R._load(config.VLM_MODEL, "cuda")
    out, fell = {}, 0
    for it in items:
        s = dict(subj.get(str(it["i"]), {}))
        subject = s.get("subject") or it["subject"]
        source = s.get("source") or canonical(it["label"])
        long, bad = R.expand_prompt(subject, source, canonical(it["label"]), mdl, proc)
        fell += not long
        s.update({"subject": subject, "long": long, "refused": bad})
        out[str(it["i"])] = s
        print(f"   {it['i']:2d} {'FALLBACK ' + ','.join(bad) if not long else 'ok'} | {subject} -> {long[:90]}",
              flush=True)
    (BENCH / f"subjects_{a.arm}X.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> subjects_{a.arm}X.json; fallback {fell}/{len(items)}")


def phase_subjects(a):
    if a.arm == "V3":
        return phase_subjects_v3(a)
    if a.arm in ("V31", "V31G"):
        return phase_subjects_v31(a)
    """Re-run ONLY the depiction step of stage 5, as reason.decide_subjects does it, with the new
    rules switched on. The gate is not re-run: every arm draws the same 33 sounds."""
    config.use_v4("590")
    config.DEVICE = "cuda"
    config.KIND_ALWAYS = True
    config.DEPICT_V2 = True
    from src.stage2_video_understanding import _sample_frames, _sample_frames_at
    from src.stage5_cross_modal_analysis import reason as R
    items = load_specs()
    mdl, proc = R._load(config.VLM_MODEL, "cuda")
    places, out = {}, {}
    by_clip = {}
    for it in items:
        by_clip.setdefault(it["clip"], []).append(it)
    for stem, its in by_clip.items():
        vp = clip_path(stem)
        frames = _sample_frames(vp, 4)
        scene = " ".join(R._ask(mdl, proc, R.SCENE_PROMPT, images=frames, max_new=40).split())[:160] \
            or "an unknown place"
        place = R._clean_phrase(R._ask(mdl, proc, R.PLACE_PROMPT, images=frames, max_new=16),
                                max_words=4) or scene
        places[stem] = place
        labels = [x["label"] for x in its]
        for it in its:
            # the frames decide_subjects keeps for this sound: the first stretch of its first
            # burst, a second either side, six frames
            a0, b0 = it["first_burst"]
            k = max(1, int(round((b0 - a0) / float(getattr(config, "VISIBILITY_STRETCH", 5.0)))))
            n = 6
            lo, hi = a0 - 1.0, a0 + (b0 - a0) / k + 1.0
            win = _sample_frames_at(vp, [lo + (hi - lo) * t / (n - 1) for t in range(n)])
            # --- the depiction step, line for line as in reason.decide_subjects (step 2)
            detail, kind = "", ""
            if it["detail"] and it["detail"] != it["label"]:
                detail = " (specifically: " + it["detail"].split(",")[0].split("(")[0].strip() + ")"
                kind = R._kind_from_frames(it["label"], win, mdl, proc)
                if kind:
                    detail = detail[:-1] + "; the frames suggest: " + kind + ")"
            else:
                kind = R._kind_from_frames(it["label"], win, mdl, proc)
                if kind:
                    detail = " (specifically: " + kind + ")"
            prompt = R.DEPICT_PROMPT_V2.format(label=it["label"], detail=detail, scene=place)
            phrase = R._without_place(R._clean_phrase(R._ask(mdl, proc, prompt, max_new=48)),
                                      place, keep=kind)
            if a.arm == "A2b":
                phrase = R._drop_place_phrase(phrase, place, it["label"], it["detail"])
            if phrase and not R._still_the_sound(phrase, it["label"], labels, mdl, proc):
                phrase = R._without_place(R._clean_phrase(R._ask(mdl, proc, R.RETRY_PROMPT.format(
                    label=it["label"], detail=detail, scene=place), max_new=48)), place, keep=kind)
            if not phrase:
                phrase = (it["detail"].split(",")[0] if it["detail"] else it["label"]) + " happening"
            out[str(it["i"])] = {"subject": phrase, "kind": kind, "place": place}
            print(f"   {it['i']:2d} {it['label'][:14]:14s} {it['subject'][:28]:28s} -> {phrase:34s} "
                  f"[kind: {kind or '-'}]", flush=True)
    name = a.arm or "A2"
    (BENCH / f"subjects_{name}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print("->", BENCH / f"subjects_{name}.json")


# -------------------------------------------------------------------------------------- draw
ARMS = {
    "A0": {"subjects": "shipped", "guard": False, "gen": "flux"},
    "A1": {"subjects": "shipped", "guard": True, "gen": "flux"},
    "A2": {"subjects": "A2", "guard": True, "gen": "flux"},
    "A3": {"subjects": "A2", "guard": True, "gen": "qwen"},
    "A3b": {"subjects": "A2b", "guard": True, "gen": "qwen"},
    "N": {"subjects": "V3", "guard": True, "gen": "qwen", "negative": True},
    "N0": {"subjects": "shipped", "guard": True, "gen": "qwen", "negative": True},
    "N1": {"subjects": "V31", "guard": True, "gen": "qwen", "negative": True},
}
# Reviewer A, round 4: a full-frame picture breaks the white-background contract the other way (a
# dark sky for thunder is a scene, not an object). Added AFTER seeing A3, and reported as such.
INK_MAX = 0.85


def arm_spec(name):
    base, _, s = name.partition("_s")
    spec = dict(ARMS[base])
    spec["seed_offset"] = 1000 * int(s) if s else 0
    spec["upper"] = base == "A3b"
    spec.setdefault("negative", False)
    return spec
GEN = {"flux": ("black-forest-labs/FLUX.1-schnell", (768, 768)),
       "qwen": ("Qwen/Qwen-Image-2512", (1024, 1024))}


def phase_draw(a):
    from src.stage6_visual_augmentation import _diffusion_image, plain_prompt, negative_for
    arm = arm_spec(a.arm)
    model, size = GEN[arm["gen"]]
    items = load_specs()
    subj = {}
    if arm["subjects"] != "shipped":
        subj = json.loads((BENCH / f"subjects_{arm['subjects']}.json").read_text(encoding="utf-8"))
    out_dir = BENCH / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for it in items:
        subject = subj.get(str(it["i"]), {}).get("subject", it["subject"])
        prompt = plain_prompt(subject)
        p = out_dir / f"{it['i']:02d}.png"
        seed = seed_of(it) + arm["seed_offset"]
        neg = negative_for(subject) if arm["negative"] else None
        ok = _diffusion_image(p, prompt, size, model=model, device="cuda", seed=seed, negative=neg)
        fired, dropped = 0, False
        bad = (lambda q: ink(q) < INK_BAR or (arm["upper"] and ink(q) > INK_MAX))
        if ok and arm["guard"]:
            while bad(p) and fired < 2:
                fired += 1
                _diffusion_image(p, prompt, size, model=model, device="cuda", seed=seed + fired,
                                 negative=neg)
            if bad(p):
                dropped = True            # counted as a failure, never shown as a blank
        manifest.append({"i": it["i"], "subject": subject, "prompt": prompt, "seed": seed,
                         "ink": round(ink(p), 4) if ok else 0.0, "guard_fired": fired,
                         "dropped": dropped, "ok": bool(ok)})
        print(f"   {a.arm} {it['i']:2d} ink {manifest[-1]['ink']:.3f} guard {fired} "
              f"{'DROPPED ' if dropped else ''}{subject}", flush=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"{a.arm}: {sum(m['guard_fired'] > 0 for m in manifest)} guard firings, "
          f"{sum(m['dropped'] for m in manifest)} dropped")


# -------------------------------------------------------------------------------------- eval
def picture_paths(arm, items):
    if arm == "today":
        return {it["i"]: Path(it["shipped_image"]) for it in items}
    if arm == "shipped":
        # the render's own pictures, copied next to the bench by specs time on the cluster
        return {it["i"]: Path(it["shipped_image"]) for it in items}
    return {it["i"]: BENCH / arm / f"{it['i']:02d}.png" for it in items}


def phase_eval(a):
    import torch
    from PIL import Image
    items = load_specs()
    paths = picture_paths(a.arm, items)
    res = {str(it["i"]): {} for it in items}

    # --- CLIP-L: rank the target among its decoys (contrastive, no "cannot tell")
    from transformers import CLIPModel, CLIPProcessor
    name = "openai/clip-vit-large-patch14"
    try:
        cm = CLIPModel.from_pretrained(name).to("cuda").eval()
        cp = CLIPProcessor.from_pretrained(name)
    except Exception as e:
        print(f"CLIP-L unavailable ({type(e).__name__}); falling back to clip-vit-base-patch32")
        name = "openai/clip-vit-base-patch32"
        cm = CLIPModel.from_pretrained(name).to("cuda").eval()
        cp = CLIPProcessor.from_pretrained(name)
    for it in items:
        p = paths[it["i"]]
        if not p.exists():
            res[str(it["i"])]["clip"] = None
            continue
        opts = [it["label"]] + it["decoys"]
        inp = cp(text=[f"a picture of {o}" for o in opts], images=Image.open(p).convert("RGB"),
                 return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            logits = cm(**inp).logits_per_image[0].float().cpu().numpy()
        res[str(it["i"])]["clip"] = opts[int(np.argmax(logits))]
    res["_clip_model"] = name
    del cm
    torch.cuda.empty_cache()

    # --- Idefics3: forced choice, target + decoys + "cannot tell", shuffled with a fixed seed
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _Auto
    except ImportError:
        from transformers import AutoModelForVision2Seq as _Auto
    mid = "HuggingFaceM4/Idefics3-8B-Llama3"
    ip = AutoProcessor.from_pretrained(mid)
    im = _Auto.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()
    for it in items:
        p = paths[it["i"]]
        if not p.exists():
            res[str(it["i"])]["idefics"] = None
            continue
        opts = [it["label"]] + it["decoys"]
        random.Random(seed_of(it)).shuffle(opts)
        opts.append(CANNOT_TELL)
        listing = "\n".join(f"{k + 1}. {o}" for k, o in enumerate(opts))
        q = ("This picture is shown beside a video to tell a deaf viewer about a sound they cannot "
             "hear. Which ONE of these sounds does the picture show?\n" + listing +
             "\nAnswer with the number only.")
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": q}]}]
        text = ip.apply_chat_template(msgs, add_generation_prompt=True)
        inp = ip(text=text, images=[Image.open(p).convert("RGB")], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = im.generate(**inp, max_new_tokens=4, do_sample=False)
        ans = ip.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        digits = "".join(ch for ch in ans if ch.isdigit())
        k = int(digits) - 1 if digits else -1
        res[str(it["i"])]["idefics"] = opts[k] if 0 <= k < len(opts) else CANNOT_TELL
        res[str(it["i"])]["idefics_raw"] = ans
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"eval_{a.arm}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    right = lambda k: sum(1 for it in items if res[str(it["i"])].get(k) == it["label"])
    print(f"{a.arm}: Idefics3 right {right('idefics')}/{len(items)}, {name.split('/')[-1]} right "
          f"{right('clip')}/{len(items)}")


# ------------------------------------------------------------------------------------- eval2
# The single declared redesign (round 3, docs/picture_quality_prereg.md, amendment). The forced
# choice failed calibration (25/32 both) because a wrong picture wins whenever the target is the
# least-bad option, and because offering the word lets a model make the pun no viewer makes. So:
# an OPEN question with no sound named, in two lines, and the answer matched as TEXT by a sentence
# embedder that is in neither the pipeline nor a judge row. Framing is not a language question and
# goes to a no-model rule. Nothing below is tuned after seeing a number; there is no second redesign.
EVAL2_PROMPT = ("This picture is shown to a viewer who cannot hear. Answer in exactly two lines."
                + chr(10) +
                "OBJECT: the main thing actually drawn, at most 4 words. Do not guess at anything "
                "that is not visible. If the picture is blank, answer: nothing."
                + chr(10) +
                "SOUND: the sound it makes, at most 6 words. If nothing in it makes a sound, "
                "answer: nothing.")
MATCHER = "sentence-transformers/all-MiniLM-L6-v2"
FRAME_MIN_HEIGHT = 0.25     # reviewer A: a subject shorter than a quarter of the frame is not a glance


def framing_ok(path) -> bool:
    from PIL import Image
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8).min(axis=2) < NEAR_WHITE
    rows = np.where(a.any(axis=1))[0]
    if not len(rows):
        return False
    return bool((rows[-1] - rows[0] + 1) / a.shape[0] >= FRAME_MIN_HEIGHT)


def _two_lines(text):
    obj, snd = "", ""
    # GLM-4.6V wraps its answer in box tokens: "<|begin_of_box|>OBJECT: cannon ...<|end_of_box|>"
    text = text.replace("<|begin_of_box|>", "").replace("<|end_of_box|>", "")
    for line in text.splitlines():
        clean = line.strip().lstrip("*-#• ").replace("**", "")   # GLM answers in markdown (P3, round 3)
        low = clean.lower()
        if low.startswith("object:"):
            obj = clean.split(":", 1)[1].strip()
        elif low.startswith("sound:"):
            snd = clean.split(":", 1)[1].strip()
    return obj, snd


def phase_eval2(a):
    import torch
    from PIL import Image
    from sentence_transformers import SentenceTransformer
    items = load_specs()
    paths = picture_paths(a.arm, items)
    from transformers import AutoProcessor
    try:
        from transformers import AutoModelForImageTextToText as _Auto
    except ImportError:
        from transformers import AutoModelForVision2Seq as _Auto
    mid = "HuggingFaceM4/Idefics3-8B-Llama3"
    ip = AutoProcessor.from_pretrained(mid)
    im = _Auto.from_pretrained(mid, dtype=torch.bfloat16).to("cuda").eval()
    st = SentenceTransformer(MATCHER, device="cuda")
    res = {}
    for it in items:
        p = paths[it["i"]]
        if not p.exists():
            res[str(it["i"])] = {"pass": False, "why": "no picture"}
            continue
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": EVAL2_PROMPT}]}]
        text = ip.apply_chat_template(msgs, add_generation_prompt=True)
        inp = ip(text=text, images=[Image.open(p).convert("RGB")], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = im.generate(**inp, max_new_tokens=40, do_sample=False)
        ans = ip.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
        obj, snd = _two_lines(ans)
        opts = [it["label"]] + it["decoys"] + ["nothing"]
        e = st.encode([snd or "nothing"] + [f"the sound of {o}" if o != "nothing" else "nothing"
                                            for o in opts], normalize_embeddings=True)
        sims = e[1:] @ e[0]
        pick = opts[int(np.argmax(sims))]
        blank = bool(ink(p) < INK_BAR)
        framed = bool(framing_ok(p))
        empty = (not obj) or obj.strip().lower().startswith("nothing")
        ok = pick == it["label"] and not empty and not blank and framed
        why = ("blank" if blank else "subject too small" if not framed else "no object named" if empty
               else "" if pick == it["label"] else f"sound read as {pick}")
        res[str(it["i"])] = {"pass": bool(ok), "object": obj, "sound": snd, "pick": pick,
                             "blank": blank, "framed": framed, "why": why, "raw": ans}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"eval2_{a.arm}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"{a.arm}: eval2 passes {sum(r['pass'] for r in res.values())}/{len(items)}")
    ref = [it for it in items if it["hand_readable"] is not None]
    if a.arm == "shipped":
        agree = sum(1 for it in ref if res[str(it["i"])]["pass"] == it["hand_readable"])
        miss = [f"{it['i']}:{it['hand']}({res[str(it['i'])]['why'] or 'pass'})" for it in ref
                if res[str(it["i"])]["pass"] != it["hand_readable"]]
        print(f"CALIBRATION (development set -- it has now seen these labels twice): "
              f"{agree}/{len(ref)}  disagreements: {', '.join(miss)}")


# ------------------------------------------------------------------------------------- check
CHECK_MODEL = "zai-org/GLM-4.6V-Flash"      # a model family used nowhere else in the pipeline
CHECK_PROMPT = ("Look at this picture. Answer in exactly two lines."
                + chr(10) + "OBJECT: the main thing actually drawn, at most 4 words. If the picture is "
                "blank, answer: nothing."
                + chr(10) + "SOUND: the sound it is making, at most 6 words. If nothing in it makes a "
                "sound, answer: nothing.")


def phase_check(a):
    """Adam's "ask what is seen" step, run as a logged verdict column. It is NOT told the sound or the
    subject, and it changes nothing: the panel ruled it may not reject or redraw anything until it has
    passed calibration against Adam's own ratings."""
    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    items = load_specs()
    proc = AutoProcessor.from_pretrained(CHECK_MODEL)
    mdl = AutoModelForImageTextToText.from_pretrained(CHECK_MODEL, dtype=torch.bfloat16).to("cuda").eval()
    res = {}
    for arm in a.arm.split(","):
        paths = picture_paths(arm, items)
        for it in items:
            p = paths[it["i"]]
            if not p.exists():
                continue
            msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": CHECK_PROMPT}]}]
            try:
                text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False,
                                                enable_thinking=False)
            except TypeError:
                text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
            inp = proc(text=[text], images=[Image.open(p).convert("RGB")], return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = mdl.generate(**inp, max_new_tokens=160, do_sample=False)
            raw = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            raw = raw.split("</" + "think>")[-1].replace("<" + "answer>", "").replace("</" + "answer>", "")
            obj, snd = _two_lines(raw)
            res[f"{arm}:{it['i']}"] = {"object": obj, "sound": snd, "raw": raw.strip()[:300]}
            print(f"   {arm} {it['i']:2d} {it['label'][:14]:14s} OBJECT {obj[:30]:30s} SOUND {snd[:34]}",
                  flush=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"check_{TAG}__{a.arm.replace(',', '+')}.json").write_text(json.dumps(res, indent=1),
                                                                        encoding="utf-8")


# ------------------------------------------------------------------------------------ report
def phase_report(a):
    items = load_specs()
    have = lambda arm: (RESULTS / f"eval_{arm}.json").exists()

    def correct(arm):
        r = json.loads((RESULTS / f"eval_{arm}.json").read_text(encoding="utf-8"))
        return {it["i"]: {k: (r[str(it["i"])].get(k) == it["label"]) for k in ("idefics", "clip")}
                for it in items}, r

    print("CALIBRATION against the hand verdicts (32 pictures; pass = at least 26 agree)")
    votes = {}
    if have("shipped"):
        c, _ = correct("shipped")
        ref = [it for it in items if it["hand_readable"] is not None]
        for k in ("idefics", "clip"):
            agree = sum(1 for it in ref if c[it["i"]][k] == it["hand_readable"])
            votes[k] = agree >= 26
            miss = [f"{it['i']}:{it['hand']}" for it in ref if c[it['i']][k] != it["hand_readable"]]
            print(f"   {k:8s} agrees on {agree}/{len(ref)}  -> {'VOTES' if votes[k] else 'does NOT vote'}"
                  f"   disagreements: {', '.join(miss)}")
    else:
        print("   (no calibration yet)")

    print("\nARMS, each against the one before it")
    order = ["A0", "A1", "A2", "A3"]
    prev = "shipped"
    from src.labels import canonical
    for arm in order:
        if not have(arm):
            print(f"   {arm}: not evaluated")
            continue
        c, r = correct(arm)
        cp, _ = correct(prev) if have(prev) else (None, None)
        man = BENCH / arm / "manifest.json"
        m = json.loads(man.read_text(encoding="utf-8")) if man.exists() else []
        blank = sum(1 for x in m if x.get("ink", 1) < INK_BAR and not x.get("dropped"))
        dropped = sum(1 for x in m if x.get("dropped"))
        fired = sum(1 for x in m if x.get("guard_fired"))
        line = f"   {arm} vs {prev}:  blank {blank}  dropped {dropped}  guard fired {fired}"
        for k in ("idefics", "clip"):
            n = sum(c[i][k] for i in c)
            if cp:
                fixed = sum(1 for i in c if c[i][k] and not cp[i][k])
                broke = sum(1 for i in c if cp[i][k] and not c[i][k])
                line += f" | {k} {n}/{len(c)} fixed {fixed} broke {broke} net {fixed - broke:+d}"
            else:
                line += f" | {k} {n}/{len(c)}"
        print(line)
        # gate 4: a pick from a different top-level branch is a possible false message
        flags = []
        for it in items:
            for k in ("idefics", "clip"):
                pick = r[str(it["i"])].get(k)
                if pick and pick not in (it["label"], CANNOT_TELL) and top_branch(pick) != top_branch(it["label"]):
                    flags.append(f"{it['i']}:{k}->{pick}")
        if flags:
            print(f"        look at by eye (picked another branch): {', '.join(flags)}")
        disagree = [str(it["i"]) for it in items if c[it["i"]]["idefics"] != c[it["i"]]["clip"]]
        print(f"        evaluators disagree on: {', '.join(disagree) or 'none'}")
        worse = [str(it["i"]) for it in items if it["hand"] == "good" and cp
                 and any(cp[it["i"]][k] and not c[it["i"]][k] for k in ("idefics", "clip"))]
        print(f"        hand-good pictures that got worse: {', '.join(worse) or 'none'}")
        prev = arm


def main():
    global TAG, BENCH
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["specs", "subjects", "expand", "draw", "eval", "eval2", "check", "report"])
    ap.add_argument("--arm", default="")
    ap.add_argument("--tag", default=TAG, help="the render whose drawn sounds are the bench")
    ap.add_argument("--bench", default="", help="bench folder (default: data/work/picture_bench)")
    ap.add_argument("--frozen", default="", help="frozen confirmation set json (GP-4): restrict specs to it")
    a = ap.parse_args()
    TAG = a.tag
    if a.bench:
        BENCH = Path(a.bench) if Path(a.bench).is_absolute() else _ROOT / a.bench
    {"specs": phase_specs, "subjects": phase_subjects, "draw": phase_draw,
     "eval": phase_eval, "eval2": phase_eval2, "check": phase_check, "report": phase_report,
     "expand": phase_expand}[a.phase](a)


if __name__ == "__main__":
    main()
