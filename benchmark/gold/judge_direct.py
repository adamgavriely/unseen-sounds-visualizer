"""The judge, with no describer and no model-written reference (night panel 2026-09-25, rounds 1-2).

Until now the judge chain was: Qwen3.8 describes the panel -> Gemma compares that sentence with a
reference -> a rubric cap. Two links in it were circular. The describer was the same weights that wrote
the picture's subject, so it read the picture with the pipeline's own eyes; and the reference was
written by that model's text side from the DETECTOR's events, so a phantom the detector heard became
something the viewer "needed". Both links are removed:

  * the reference is a fixed template over the annotator's own ticks (`score_per_sound.load_gold`): the
    needed sounds with their times, or "nothing beyond the picture" when the clip has none;
  * Gemma-4-31B (multimodal) looks at the pictures the viewer was shown, with the seconds each was on
    screen, and scores them against that reference. The caption arm is shown its text tags.

Nothing in the chain is a pipeline model, and every input except the pictures is the annotator's -- the
same property the headline metric has. The trust checks (docs/judge_plan.md B1, B2) are rerun on this
chain before any ranking is shown, with `judge_trust.py --results <this output>`; `--repeat` re-judges a
fixed subset with sampling on, for the self-agreement check (B3).

    python benchmark/gold/judge_direct.py --tag v4b4 --systems proposed,blind_a2i,audio_caption \
        --out benchmark/judge_direct_v4b4.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S          # noqa: E402
from benchmark.gold.error_taxonomy import GOLD          # noqa: E402

JUDGE = "google/gemma-4-31B-it"
NOTHING = "nothing beyond the picture"
CAP = 2
MAX_PICTURES = 4

PROMPT = (
    "You are checking an aid for a deaf viewer. Beside a video, the aid shows a picture (or a short "
    "text tag) while a sound happens that the viewer cannot see on screen."
    + chr(10) + chr(10) +
    "WHAT THE VIEWER NEEDED, according to a human annotator who watched and heard the clip:"
    + chr(10) + "{reference}"
    + chr(10) + chr(10) +
    "WHAT THE AID SHOWED:" + chr(10) + "{shown}"
    + chr(10) + chr(10) +
    "Score how well what was shown gives the viewer what they needed, without telling them anything "
    "false. Judge each picture by what it actually shows, not by what it may have been meant to show."
    + chr(10) +
    "  4 = every needed sound is shown recognisably, near the right time, and nothing false or extra"
    + chr(10) +
    "  3 = most of it, a minor omission or a vague picture"
    + chr(10) +
    "  2 = partly: a needed sound is missing or unrecognisable, or an extra picture shows something "
    "the viewer did not need"
    + chr(10) + "  1 = barely related" + chr(10) + "  0 = unrelated or misleading"
    + chr(10) +
    "If nothing was needed: 4 when nothing was shown, at most 2 when something was."
    + chr(10) +
    'Answer as JSON only: {{"score": <0-4>, "why": "<one short sentence>"}}'
)


def reference(snds) -> str:
    need = sorted((s for s in snds if s["needed"] and s["importance"] >= 2), key=lambda s: s["start"])
    if not need:
        return NOTHING
    return "; ".join(f"{s['label']} (from {s['start']:.1f} s to {s['end']:.1f} s)" for s in need)


def shown_items(work: Path, stem: str, system: str):
    """what the viewer saw, as the scorer reconstructs it: [(label, a, b, image path or None)]"""
    f = work / stem / "augmentations.json"
    if not f.exists():
        return None
    specs = json.loads(f.read_text(encoding="utf-8"))
    spans = S.load_pictures(work, stem, system) or []
    by_label = {}
    for s in specs:
        if s.get("augment") and s.get("image_path"):
            by_label.setdefault(s["event_label"], s["image_path"])
    out = []
    for lab, a, b in spans:
        img = None
        if system != "audio_caption":
            p = by_label.get(lab)
            if p:
                p = Path(p)
                if not p.is_absolute():
                    p = _ROOT / p
                img = p if p.exists() else None
        out.append((lab, float(a), float(b), img))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--systems", default="proposed,blind_a2i,audio_caption")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", default="", help="optional file with one clip stem per line")
    ap.add_argument("--repeat", type=int, default=0, help="B3: re-judge the first N clips with sampling on")
    a = ap.parse_args()

    import torch
    from PIL import Image
    from transformers import AutoProcessor, AutoModelForImageTextToText
    proc = AutoProcessor.from_pretrained(JUDGE)
    mdl = AutoModelForImageTextToText.from_pretrained(JUDGE, dtype=torch.bfloat16, device_map="auto").eval()

    gold = S.load_gold([GOLD])
    stems = sorted(gold)
    if a.clips:
        want = {l.strip() for l in Path(a.clips).read_text(encoding="utf-8").splitlines() if l.strip()}
        stems = [s for s in stems if s in want]
    out_f = Path(a.out)
    rows = json.loads(out_f.read_text(encoding="utf-8")) if out_f.exists() else []
    done = {(r["clip"], r["system"], r.get("repeat", 0)) for r in rows}
    torch.manual_seed(1)
    for system in a.systems.split(","):
        work = _ROOT / "data" / "work" / f"protocol_{system}_{a.tag}"
        todo = stems[:a.repeat] if a.repeat else stems
        for stem in todo:
            key = (stem, system, 1 if a.repeat else 0)
            if key in done:
                continue
            items = shown_items(work, stem, system)
            if items is None:
                continue
            ref = reference(gold[stem])
            content, lines, imgs = [], [], []
            for k, (lab, x, y, img) in enumerate(items, 1):
                if system == "audio_caption":
                    lines.append(f'text tag "{lab}", on screen from {x:.1f} s to {y:.1f} s')
                elif img is not None and len(imgs) < MAX_PICTURES:
                    imgs.append(Image.open(img).convert("RGB").resize((512, 512)))
                    lines.append(f"picture {len(imgs)} (attached), on screen from {x:.1f} s to {y:.1f} s")
            shown = chr(10).join(lines) if lines else "nothing"
            for im in imgs:
                content.append({"type": "image", "image": im})
            content.append({"type": "text", "text": PROMPT.format(reference=ref, shown=shown)})
            msgs = [{"role": "user", "content": content}]
            inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                           return_dict=True, return_tensors="pt").to(mdl.device)
            kw = {"do_sample": True, "temperature": 1.0} if a.repeat else {"do_sample": False}
            with torch.no_grad():
                gen = mdl.generate(**inp, max_new_tokens=120, **kw)
            raw = proc.decode(gen[0, inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            m = re.search(r'"score"\s*:\s*([0-4])', raw)
            w = re.search(r'"why"\s*:\s*"([^"]*)', raw)
            score = int(m.group(1)) if m else None
            row = {"clip": stem, "system": system, "reference": ref, "shown": shown,
                   "n_augmentations": len(lines), "score_permissive": score, "score": score,
                   "why": w.group(1) if w else "UNPARSED: " + raw[:200], "judge_model": JUDGE + " (direct)",
                   "repeat": 1 if a.repeat else 0}
            # the rubric cap of judge v4, applied in code exactly as before (scripts/rubric_enforce.py)
            if score is not None and ref == NOTHING and lines and score > CAP:
                row["score"] = CAP
            rows.append(row)
            out_f.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
            print(f"  {system:13s} {stem[:34]:34s} {score}  {row['why'][:60]}", flush=True)
    bad = sum(1 for r in rows if r["score"] is None)
    print(f"{len(rows)} rows, {bad} unparsed -> {out_f}")


if __name__ == "__main__":
    main()
