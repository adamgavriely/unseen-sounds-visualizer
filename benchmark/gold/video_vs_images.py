"""Does the gate improve when the frames are sent as a VIDEO instead of as loose images? (2026-09-23)

Adam: "maybe it needed to see the motion itself, as in video not just images?"

He is right that we never tried. `reason._ask` builds the prompt with one `{"type": "image"}` entry
per frame, so the model receives six unrelated pictures. Qwen2.5/3-VL align their temporal position
ids with real timestamps -- but only for `{"type": "video"}` input with an fps. As it stands the
model cannot know the frames are consecutive moments of one scene, how far apart they are, or which
of them is the instant the sound began.

This re-asks the SAME question, on the SAME stretches, with the SAME model and frames, changing only
the content type. Two sets of stretches:

  the 11 blind leaks   the gate said "not visible" and the annotator said the source was plainly
                       there. Every one flipped to "visible" is a wrong picture removed.
  the DEV hits         sounds the gate correctly drew. Every one that flips is a sound LOST.

Go/no-go, fixed before the run and matching amendments 13 and 15: adopt only if at least 4 of the
11 flip to visible AND at most 1 hit is lost.

    python benchmark/gold/video_vs_images.py --half dev
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from benchmark.gold import score_per_sound as S
from benchmark.gold.error_taxonomy import bucket, GOLD, EARLY, LATE
from benchmark.gold.veto_sweep import keep

STRETCH = 5.0
NFRAMES = 6


def clip_path(stem):
    for sub in ("mixed", "seen_ambient", "unseen_ambient", "no_ambient", "unsorted", "_dropped"):
        p = _ROOT / "data" / "input" / "benchmark" / sub / f"{stem}.mp4"
        if p.exists():
            return p
    for p in (_ROOT / "data" / "input" / "gold139" / "all").glob(stem + ".*"):
        return p
    return None


def ask_video(mdl, proc, prompt, frames, fps):
    """the same question, with the frames declared as a video so temporal ids exist"""
    import torch
    content = [{"type": "video"}, {"type": "text", "text": prompt}]
    text = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                    add_generation_prompt=True)
    kw = {"text": [text], "videos": [frames], "return_tensors": "pt"}
    try:
        inputs = proc(**kw, fps=[fps])
    except TypeError:
        inputs = proc(**kw)
    inputs = inputs.to(mdl.device)
    with torch.no_grad():
        out = mdl.generate(**inputs, max_new_tokens=24, do_sample=False)
    return proc.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--half", default="dev")
    a = ap.parse_args()
    config.use_v4("59")
    from src.stage2_video_understanding import _sample_frames_at
    from src.stage5_cross_modal_analysis import reason

    gold = S.load_gold([GOLD])
    subs = S.subsets_of(gold)
    root = _ROOT / "data" / "work" / "protocol_proposed_v4b6"
    mdl, proc = reason._load(config.VLM_MODEL, "cuda")

    # the two sets of stretches, built from the logged votes
    blind, hits = [], []
    for stem in sorted(set(subs[a.half]) & set(subs["bench"])):
        f = root / stem / "gate_votes.json"
        pics = S.load_pictures(root, stem, "proposed")
        if not f.exists() or pics is None:
            continue
        V = json.loads(f.read_text(encoding="utf-8"))
        snds = gold[stem]
        kept = [p for p in pics if keep(stem, p[0], 0.3, None)]
        for lab, x, y in kept:
            right = any(S.same_family(lab, s["label"]) and S.in_window(x, s["start"], EARLY, LATE)
                        and s["needed"] and s["importance"] >= 2 for s in snds)
            rec = [v for v in V if v.get("label") == lab]
            if not rec:
                continue
            for v in rec:
                item = (stem, lab, float(v["stretch"][0]), float(v["stretch"][1]))
                if right:
                    hits.append(item)
                    break
            if right:
                continue
            bk = bucket(lab, x, y, snds)
            other = [s for s in snds if not S.same_family(lab, s["label"]) and s["start"] - 1.0 <= y and x <= s["end"] + 1.0]
            if bk == "invented" and other:
                bk = "wrong-family"
            if bk != "gate-leak":
                continue
            if max(sum(1 for k in ("name", "ab", "desc") if v.get(k) is True) for v in rec) >= 2:
                continue
            blind.append((stem, lab, float(rec[0]["stretch"][0]), float(rec[0]["stretch"][1])))

    print(f"== {a.half}: {len(blind)} blind leak stretches, {len(hits)} hit stretches\n")
    res = {"blind": [], "hits": []}
    for name, group in (("blind", blind), ("hits", hits)):
        for stem, lab, lo, hi in group:
            vp = clip_path(stem)
            if vp is None:
                continue
            times = [lo - 1.0 + (hi + 1.0 - (lo - 1.0)) * i / (NFRAMES - 1) for i in range(NFRAMES)]
            frames = _sample_frames_at(vp, times)
            if not frames:
                continue
            fps = (NFRAMES - 1) / max(0.1, (hi + 1.0) - (lo - 1.0))
            q = reason.NAME_PROMPT.format(label=lab) if hasattr(reason, "NAME_PROMPT") else \
                (f"These frames are from the moment a sound of {lab} was heard.\n"
                 f"Name the thing in these frames that is making that sound. Answer with a short "
                 f"noun phrase. If the thing making that sound is not visible in these frames, "
                 f"answer exactly: nothing.")
            try:
                as_video = ask_video(mdl, proc, q, frames, fps)
            except Exception as e:
                as_video = f"<error {type(e).__name__}: {e}>"
            as_images = reason._ask(mdl, proc, q, images=frames, max_new=24)
            flipped = (as_images.strip().lower().startswith("nothing")
                       and not as_video.strip().lower().startswith("nothing")
                       and not as_video.startswith("<error"))
            res[name].append({"clip": stem, "label": lab, "stretch": [lo, hi],
                              "as_images": as_images, "as_video": as_video, "flipped": flipped})
            print(f"   [{name:5s}] {stem[:26]:26s} {lab[:16]:16s} images={as_images[:26]!r:28s} video={as_video[:26]!r}")

    nb = sum(1 for r in res["blind"] if r["flipped"])
    nh = sum(1 for r in res["hits"] if r["flipped"])
    print(f"\nblind leaks that now name the source: {nb}/{len(res['blind'])}")
    print(f"hits that would be LOST (now named, so silenced): {nh}/{len(res['hits'])}")
    ok = nb >= 4 and nh <= 1
    print(f"\ngo/no-go (>=4 flips, <=1 hit lost): {'PASS -- worth an amendment' if ok else 'FAIL -- video encoding is not the floor'}")
    out = _ROOT / "benchmark" / "gold" / f"video_vs_images_{a.half}.json"
    out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", out)


if __name__ == "__main__":
    main()
