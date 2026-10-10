"""Video idea 3 (Adam 10 Oct: use the video more; Fable): Qwen3-Omni-30B-A3B-Instruct WATCHES a cut of the video WITH its
sound around each v1.4 picture ([start - 1, min(end, start + 4) + 1] s, use_audio_in_video) and answers yes/no; the
score is logit(yes) - logit(no) at the first answer token:
  heard    "Is a sound of {l} heard in this clip?"                                   (drop when low)
  visible  "Is the thing making the {l} sound visible on screen when it sounds?"      (drop when high)
  offscr   "The {l} sound comes from something we do not see on screen. True or false?" read as yes=true (drop when low)
Clip-grouped 5-fold CV (seed 0) on all 158 clips, by onset cost and by "no hit lost".
    GPU (env ~/venvs/qomni_av): python benchmark/gold/coverage/omni_av_veto.py --score   (omni_av_veto.json)
    CPU:                        python benchmark/gold/coverage/omni_av_veto.py           -> omni_av_veto.md"""
import json, random, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import score_per_sound as S
from benchmark.gold.v14_score import GOLD
from benchmark.gold.coverage.screen_reason import base_pics, TG

HERE = Path(__file__).resolve().parent
FEAT = HERE / "omni_av_veto.json"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
Q = {"heard": ("Is a sound of {l} heard in this clip? Answer yes or no.", +1),
     "visible": ("Is the thing making the {l} sound visible on screen when it sounds? Answer yes or no.", -1),
     "offscr": ("The {l} sound in this clip comes from something we do not see on screen. Is that true? Answer yes or no.", +1)}


def short(l):
    return l.split(",")[0].split("(")[0].strip().lower()


def score(pics):
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    yes = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text())
    out = json.loads(FEAT.read_text()) if FEAT.exists() else {}
    td = Path(tempfile.mkdtemp())
    for st, ps in pics.items():
        for lab, a, b in ps:
            k = f"{st}|{lab}|{a:.3f}"
            if k in out:
                continue
            lo = max(0.0, a - 1.0); hi = min(b, a + 4.0) + 1.0
            cut = td / "cut.mp4"
            subprocess.run([str(Path.home() / "miniconda3/envs/msproj/bin/ffmpeg"), "-y", "-loglevel", "error", "-ss", f"{lo:.2f}", "-t", f"{hi - lo:.2f}", "-i", str(TG / vids[st]),
                            "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", str(cut)], check=True)
            r = {}
            for q, (pr, _s) in Q.items():
                conv = [{"role": "user", "content": [{"type": "video", "video": str(cut)}, {"type": "text", "text": pr.format(l=short(lab))}]}]
                text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
                audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
                inp = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
                inp = inp.to(model.thinker.device).to(torch.bfloat16)
                with torch.inference_mode():
                    lg = model.thinker(**inp, use_audio_in_video=True).logits[0, -1].float()
                r[q] = float(lg[yes].max() - lg[no].max())
            out[k] = r
            print(k, r, flush=True)
            FEAT.write_text(json.dumps(out))


def main():
    gold = S.load_gold([GOLD]); pics = base_pics()
    if "--score" in sys.argv:
        score(pics); return
    F = json.loads(FEAT.read_text()); allc = list(pics); grid = [None]
    for q, (_p, sgn) in Q.items():
        vals = np.array([v[q] for v in F.values()])
        qs = np.linspace(0.02, 0.5, 13) if sgn > 0 else np.linspace(0.5, 0.98, 13)
        grid += [(q, float(x)) for x in np.unique(np.round(np.quantile(vals, qs), 3))]

    def drop(st, p, g):
        v = F.get(f"{st}|{p[0]}|{p[1]:.3f}")
        return v is not None and (v[g[0]] < g[1] if Q[g[0]][1] > 0 else v[g[0]] > g[1])
    memo = {}
    row = lambda st, g: memo.setdefault((st, g), S.score_clip(gold[st], pics[st] if g is None else [p for p in pics[st] if not drop(st, p, g)]))
    hits = lambda rr: sum(x["hit"] for x in rr)
    wr = lambda rr: sum(x["visible"] + x["cross"] + x["phantom"] for x in rr)
    cost = lambda rr: (4 * sum(x["miss"] for x in rr) + 2 * wr(rr)) / len(rr)
    r0 = [row(st, None) for st in allc]
    L = ["# Qwen3-Omni watches video + sound to drop v1.4 pictures, all 158 clips", "", f"pictures scored: {len(F)}", "",
         "| question | bar | hits | wrong | cost |", "|---|---|---|---|---|", f"| v1.4 | - | {hits(r0)} | {wr(r0)} | {cost(r0):.3f} |"]
    for g in grid[1:]:
        rr = [row(st, g) for st in allc]; L.append(f"| {g[0]} | {g[1]} | {hits(rr)} | {wr(rr)} | {cost(rr):.3f} |")
    for mode in ("cost", "no hit lost"):
        rng = random.Random(0); sh = allc[:]; rng.shuffle(sh); oof, ch = {}, []
        for k in range(5):
            te = set(sh[k::5]); tr = [c for c in allc if c not in te]
            if mode == "cost":
                best = min(grid, key=lambda g: cost([row(st, g) for st in tr]))
            else:
                h0 = hits([row(st, None) for st in tr])
                best = min([g for g in grid if hits([row(st, g) for st in tr]) >= h0], key=lambda g: wr([row(st, g) for st in tr]))
            ch.append(best)
            for st in te:
                oof[st] = row(st, best)
        rr = [oof[st] for st in allc]
        L += ["", f"CV by {mode}: choices {ch}; out of fold hits {hits(rr)}, wrong {wr(rr)}, cost {cost(rr):.3f}"]
    (HERE / "omni_av_veto.md").write_text("\n".join(L) + "\n", encoding="utf-8"); print("\n".join(L))


if __name__ == "__main__":
    main()
