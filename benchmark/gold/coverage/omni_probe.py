"""Step 3 prep (PREREG_step3_omni_probe.md): two Qwen3-Omni probes on DEV. Measure only.

    python benchmark/gold/coverage/omni_probe.py items    # local: omni_items_dev.json (no gold read)
    python benchmark/gold/coverage/omni_probe.py run      # cluster GPU (msproj + ~/venvs/qomni_av): omni_probe_dev.json
    python benchmark/gold/coverage/omni_probe.py report   # local: truth from gold, AUROCs -> omni_probe_dev.md
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
HERE = Path(__file__).resolve().parent
ITEMS, OUT = HERE / "omni_items_dev.json", HERE / "omni_probe_dev.json"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SR = 16000
QA = ("You are watching a short video with its sound. A sound of {label} is heard in it. Is the thing making this sound "
      "visible on screen, and can you see it making the sound? Answer yes or no.")
QB = "Which of these sounds can you hear in this recording? Answer with every letter that applies, separated by commas."
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# ----------------------------------------------------------------------------- items (local)
def items():
    from src.labels import canonical, is_music
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    dev = {c["clip"]: c for c in dj["clips"] if c["split"] == "DEV"}
    pf = json.loads((HERE / "pics_frozen.json").read_text(encoding="utf-8"))["clips"]
    gd = json.loads((HERE / "gate_dev.json").read_text(encoding="utf-8"))
    dur = {st: float(pf[st]["dur"]) for st in dev}
    A = []
    for st, r in gd.items():
        for gi, g in enumerate(r["gate"]):
            t = float(g["start"])
            while t < float(g["end"]) - 1e-6:
                A.append({"id": f"a{len(A)}", "clip": st, "gate": gi, "label": g["label"], "t": round(t, 2),
                          "cut": [round(max(0.0, t - 1.0), 2), round(min(dur[st], t + 2.0), 2)]})
                t += 1.0
    keep = lambda lab: canonical(lab) not in ("Speech", "Music") and lab not in ("Speech", "Music") and not is_music(lab)
    B, seen = [], set()
    for st, c in dev.items():
        cands = [x for x in c["cands"] if keep(x["label"])]
        pics = [{"label": p["label"], "start": p["start"], "end": p["end"], "fate": "picture", "class": p.get("class")}
                for p in c["pictures"]]
        for x in cands + pics:
            fam = canonical(x["label"])
            k = (st, fam, round(x["start"] * 4) / 4, round(x["end"] * 4) / 4, x["fate"] == "picture")
            if k in seen:
                continue
            seen.add(k)
            a, b = max(0.0, float(x["start"]) - 2.0), min(dur[st], float(x["end"]) + 2.0)
            others = sorted({canonical(y["label"]) for y in cands if y["end"] > a and y["start"] < b} - {fam})[:len(LETTERS) - 2]
            opts = sorted(others + [fam])
            B.append({"id": f"b{len(B)}", "clip": st, "label": x["label"], "family": fam, "start": x["start"], "end": x["end"],
                      "fate": x["fate"], "class": x.get("class"), "cut": [round(a, 2), round(b, 2)], "options": opts})
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text(encoding="utf-8"))
    ITEMS.write_text(json.dumps({"a": A, "b": B, "videos": {st: vids[st] for st in dev}}, indent=0), encoding="utf-8")
    print(f"(a) {len(A)} seconds over {sum(len(r['gate']) for r in gd.values())} gate records; (b) {len(B)} items -> {ITEMS}")


# ----------------------------------------------------------------------------- run (cluster GPU)
def run():
    import soundfile as sf
    import librosa
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    it = json.loads(ITEMS.read_text(encoding="utf-8"))
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"meta": {"model": MODEL, "qa": QA, "qb": QB}, "a": {}, "b": {}}
    t0 = time.time()
    proc = Qwen3OmniMoeProcessor.from_pretrained(MODEL)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(MODEL, dtype=torch.bfloat16, device_map="auto").eval()
    try:
        model.disable_talker()
    except Exception:
        pass
    tok = proc.tokenizer
    yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("yes", "Yes", " yes", " Yes")})
    no_ids = sorted({tok.encode(w, add_special_tokens=False)[0] for w in ("no", "No", " no", " No")})
    let_ids = [tok.encode(L, add_special_tokens=False)[0] for L in LETTERS]
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    done["meta"].update(yes_ids=yes_ids, no_ids=no_ids)
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    tmp = Path(tempfile.mkdtemp(prefix="omni_probe_"))

    def save(n):
        if n % 100 == 0:
            OUT.write_text(json.dumps(done), encoding="utf-8")
            print(f"{n} answers ({time.time() - t0:.0f} s)", flush=True)

    # (a) video + audio, yes/no logit
    n = 0
    for x in it["a"]:
        if x["id"] in done["a"]:
            continue
        src = _ROOT / it["videos"][x["clip"]]
        cut = tmp / f"{x['id']}.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(x["cut"][0]), "-to", str(x["cut"][1]), "-i", str(src),
                        "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac", "-ac", "1", "-ar", str(SR), str(cut)], check=True)
        conv = [{"role": "user", "content": [{"type": "video", "video": str(cut)},
                                             {"type": "text", "text": QA.format(label=x["label"].lower())}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
        inp = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            lg = model.thinker(**inp, use_audio_in_video=True).logits[0, -1].float()
        done["a"][x["id"]] = float(lg[yes_ids].max() - lg[no_ids].max())
        cut.unlink(missing_ok=True)
        n += 1; save(n)
    OUT.write_text(json.dumps(done), encoding="utf-8")
    print(f"(a) done ({time.time() - t0:.0f} s)", flush=True)

    # (b) audio, closed choice, both option orders
    wav = {}
    for x in it["b"]:
        if x["id"] in done["b"]:
            continue
        if x["clip"] not in wav:
            wav.clear()
            w, sr = sf.read(str(Path.home() / "MscProj" / "data" / "work" / "gold_wav_flat" / f"{x['clip']}.wav"), dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            wav[x["clip"]] = w if sr == SR else librosa.resample(w, orig_sr=sr, target_sr=SR)
        w = wav[x["clip"]]
        seg = w[int(x["cut"][0] * SR):max(int(x["cut"][1] * SR), int(x["cut"][0] * SR) + SR)]
        res = {}
        for order in ("fwd", "rev"):
            opts = list(x["options"]) if order == "fwd" else list(reversed(x["options"]))
            opts = opts + ["none of these"]
            q = QB + "\n" + "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(opts))
            conv = [{"role": "user", "content": [{"type": "audio", "audio": seg}, {"type": "text", "text": q}]}]
            text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
            inp = proc(text=text, audio=[seg], return_tensors="pt", padding=True, use_audio_in_video=False)
            inp = inp.to(model.thinker.device).to(torch.bfloat16)
            with torch.inference_mode():
                lg = model.thinker(**inp).logits[0, -1].float()
                p = torch.softmax(lg[let_ids[:len(opts)]], dim=0).cpu().numpy()
                out = model.thinker.generate(**inp, max_new_tokens=16, do_sample=False, eos_token_id=eos, pad_token_id=eos[-1])
            ans = proc.batch_decode(out[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()
            own = opts.index(x["family"])
            named = {L for L in LETTERS[:len(opts)] if f"{L}" in [t.strip(" ().") for t in ans.replace(",", " ").split()]}
            res[order] = {"p_own": float(p[own]), "p_none": float(p[-1]), "answer": ans, "own_named": LETTERS[own] in named}
        done["b"][x["id"]] = res
        n += 1; save(n)
    OUT.write_text(json.dumps(done), encoding="utf-8")
    print(f"DONE ({time.time() - t0:.0f} s)", flush=True)


# ----------------------------------------------------------------------------- report (local)
def auroc(scores, labels):
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    if y.all() or not y.any():
        return None
    from scipy.stats import rankdata
    r = rankdata(s)
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def report():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.coverage import score_coverage as V
    gold = S.load_gold([V.GOLD])
    it = json.loads(ITEMS.read_text(encoding="utf-8"))
    ans = json.loads(OUT.read_text(encoding="utf-8"))
    gd = json.loads((HERE / "gate_dev.json").read_text(encoding="utf-8"))
    L = ["# Step 3 prep: Qwen3-Omni probes on DEV (measure only)", ""]
    # (a)
    rows, left = [], {"obvious only": 0, "no gold sound": 0}
    for st, r in gd.items():
        for gi, g in enumerate(r["gate"]):
            m = [x for x in gold[st] if S.same_family(x["label"], g["label"]) and x["end"] > g["start"] and x["start"] < g["end"]]
            if not m:
                left["no gold sound"] += 1; continue
            if any(x["visible"] for x in m):
                y = True
            elif all(x["needed"] for x in m):
                y = False
            else:
                left["obvious only"] += 1; continue
            sc = [ans["a"][x["id"]] for x in it["a"] if x["clip"] == st and x["gate"] == gi and x["id"] in ans["a"]]
            st_ = g["stretches"]
            maj = np.mean([v["seen"] for v in st_]) if st_ else 0.0
            ab = np.mean([(v["ab"] if v["ab"] is not None else v["seen"]) for v in st_]) if st_ else 0.0
            rows.append({"y": y, "mean": float(np.mean(sc)) if sc else 0.0, "max": float(np.max(sc)) if sc else 0.0, "maj": maj, "ab": ab})
    y = [r["y"] for r in rows]
    L += ["## (a) is the source visibly making the sound (gate records)", "",
          f"{sum(y)} visible, {len(y) - sum(y)} not visible; left out: {left}", "",
          "| score | AUROC (visible vs not) |", "|---|---|"]
    for k, nm in (("mean", "Omni, mean over seconds"), ("max", "Omni, max over seconds"), ("maj", "current gate (share of stretches seen)"),
                  ("ab", "a/b rule (share seen, split -> majority)")):
        a = auroc([r[k] for r in rows], y)
        L.append(f"| {nm} | {'-' if a is None else f'{a:.3f}'} |")
    # (b)
    def truth(st, fam, a, b):
        return any(S.same_family(x["label"], fam) and x["end"] > a - 0.5 and x["start"] < b + 0.5 for x in gold[st])
    B = []
    for x in it["b"]:
        if x["id"] not in ans["b"]:
            continue
        r = ans["b"][x["id"]]
        B.append({"fate": x["fate"], "class": x["class"], "y": truth(x["clip"], x["family"], x["start"], x["end"]),
                  "p": (r["fwd"]["p_own"] + r["rev"]["p_own"]) / 2, "both": r["fwd"]["own_named"] and r["rev"]["own_named"]})
    L += ["", "## (b) closed choice: which of these heard families, or none (audio)", "",
          "| set | n | right | AUROC (first-token prob of own letter) | named in both orders: right / wrong |", "|---|---|---|---|---|"]
    sets = {"all candidates": [r for r in B if r["fate"] != "picture"], "kept (drawn)": [r for r in B if r["fate"] == "drawn"],
            "dropped": [r for r in B if r["fate"] == "dropped"]}
    pics = [dict(r, y=not str(r["class"]).startswith("wrong: a different") and not str(r["class"]).startswith("wrong: no such"))
            for r in B if r["fate"] == "picture"]
    sets["frozen DEV pictures (right = not other-sound / no-sound)"] = pics
    for nm, rr in sets.items():
        a = auroc([r["p"] for r in rr], [r["y"] for r in rr])
        yr = [r for r in rr if r["y"]]; nr = [r for r in rr if not r["y"]]
        L.append(f"| {nm} | {len(rr)} | {len(yr)} | {'-' if a is None else f'{a:.3f}'} | "
                 f"{sum(r['both'] for r in yr)}/{len(yr)} / {sum(r['both'] for r in nr)}/{len(nr)} |")
    (HERE / "omni_probe_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"items": items, "run": run, "report": report}[sys.argv[1]]()
