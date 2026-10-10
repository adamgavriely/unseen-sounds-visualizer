"""Step 3b (PREREG_step3b_omni_verify.md): uniform Qwen3-Omni verification of every DEV candidate burst. Measure only.

    python benchmark/gold/coverage/omni_verify.py items    # local: verify_items_dev.json (no gold read)
    python benchmark/gold/coverage/omni_verify.py run      # cluster GPU: omni_verify_dev.json (resumes)
    python benchmark/gold/coverage/omni_verify.py report   # local: AUROCs -> omni_verify_dev.md
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
SPLIT = os.environ.get("SPLIT", "DEV")
SUF = "dev" if SPLIT == "DEV" else "test"
ITEMS, OUT = HERE / f"verify_items_{SUF}.json", HERE / f"omni_verify_{SUF}.json"
MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SR = 16000
GAP, MAX_SIB = 2.5, 12
Q1 = "Which ONE of these best names the sound you hear in this recording? Answer with the letter only."
Q2 = ("You are watching a short video with its sound. A sound of {fam} starts in it. Is the thing making this sound on "
      "screen, or is there a visible event that makes it, like a flash or a blast? Answer yes or no.")
Q3 = "Is the sound of {fam} heard in this recording? Answer yes or no."
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def items():
    from src.labels import canonical, is_music, _parents
    s = (_ROOT / "docs" / "decision_trail" / "data.js").read_text(encoding="utf-8")
    dj = json.loads(s[s.index("=") + 1:].rstrip().rstrip(";"))
    pf = json.loads((HERE / "pics_frozen.json").read_text(encoding="utf-8"))["clips"]
    par = _parents()
    kids = {}
    for c, p in par.items():
        for q in (p if isinstance(p, (list, tuple, set)) else [p]):
            kids.setdefault(q, set()).add(c)
    bad = lambda lab: canonical(lab) in ("Speech", "Music") or lab in ("Speech", "Music") or is_music(lab)
    bursts, secs = [], []
    for c in (c for c in dj["clips"] if c["split"] == SPLIT):
        st, dur = c["clip"], float(pf[c["clip"]]["dur"])
        by = {}
        for x in c["cands"]:
            if not bad(x["label"]):
                by.setdefault(canonical(x["label"]), []).append(x)
        for fam, xs in sorted(by.items()):
            xs.sort(key=lambda x: x["start"])
            cur = None
            for x in xs:
                if cur and x["start"] - cur["end"] <= GAP:
                    cur["end"] = max(cur["end"], x["end"]); cur["members"].append(x["id"]); cur["labels"].add(x["label"])
                    cur["origins"].add(x["origin"]); cur["drawn"] |= x["fate"] == "drawn"
                else:
                    cur = {"clip": st, "family": fam, "start": x["start"], "end": x["end"], "members": [x["id"]],
                           "labels": {x["label"]}, "origins": {x["origin"]}, "drawn": x["fate"] == "drawn"}
                    bursts.append(cur)
        for b in bursts:
            if b["clip"] != st or "id" in b:
                continue
            b["id"] = f"v{len([z for z in bursts if 'id' in z])}"
            sib = set()
            for lab in b["labels"]:
                p = par.get(lab)
                for q in (p if isinstance(p, (list, tuple, set)) else [p] if p else []):
                    sib |= {canonical(k) for k in kids.get(q, ())}
            sib = sorted(x for x in sib - {b["family"]} if not bad(x))[:MAX_SIB]
            b["options"] = sorted(sib + [b["family"]])
            b["cut1"] = [round(max(0.0, b["start"] - 2.0), 2), round(min(dur, b["end"] + 2.0), 2)]
            b["cut2"] = [round(max(0.0, b["start"] - 1.0), 2), round(min(dur, b["start"] + 1.0), 2)]
            t = float(b["start"])
            while t < min(dur, b["end"] + 3.0) - 1e-6:
                secs.append({"id": f"{b['id']}s{len(secs)}", "burst": b["id"], "clip": st, "family": b["family"], "t": round(t, 2),
                             "cut": [round(max(0.0, t - 0.5), 2), round(min(dur, t + 1.5), 2)]})
                t += 1.0
            b["labels"], b["origins"] = sorted(b["labels"]), sorted(b["origins"])
    vids = json.loads((HERE.parent / "one_model_baseline" / "videos.json").read_text(encoding="utf-8"))
    ITEMS.write_text(json.dumps({"bursts": bursts, "seconds": secs, "videos": {b["clip"]: vids[b["clip"]] for b in bursts}}),
                     encoding="utf-8")
    print(f"{len(bursts)} bursts, {len(secs)} seconds -> {ITEMS}")


def run():
    import soundfile as sf
    import librosa
    import torch
    from qwen_omni_utils import process_mm_info
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor
    it = json.loads(ITEMS.read_text(encoding="utf-8"))
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {"meta": {"model": MODEL, "q1": Q1, "q2": Q2, "q3": Q3},
                                                                             "v1": {}, "v2": {}, "v3": {}}
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
    print(f"loaded in {time.time() - t0:.0f} s", flush=True)
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")
    tmp = Path(tempfile.mkdtemp(prefix="omni_verify_"))
    wav = {}

    def audio(st, a, b):
        if st not in wav:
            wav.clear()
            w, sr = sf.read(str(Path.home() / "MscProj" / "data" / "work" / "gold_wav_flat" / f"{st}.wav"), dtype="float32")
            if w.ndim > 1:
                w = w.mean(axis=1)
            wav[st] = w if sr == SR else librosa.resample(w, orig_sr=sr, target_sr=SR)
        w = wav[st]
        return w[int(a * SR):max(int(b * SR), int(a * SR) + SR)]

    def last_logits(content, use_av=False, audio_arr=None):
        conv = [{"role": "user", "content": content}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        if use_av:
            audios, images, videos = process_mm_info(conv, use_audio_in_video=True)
            inp = proc(text=text, audio=audios, images=images, videos=videos, return_tensors="pt", padding=True, use_audio_in_video=True)
        else:
            inp = proc(text=text, audio=[audio_arr], return_tensors="pt", padding=True, use_audio_in_video=False)
        inp = inp.to(model.thinker.device).to(torch.bfloat16)
        with torch.inference_mode():
            kw = {"use_audio_in_video": True} if use_av else {}
            return model.thinker(**inp, **kw).logits[0, -1].float()

    def p_yes(lg):
        y, n = lg[yes_ids].max(), lg[no_ids].max()
        return float(torch.sigmoid(y - n))

    n = [0]

    def tick():
        n[0] += 1
        if n[0] % 200 == 0:
            OUT.write_text(json.dumps(done), encoding="utf-8")
            print(f"{n[0]} answers ({time.time() - t0:.0f} s)", flush=True)

    for b in it["bursts"]:                                   # (1) verifier
        if b["id"] in done["v1"]:
            continue
        seg = audio(b["clip"], *b["cut1"])
        res = {}
        for order in ("fwd", "rev"):
            opts = (list(b["options"]) if order == "fwd" else list(reversed(b["options"]))) + ["none of these"]
            q = Q1 + "\n" + "\n".join(f"({LETTERS[i]}) {o}" for i, o in enumerate(opts))
            lg = last_logits([{"type": "audio", "audio": seg}, {"type": "text", "text": q}], audio_arr=seg)
            p = torch.softmax(lg[let_ids[:len(opts)]], dim=0).cpu().numpy().tolist()
            res[order] = {"options": opts, "p": p}
        done["v1"][b["id"]] = res
        tick()
    OUT.write_text(json.dumps(done), encoding="utf-8")
    print(f"(1) done ({time.time() - t0:.0f} s)", flush=True)
    for b in it["bursts"]:                                   # (2) onset gate, audio + 4 fps video
        if b["id"] in done["v2"]:
            continue
        cut = tmp / f"{b['id']}.mp4"
        subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-ss", str(b["cut2"][0]), "-to", str(b["cut2"][1]),
                        "-i", str(_ROOT / it["videos"][b["clip"]]), "-c:v", "libx264", "-preset", "veryfast", "-c:a", "aac",
                        "-ac", "1", "-ar", str(SR), str(cut)], check=True)
        lg = last_logits([{"type": "video", "video": str(cut), "fps": 4.0},
                          {"type": "text", "text": Q2.format(fam=b["family"].lower())}], use_av=True)
        done["v2"][b["id"]] = p_yes(lg)
        cut.unlink(missing_ok=True)
        tick()
    OUT.write_text(json.dumps(done), encoding="utf-8")
    print(f"(2) done ({time.time() - t0:.0f} s)", flush=True)
    for s in it["seconds"]:                                  # (3) still heard
        if s["id"] in done["v3"]:
            continue
        seg = audio(s["clip"], *s["cut"])
        lg = last_logits([{"type": "audio", "audio": seg}, {"type": "text", "text": Q3.format(fam=s["family"].lower())}], audio_arr=seg)
        done["v3"][s["id"]] = p_yes(lg)
        tick()
    OUT.write_text(json.dumps(done), encoding="utf-8")
    print(f"DONE ({time.time() - t0:.0f} s)", flush=True)


def report():
    from benchmark.gold import score_per_sound as S
    from benchmark.gold.coverage import score_coverage as V
    from benchmark.gold.coverage.omni_probe import auroc
    gold = S.load_gold([V.GOLD])
    it = json.loads(ITEMS.read_text(encoding="utf-8"))
    ans = json.loads(OUT.read_text(encoding="utf-8"))
    gd = json.loads((HERE / "gate_dev.json").read_text(encoding="utf-8"))
    L = ["# Step 3b: Qwen3-Omni verification of every DEV candidate burst (measure only)", ""]
    match = lambda b: [x for x in gold[b["clip"]] if S.same_family(x["label"], b["family"]) and x["end"] > b["start"] - 0.5 and x["start"] < b["end"] + 0.5]
    rows = []
    for b in it["bursts"]:
        if b["id"] not in ans["v1"]:
            continue
        r = ans["v1"][b["id"]]
        p = np.mean([r[o]["p"][r[o]["options"].index(b["family"])] for o in ("fwd", "rev")])
        pn = np.mean([r[o]["p"][-1] for o in ("fwd", "rev")])
        rows.append({"b": b, "y": bool(match(b)), "p": float(p), "pnone": float(pn)})
    L += ["## (1) verifier: P(own family) in a closed choice {family, siblings, none}", "",
          "| bursts | n | right | AUROC |", "|---|---|---|---|"]
    groups = {"all": rows, "drawn by the frozen system": [r for r in rows if r["b"]["drawn"]],
              "not drawn": [r for r in rows if not r["b"]["drawn"]]}
    for o in ("beats", "flexsed", "flexsed band", "dasm"):
        groups[f"has a {o} member"] = [r for r in rows if o in r["b"]["origins"]]
    for nm, rr in groups.items():
        a = auroc([r["p"] for r in rr], [r["y"] for r in rr])
        L.append(f"| {nm} | {len(rr)} | {sum(r['y'] for r in rr)} | {'-' if a is None else f'{a:.3f}'} |")
    a = auroc([-r["pnone"] for r in rows], [r["y"] for r in rows])
    L.append(f"| all, score = 1 - P(none) | {len(rows)} | {sum(r['y'] for r in rows)} | {'-' if a is None else f'{a:.3f}'} |")
    # (2) gate
    G = []
    for b in it["bursts"]:
        m = match(b)
        if not m or b["id"] not in ans["v2"]:
            continue
        if any(x["visible"] for x in m):
            y = True
        elif all(x["needed"] for x in m):
            y = False
        else:
            continue
        rec = [g for g in gd[b["clip"]]["gate"] if S.same_family(g["label"], b["family"]) and g["end"] > b["start"] and g["start"] < b["end"]]
        st = [v for g in rec for v in g["stretches"]]
        G.append({"y": y, "p": ans["v2"][b["id"]], "judged": bool(st),
                  "maj": float(np.mean([v["seen"] for v in st])) if st else None,
                  "ab": float(np.mean([(v["ab"] if v["ab"] is not None else v["seen"]) for v in st])) if st else None})
    J = [g for g in G if g["judged"]]
    f = lambda v: "-" if v is None else f"{v:.3f}"
    L += ["", "## (2) onset gate: P(yes, the maker or a visible event is on screen), audio + 4 fps video", "",
          "| set | n | visible | Omni AUROC | current gate AUROC | a/b rule AUROC |", "|---|---|---|---|---|---|",
          f"| right-family bursts with a visibility label | {len(G)} | {sum(g['y'] for g in G)} | {f(auroc([g['p'] for g in G], [g['y'] for g in G]))} | - | - |",
          f"| of these, judged by the current gate | {len(J)} | {sum(g['y'] for g in J)} | {f(auroc([g['p'] for g in J], [g['y'] for g in J]))} | "
          f"{f(auroc([g['maj'] for g in J], [g['y'] for g in J]))} | {f(auroc([g['ab'] for g in J], [g['y'] for g in J]))} |"]
    # (3) still heard
    sc, ys = [], []
    for s in it["seconds"]:
        if s["id"] not in ans["v3"]:
            continue
        t = s["t"] + 0.5
        ys.append(any(S.same_family(x["label"], s["family"]) and x["start"] <= t <= x["end"] for x in gold[s["clip"]]))
        sc.append(ans["v3"][s["id"]])
    L += ["", "## (3) still heard? per second over burst + 3 s", "", f"{len(sc)} seconds, {sum(ys)} inside a same-family gold sound; AUROC {f(auroc(sc, ys))}"]
    (HERE / "omni_verify_dev.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"items": items, "run": run, "report": report}[sys.argv[1]]()
