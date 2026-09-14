"""An evaluation reference that no part of the evaluated system wrote.

The reference is the sentence the judge scores every system against: what a hearing
viewer gets from this clip that a deaf viewer misses. Until 2026-09-14 it was built from
the DETECTOR's own event list and written by the PROPOSED SYSTEM's own VLM, so the blind
audio-to-image baseline -- which draws exactly that event list -- was being scored
against its own input and could not lose, and the proposed system's reasoning model was
writing the answer sheet. (Found in a design review by a second model, which also
shaped what follows.) The supervisor wants no humans in the loop, so the fix is a
reference produced by models that never see the system's output:

    LISTEN   Qwen2-Audio-7B-Instruct on the audio with vocals REMOVED (Demucs), in 10 s
             chunks, lists non-speech sound sources. Vocals out, so dialogue cannot leak
             into "sounds heard"; chunks, so a 25 s clip is not truncated.
    VERIFY   CLAP zero-shot: each listed item is kept only if the audio actually
             resembles it more than it resembles random AudioSet labels. The audio LM's
             hallucination filter.
    LOOK     a VLM that is NOT the system's (Idefics3-8B-Llama3: SigLIP + Llama-3.1)
             lists what is visible that could make sound.
    WRITE    a text LLM that is neither the system's model nor the judge (Llama-3.1-8B
             if licensed, else Phi-3.5-mini) turns those two lists into the sentence.
    SENTINEL "nothing beyond the picture" is decided by a rule, not by any model
             saying "none": every verified sound has a visible match at cosine >= tau
             (sentence embeddings; tau calibrated on DCASE 2025 gold).

Both references are reported: the detector-derived one as the proposal specified, this
one as the corrected primary. Each model is loaded, used for all clips, and unloaded
before the next, because none of them co-fit with the judge on a 23 GB card.
"""
from __future__ import annotations

import gc
import json
import sys
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

import config
from src.text_similarity import _as_tensor

LISTEN_MODEL = "Qwen/Qwen2-Audio-7B-Instruct"
CLAP_MODEL = "laion/clap-htsat-fused"     # the unfused checkpoint is .bin-only; transformers 5 refuses it on torch 2.5
LOOK_MODEL = "HuggingFaceM4/Idefics3-8B-Llama3"
WRITE_MODELS = ["meta-llama/Llama-3.1-8B-Instruct", "microsoft/Phi-3.5-mini-instruct"]
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

LISTEN_PROMPT = ("List every distinct non-speech sound source in this audio, one per "
                 "line, as \"source: sound\". Only sounds actually audible. If none, "
                 "output exactly: none.")
LOOK_PROMPT = ("List concrete objects, people or vehicles visible in these frames that "
               "could be producing sound, one per line, max 12. No guessing.")
WRITE_PROMPT = ("A video's soundtrack, with the dialogue removed, contains these sounds: "
                "{sounds}.\nVerified visible on screen: {visible}.\n{speech}\n"
                "In ONE sentence, state the information a hearing viewer gets from the "
                "sounds that a deaf viewer would miss from the picture alone. Name the "
                "specific sound sources.")
NOTHING_MISSING = "nothing beyond the picture"
CHUNK = 10.0
CLAP_MARGIN = 2.0        # keep an item if cos(audio, item) > mean + MARGIN*std over decoys
MATCH_TAU = 0.50         # visible-match cosine; calibrated on DCASE gold (see notes)


def _free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# ----------------------------------------------------------------------------- audio
def _remove_vocals(wav: Path, out_dir: Path) -> Path:
    """Demucs (htdemucs) no_vocals stem at 16 kHz mono; the input if Demucs fails."""
    try:
        subprocess.run(["python", "-m", "demucs", "--two-stems=vocals", "-n", "htdemucs",
                        "-o", str(out_dir), str(wav)],
                       check=True, capture_output=True, text=True)
        stem = out_dir / "htdemucs" / wav.stem / "no_vocals.wav"
        mono = out_dir / (wav.stem + "_novocals16k.wav")
        subprocess.run(["ffmpeg", "-y", "-i", str(stem), "-ac", "1", "-ar", "16000",
                        str(mono), "-loglevel", "error"], check=True)
        return mono
    except Exception as e:
        print(f"       [ref] demucs unavailable ({type(e).__name__}); using the mix")
        return wav


def _with_audio(proc, audio_list, sr, **kw):
    """Call a processor with audio under whichever keyword this transformers accepts.

    transformers 5 renamed ``audios`` to ``audio`` and IGNORES the old name with a one-line
    warning; the first reference run (job 28942612) then asked Qwen2-Audio about a prompt
    with no sound attached and got "none." for all 100 clips. Fail loudly instead.
    """
    import inspect
    key = "audio" if "audio" in inspect.signature(proc.__call__).parameters else "audios"
    out = proc(**{key: audio_list}, sampling_rate=sr, return_tensors="pt", **kw)
    if "input_features" not in out:
        raise RuntimeError(f"{type(proc).__name__} produced no audio features (keyword {key!r})")
    return out


def _split_items(lines: List[str]) -> List[str]:
    """Qwen2-Audio answers in its own tag style -- one line such as
    'source: wind, source: bird vocalization; source: clock ticking.' -- rather than one
    sound per line. Split on the 'source:' markers and semicolons, keep the commas inside
    a tag ('clock, clicking, tick' is one AudioSet name), drop empties and 'none'.
    """
    import re
    out: List[str] = []
    for line in lines:
        for part in re.split(r"(?:^|[;,]?\s*)source\s*:\s*|;", line, flags=re.I):
            part = part.strip(" -*•.	").strip()
            if part and part.lower() != "none" and part not in out:
                out.append(part)
    return out


def _chunks(audio: np.ndarray, sr: int, sec: float = CHUNK):
    n = int(sec * sr)
    for i in range(0, max(1, len(audio)), n):
        seg = audio[i:i + n]
        if len(seg) >= sr:            # at least one second
            yield seg


# Tags the audio LM uses for speech; dropped from the MIX pass only, since the whole point
# of listening to the no-vocals stem first is to keep the dialogue out of the reference.
SPEECH_TAGS = {"speech", "male speech", "female speech", "male speech, man speaking",
               "female speech, woman speaking", "conversation", "narration", "narration, monologue",
               "human voice", "speech synthesizer", "child speech, kid speaking", "talking"}


def _is_speech_tag(item: str) -> bool:
    low = item.lower().strip()
    return low in SPEECH_TAGS or low.startswith("speech") or low.endswith("speech")


def _listen_once(proc, mdl, audio: np.ndarray, sr: int) -> List[str]:
    import torch
    items: List[str] = []
    reply = ""
    for seg in _chunks(audio, sr):
        conv = [{"role": "user", "content": [{"type": "audio", "audio_url": ""},
                                             {"type": "text", "text": LISTEN_PROMPT}]}]
        text = proc.apply_chat_template(conv, add_generation_prompt=True, tokenize=False)
        inputs = _with_audio(proc, [seg], sr, text=text).to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**inputs, max_new_tokens=96, do_sample=False)
        reply = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0].strip()
        for line in reply.splitlines():
            line = line.strip(" -*•").strip()
            if line and line.lower().rstrip(".") != "none" and line not in items:
                items.append(line)
    return _split_items(items)


def listen_all(clips: Dict[str, Path], device: str) -> Dict[str, List[str]]:
    """clip -> sounds named by the audio LM: the union of a pass over the no-vocals stem
    and a pass over the raw mix (speech tags dropped from the latter).

    The stem pass alone came back empty on 12 of the first 37 clips of job 28942680,
    among them an aviary, crossing bells and two storms: Demucs pulls tonal sounds into
    the vocals stem, and a 7B audio LM takes the "none" exit on quiet audio. The mix pass
    restores recall; CLAP verification against decoys (verify_all) remains the only
    filter. Where each item was heard (stem / mix / both) is logged so the report can say
    how often the stem was hollow -- a limitation of the system's own separation front
    end as much as of this reference. (Review, 2026-09-14.)
    """
    import librosa
    from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
    import torch
    proc = AutoProcessor.from_pretrained(LISTEN_MODEL)
    mdl = Qwen2AudioForConditionalGeneration.from_pretrained(
        LISTEN_MODEL, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None).eval()
    sr = proc.feature_extractor.sampling_rate
    out = {}
    with tempfile.TemporaryDirectory() as td:
        for name, wav in clips.items():
            nov = _remove_vocals(Path(wav), Path(td))
            stem_audio, _ = librosa.load(str(nov), sr=sr, mono=True)
            mix_audio, _ = librosa.load(str(wav), sr=sr, mono=True)
            stem_items = _listen_once(proc, mdl, stem_audio, sr)
            mix_items = [it for it in _listen_once(proc, mdl, mix_audio, sr) if not _is_speech_tag(it)]
            items = list(stem_items)
            for it in mix_items:
                if it not in items:
                    items.append(it)
            where = {it: ("both" if it in stem_items and it in mix_items else
                          "stem" if it in stem_items else "mix") for it in items}
            out[name] = items
            reply = f"stem={stem_items} mix={mix_items}"
            print(f"       [ref] listen {name}: " +
                  " | ".join(f"{it} [{where[it]}]" for it in items), flush=True)
            if len(out) == 1 and not items:
                # sanity gate (review, 2026-09-14): a blind audio LM fails in two minutes,
                # not two hours. The first clip is a random benchmark clip; an empty list
                # there is far likelier to be a broken input than a genuinely silent clip.
                sys.exit(f"[ref] listen returned nothing for the first clip {name}; "
                         f"last reply was {reply!r} -- input or model misconfigured")
    del mdl, proc
    _free()
    return out


def verify_all(clips: Dict[str, Path], heard: Dict[str, List[str]], device: str,
               decoys: int = 64) -> Dict[str, List[str]]:
    """Keep the items the audio actually resembles (CLAP), per clip."""
    import librosa
    import torch
    from transformers import ClapModel, ClapProcessor
    proc = ClapProcessor.from_pretrained(CLAP_MODEL)
    mdl = ClapModel.from_pretrained(CLAP_MODEL).to(device).eval()
    names = json.loads((Path(__file__).resolve().parents[1] / "audioset_mid_names.json")
                       .read_text("utf-8"))
    pool = sorted(set(names.values()))
    rng = np.random.default_rng(0)
    out = {}
    for name, wav in clips.items():
        # speech tags leak through the stem pass too (Demucs leaves residue); the
        # reference is about the non-speech soundtrack by definition of the task
        items = [it for it in _split_items(heard.get(name, [])) if not _is_speech_tag(it)]
        if not items:
            out[name] = []
            continue
        audio, _ = librosa.load(str(wav), sr=48000, mono=True)
        with torch.no_grad():
            a = _with_audio(proc, [audio], 48000).to(device)
            ea = _as_tensor(mdl.get_audio_features(**a))   # ModelOutput on transformers 5
            ea = ea / ea.norm(dim=-1, keepdim=True)
            texts = [f"the sound of {it}" for it in items]
            dec = [f"the sound of {pool[i].lower()}" for i in rng.choice(len(pool), decoys, replace=False)]
            t = proc(text=texts + dec, return_tensors="pt", padding=True).to(device)
            et = _as_tensor(mdl.get_text_features(**t))
            et = et / et.norm(dim=-1, keepdim=True)
            sims = (ea @ et.T)[0].cpu().numpy()
        s_items, s_dec = sims[:len(items)], sims[len(items):]
        bar = float(s_dec.mean() + CLAP_MARGIN * s_dec.std())
        kept = [it for it, sc in zip(items, s_items) if sc > bar]
        print(f"       [ref] verify {name}: bar={bar:.3f} " +
              " ".join(f"{it[:24]}={sc:.3f}{'*' if sc > bar else ''}" for it, sc in zip(items, s_items)),
              flush=True)
        out[name] = kept
    del mdl, proc
    _free()
    return out


# ----------------------------------------------------------------------------- video
def look_all(videos: Dict[str, Path], device: str, n_frames: int = 6) -> Dict[str, List[str]]:
    """clip -> list of visible sound-capable things, from a VLM that is not the system's."""
    import torch
    from transformers import AutoProcessor, Idefics3ForConditionalGeneration
    from src.stage2_video_understanding import _sample_frames
    proc = AutoProcessor.from_pretrained(LOOK_MODEL)
    mdl = Idefics3ForConditionalGeneration.from_pretrained(
        LOOK_MODEL, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None).eval()
    out = {}
    for name, video in videos.items():
        frames = _sample_frames(Path(video), n_frames)
        if not frames:
            out[name] = []
            continue
        content = [{"type": "image"} for _ in frames] + [{"type": "text", "text": LOOK_PROMPT}]
        text = proc.apply_chat_template([{"role": "user", "content": content}],
                                        add_generation_prompt=True)
        inputs = proc(text=text, images=frames, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**inputs, max_new_tokens=96, do_sample=False)
        reply = proc.batch_decode(gen[:, inputs["input_ids"].shape[1]:],
                                  skip_special_tokens=True)[0]
        reply = reply.split("Assistant:")[-1].strip()
        items = [l.strip(" -*•123456789.").strip() for l in reply.splitlines()]
        items = [i for i in items if i and len(i.split()) <= 8][:12]
        out[name] = items
        print(f"       [ref] look {name}: {items}", flush=True)
    del mdl, proc
    _free()
    return out


# ----------------------------------------------------------------------------- write
def _matches(sounds: List[str], visible: List[str], device: str):
    """For each sound, the best cosine against the visible list (sentence embeddings)."""
    if not sounds:
        return []
    if not visible:
        return [0.0] * len(sounds)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(EMBED_MODEL, device=device)
    es = m.encode(sounds, normalize_embeddings=True)
    ev = m.encode(visible, normalize_embeddings=True)
    return (es @ ev.T).max(axis=1).tolist()


def write_all(sounds: Dict[str, List[str]], visible: Dict[str, List[str]],
              transcripts: Dict[str, str], device: str) -> Dict[str, dict]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    mdl = tok = used = None
    for repo in WRITE_MODELS:
        try:
            tok = AutoTokenizer.from_pretrained(repo)
            mdl = AutoModelForCausalLM.from_pretrained(
                repo, torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None).eval()
            used = repo
            break
        except Exception as e:
            print(f"       [ref] writer {repo} unavailable ({type(e).__name__}); trying next")
    if mdl is None:
        raise RuntimeError("no reference writer available")
    print(f"       [ref] writer: {used}", flush=True)
    out = {}
    for name in sounds:
        snd, vis = sounds[name], visible.get(name, [])
        sims = _matches(snd, vis, device)
        unmatched = [s for s, c in zip(snd, sims) if c < MATCH_TAU]
        if not snd or not unmatched:
            out[name] = {"reference": NOTHING_MISSING, "sounds": snd, "visible": vis,
                         "matches": sims, "writer": used, "rule": "all sounds visible" if snd else "no sounds"}
            continue
        speech = ("Someone is speaking; the dialogue is already captioned, so ignore it."
                  if transcripts.get(name, "").strip() else "There is no speech.")
        prompt = WRITE_PROMPT.format(sounds="; ".join(unmatched), visible="; ".join(vis) or "nothing relevant",
                                     speech=speech)
        msgs = [{"role": "user", "content": prompt}]
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(text, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            gen = mdl.generate(**inputs, max_new_tokens=80, do_sample=False)
        ref = tok.decode(gen[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out[name] = {"reference": ref, "sounds": snd, "visible": vis, "matches": sims,
                     "unmatched": unmatched, "writer": used, "rule": "written"}
        print(f"       [ref] write {name}: {ref}", flush=True)
    del mdl, tok
    _free()
    return out


# ----------------------------------------------------------------------------- driver
def build_all(clips: Dict[str, Path], videos: Dict[str, Path], transcripts: Dict[str, str],
              cache: Path, device: str = "cuda") -> Dict[str, dict]:
    """All clips through the four steps, each model resident once. Resumable per step."""
    state = json.loads(cache.read_text("utf-8")) if cache.exists() else {}
    def save():
        tmp = cache.with_suffix(".tmp")   # atomic: a kill mid-write leaves the old cache
        tmp.write_text(json.dumps(state, indent=1, ensure_ascii=False), encoding="utf-8")
        tmp.replace(cache)
    if "heard" not in state:
        state["heard"] = listen_all(clips, device); save()
    if "verified" not in state:
        state["verified"] = verify_all(clips, state["heard"], device); save()
    if "visible" not in state:
        state["visible"] = look_all(videos, device); save()
    if "references" not in state:
        state["references"] = write_all(state["verified"], state["visible"], transcripts, device); save()
    return state["references"]
