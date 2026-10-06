"""Candidates for a new labelled set: the 422 fresh AudioSet-Strong clips (benchmark/gold/audioset_fresh.json on the
cluster, EVAL split, every benchmark YouTube id excluded at build time) that have at least one drawable non-speech
event >= 1 s (label passes the shipped "depictable" filter, is not speech / music). Reads AudioSet labels only.

    python benchmark/gold/coverage/fresh_candidates.py path/to/audioset_fresh.json   -> fresh_candidates.json / .md
"""
import json
import sys
from collections import Counter
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
import config
from src.labels import canonical, is_salient_nonspeech, is_music

HERE = Path(__file__).resolve().parent
MIN_S = 1.0


def drawable(label):
    old = config.LABEL_FILTER
    config.LABEL_FILTER = "depictable"
    try:
        return bool(is_salient_nonspeech(label)) and not is_music(label) and canonical(label) not in ("Speech", "Music")
    finally:
        config.LABEL_FILTER = old


def main():
    d = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    out, fam = [], Counter()
    for c in d["clips"]:
        ev = [e for e in c["events"] if e["end"] - e["start"] >= MIN_S and drawable(e["label"])]
        if ev:
            fams = sorted({canonical(e["label"]) for e in ev})
            fam.update(fams)
            out.append({"id": c["id"], "stratum": c["stratum"], "duration": c["duration"], "families": fams,
                        "events": [{k: e[k] for k in ("label", "start", "end", "masked")} for e in ev]})
    (HERE / "fresh_candidates.json").write_text(json.dumps({"source": "audioset_fresh.json (422 clips)", "min_s": MIN_S,
                                                            "n": len(out), "clips": out}, indent=1), encoding="utf-8")
    L = [f"# New-set candidates: {len(out)} of {len(d['clips'])} fresh AudioSet-Strong clips have a drawable non-speech event >= {MIN_S:.0f} s",
         "", f"Strata: {dict(Counter(c['stratum'] for c in out))}. Families (clips):", "",
         "| family | clips |", "|---|---|"] + [f"| {f} | {n} |" for f, n in fam.most_common()]
    (HERE / "fresh_candidates.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:3]), f"\n{len(fam)} families; top: {fam.most_common(12)}")


if __name__ == "__main__":
    main()
