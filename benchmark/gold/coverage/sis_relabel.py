"""PREREG_sis_relabel.md step 1 (cluster CPU, from ~/wt_slice; mpnet cached): the "Specific impact sounds" DASM runs of the
P4 caches, both listeners' V4 answers resolved to families with the pipeline's V4 matcher; writes sis_candidates.json.
    python benchmark/gold/coverage/sis_relabel.py"""
import json, re, sys
from pathlib import Path
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_ROOT))
from benchmark.gold import listener_variants as LV

HERE = Path(__file__).resolve().parent
P4 = Path.home() / "MscProj_tg" / "benchmark" / "gold"


def lines_of(txt):
    from benchmark.gold.listener_open_inventory import lines_of as L
    return L(txt)


def main():
    from sentence_transformers import SentenceTransformer
    O = LV.Onto()
    voc = next(p for p in (_ROOT / "benchmark/gold/depictable_vocab.json", Path.home() / "MscProj_r13/benchmark/gold/depictable_vocab.json") if p.exists())
    fams = sorted(json.loads(voc.read_text(encoding="utf-8"))["families"])
    names = {f: LV.match_names(O, f) for f in fams}
    emb = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    femb = emb.encode([f.lower() for f in fams], normalize_embeddings=True, convert_to_numpy=True)

    def fam_parse(txt):
        ls = lines_of(txt)
        if not ls:
            return []
        e = emb.encode(ls, normalize_embeddings=True, convert_to_numpy=True)
        cos = e @ femb.T
        got = []
        for li, ln in enumerate(ls):
            for j, f in enumerate(fams):
                word = any(re.search(r"\b" + re.escape(nm) + r"(?:s|es)?\b", ln.lower()) for nm in names[f])
                if (word or cos[li, j] > LV.COS) and f not in got:
                    got.append(f)
        return got
    out = []
    for split in ("dev", "test", "dev2", "test2"):
        d = json.loads((P4 / f"{split}_listener_p4.json").read_text(encoding="utf-8"))
        for x in (d["items"] if isinstance(d, dict) else d):
            if not str(x.get("family", "")).startswith("Specific impact"):
                continue
            q, a = fam_parse(x.get("qwen_v4_text") or ""), fam_parse(x.get("afn_v4_text") or "")
            both = [f for f in q if f in a]
            out.append({"split": split, "clip": x["clip"], "start": float(x["start"]), "end": float(x["end"]), "peak": float(x["peak"]),
                        "qwen": q, "af": a, "label": both[0] if both else None})
    (HERE / "sis_candidates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for o in out:
        print(o["clip"], o["start"], o["label"], "| Q", o["qwen"][:4], "| AF", o["af"][:4])
    print("relabelled", sum(o["label"] is not None for o in out), "of", len(out))


if __name__ == "__main__":
    main()
