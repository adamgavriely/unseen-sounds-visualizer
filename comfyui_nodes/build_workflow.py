"""Generate the saved demo workflow so the graph opens ready-made.

ComfyUI opens on its own stock text-to-image example, which has nothing to do with this project and
shows errors because those checkpoints are not installed. This writes comfyui_nodes/demo.json: the
seven stages already wired, with the gate switch at the top.

    python comfyui_nodes/build_workflow.py [--video PATH] [--out comfyui_nodes/demo.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent

# (id, class, title, position, link-inputs [(name, type)], outputs [(name, type)], widget values)
SPEC = [
    (1, "MscSystem", "0 · System — gate ON = ours, OFF = blind baseline", (40, 60),
     [], [("cfg", "MSC_CFG"), ("system", "STRING")], [True]),
    (2, "MscLoadVideo", "1 · Load video + extract audio", (40, 260),
     [("cfg", "MSC_CFG")], [("media", "MSC_MEDIA")],
     ["data/input/demo/un_dog_barking_3doKyrCe.mp4"]),
    (3, "MscSceneUnderstanding", "2 · What is on screen", (460, 60),
     [("media", "MSC_MEDIA")], [("scene", "MSC_SCENE"), ("visible_entities", "STRING")],
     ["owlv2", 6]),
    (4, "MscTranscribe", "3 · Speech recognition", (460, 300),
     [("media", "MSC_MEDIA")], [("segments", "MSC_SEGMENTS"), ("transcript", "STRING")],
     [True]),
    (5, "MscDetectEvents", "4 · Sound detection + the two vetoes", (460, 520),
     [("media", "MSC_MEDIA")], [("events", "MSC_EVENTS"), ("heard", "STRING")],
     [0.8, 0.3, 0.05]),
    (6, "MscCrossModalGate", "5 · Cross-modal gate  ←  the contribution", (900, 240),
     [("media", "MSC_MEDIA"), ("scene", "MSC_SCENE"), ("segments", "MSC_SEGMENTS"),
      ("events", "MSC_EVENTS"), ("cfg", "MSC_CFG")],
     [("specs", "MSC_SPECS"), ("decisions", "STRING")], []),
    (7, "MscGeneratePictures", "6 · Generate pictures", (1320, 200),
     [("media", "MSC_MEDIA"), ("specs", "MSC_SPECS")],
     [("specs", "MSC_SPECS"), ("preview", "IMAGE")], ["diffusion"]),
    (8, "MscComposite", "7 · Composite beside the video", (1740, 240),
     [("media", "MSC_MEDIA"), ("specs", "MSC_SPECS"), ("events", "MSC_EVENTS")],
     [("output_video", "STRING")], []),
]

# (from node, from output index, to node, to input index)
WIRES = [
    (1, 0, 2, 0),          # cfg      -> load
    (2, 0, 3, 0),          # media    -> scene
    (2, 0, 4, 0),          # media    -> transcribe
    (2, 0, 5, 0),          # media    -> detect
    (2, 0, 6, 0),          # media    -> gate
    (3, 0, 6, 1),          # scene    -> gate
    (4, 0, 6, 2),          # segments -> gate
    (5, 0, 6, 3),          # events   -> gate
    (1, 0, 6, 4),          # cfg      -> gate
    (2, 0, 7, 0),          # media    -> generate
    (6, 0, 7, 1),          # specs    -> generate
    (2, 0, 8, 0),          # media    -> composite
    (7, 0, 8, 1),          # specs    -> composite
    (5, 0, 8, 2),          # events   -> composite
]


def build(video: str | None):
    by_id = {s[0]: s for s in SPEC}
    links, nid = [], 0
    in_link = {}
    out_links = {}
    for src, sslot, dst, dslot in WIRES:
        nid += 1
        typ = by_id[src][5][sslot][1]
        links.append([nid, src, sslot, dst, dslot, typ])
        in_link[(dst, dslot)] = nid
        out_links.setdefault((src, sslot), []).append(nid)

    nodes = []
    for order, (i, cls, title, pos, ins, outs, widgets) in enumerate(SPEC):
        w = list(widgets)
        if cls == "MscLoadVideo" and video:
            w = [video]
        nodes.append({
            "id": i, "type": cls, "title": title, "pos": list(pos),
            "size": [380, 120 + 26 * (len(ins) + len(w))],
            "flags": {}, "order": order, "mode": 0,
            "inputs": [{"name": n, "type": t, "link": in_link.get((i, k))}
                       for k, (n, t) in enumerate(ins)],
            "outputs": [{"name": n, "type": t, "links": out_links.get((i, k), []),
                         "slot_index": k} for k, (n, t) in enumerate(outs)],
            "properties": {"Node name for S&R": cls},
            "widgets_values": w,
        })

    return {
        "id": "mscproj-demo", "revision": 0,
        "last_node_id": max(s[0] for s in SPEC), "last_link_id": nid,
        "nodes": nodes, "links": links, "groups": [
            {"id": 1, "title": "the pipeline (docs/prereg_v4.md)", "bounding": [20, 10, 2110, 700],
             "color": "#3f789e", "font_size": 24, "flags": {}}
        ],
        "config": {}, "extra": {}, "version": 0.4,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=None)
    ap.add_argument("--out", default=str(_ROOT / "comfyui_nodes" / "demo.json"))
    a = ap.parse_args()
    g = build(a.video)
    Path(a.out).write_text(json.dumps(g, indent=1), encoding="utf-8")
    print(f"{len(g['nodes'])} nodes, {len(g['links'])} links -> {a.out}")


if __name__ == "__main__":
    main()
