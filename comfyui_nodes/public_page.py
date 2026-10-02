"""A simple public page on top of ComfyUI: upload a video, press the button, get the video with the sound pictures.

People never see the ComfyUI graph. The page sends the video to the ComfyUI server that runs next to it (same
cluster job, slurm/job_comfy.sh), queues the same three boxes as the MscProj_video workflow, follows the step text
the node sends, and shows the finished video and the list of pictures. One video at a time; others wait in line.

    ~/venv_gradio/bin/python comfyui_nodes/public_page.py [--comfy http://127.0.0.1:8188] [--port 7860] [--share]

With --share, Gradio prints a public https://....gradio.live link (works without the BIU VPN, only while the job runs).
"""
from __future__ import annotations

import argparse
import json
import struct
import time
import uuid
from pathlib import Path

import gradio as gr
import requests

COMFY = "http://127.0.0.1:8188"


def _workflow(name: str) -> dict:
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": name}},
        "2": {"class_type": "MscAugmentVideo", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0], "filename_prefix": "video/MscProj_public",
                                                    "format": "auto", "format.codec": "auto", "codec": "auto"}},
        "4": {"class_type": "PreviewAny", "inputs": {"source": ["2", 1]}},
    }


def run(video_path):
    if not video_path:
        yield None, "Please upload a video first.", ""
        return
    src = Path(video_path)
    with open(src, "rb") as f:
        r = requests.post(f"{COMFY}/upload/image", files={"image": (f"public_{uuid.uuid4().hex[:8]}{src.suffix or '.mp4'}", f)},
                          data={"type": "input"}, timeout=300)
    r.raise_for_status()
    name = r.json()["name"]
    client = uuid.uuid4().hex
    from websockets.sync.client import connect
    ws = connect(f"{COMFY.replace('http', 'ws', 1)}/ws?clientId={client}", max_size=None, open_timeout=30)
    r = requests.post(f"{COMFY}/prompt", json={"prompt": _workflow(name), "client_id": client}, timeout=60)
    if r.status_code != 200:
        ws.close()
        yield None, "The server did not accept the video: " + r.text[:300], ""
        return
    pid = r.json()["prompt_id"]
    t0 = time.time()
    q = requests.get(f"{COMFY}/queue", timeout=30).json()
    ahead = len(q.get("queue_running", [])) + len(q.get("queue_pending", [])) - 1
    step = f"waiting in line ({ahead} video(s) before yours)" if ahead > 0 else "starting"
    yield None, step, ""
    last = 0.0
    try:
        while True:
            try:
                msg = ws.recv(timeout=15)
            except TimeoutError:
                msg = None
            if isinstance(msg, bytes) and len(msg) > 8 and struct.unpack(">I", msg[:4])[0] == 3:   # progress text
                n = struct.unpack(">I", msg[4:8])[0]
                step = msg[8 + n:].decode("utf-8", "replace")
            elif isinstance(msg, str):
                d = json.loads(msg)
                if d.get("type") == "execution_error" and d["data"].get("prompt_id") == pid:
                    yield None, "The pipeline stopped with an error:\n" + str(d["data"].get("exception_message", ""))[:800], ""
                    return
                if d.get("type") == "executing" and d["data"].get("prompt_id") == pid and d["data"].get("node") is None:
                    break                                   # this prompt is finished
                if d.get("type") == "execution_success" and d["data"].get("prompt_id") == pid:
                    break
            if time.time() - last > 5:
                last = time.time()
                yield None, f"Working ({(time.time() - t0) / 60:.0f} min so far): {step}", ""
    finally:
        ws.close()
    h = requests.get(f"{COMFY}/history/{pid}", timeout=60).json().get(pid, {})
    outs = h.get("outputs", {})
    vids = (outs.get("3", {}).get("images") or outs.get("3", {}).get("videos") or [])
    text = "\n".join(outs.get("4", {}).get("text", []))
    if not vids:
        yield None, "Finished, but no video came back. Ask Adam to check the ComfyUI log.", text
        return
    v = vids[0]
    out = Path(f"/tmp/mscproj_public_{pid}.mp4") if not Path.home().joinpath("tmp_comfy").exists() else \
        Path.home() / "tmp_comfy" / f"mscproj_public_{pid}.mp4"
    with requests.get(f"{COMFY}/view", params={"filename": v["filename"], "subfolder": v.get("subfolder", ""),
                                               "type": v.get("type", "output")}, stream=True, timeout=300) as resp:
        resp.raise_for_status()
        with open(out, "wb") as f:
            for chunk in resp.iter_content(1 << 20):
                f.write(chunk)
    yield str(out), f"Done in {(time.time() - t0) / 60:.0f} min.", text


INTRO = """# See the sounds you cannot see

Upload a short video (best under 30 seconds). The system listens to the sound track and finds sounds whose source
is **not** on screen, for example a siren behind the camera or thunder from a dark sky. It draws a small picture for
each such sound and shows it next to the video, at the moment you hear it. Sounds whose source you can already see
get no picture.

A 15-second clip takes about 20-25 minutes, because many large AI models run one after another on one GPU.
Only one video runs at a time; if someone else is using it, you wait in line.
"""


def main():
    global COMFY
    ap = argparse.ArgumentParser()
    ap.add_argument("--comfy", default=COMFY)
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true")
    a = ap.parse_args()
    COMFY = a.comfy.rstrip("/")
    with gr.Blocks(title="See the sounds") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column():
                inp = gr.Video(label="1. Your video", sources=["upload"])
                btn = gr.Button("2. Add sound pictures", variant="primary")
                status = gr.Textbox(label="Status", interactive=False)
            with gr.Column():
                out = gr.Video(label="3. Your new video")
                drawn = gr.Textbox(label="What was drawn", lines=8, interactive=False)
        btn.click(run, inputs=inp, outputs=[out, status, drawn], concurrency_limit=1)
    demo.queue(default_concurrency_limit=1).launch(server_name="0.0.0.0", server_port=a.port, share=a.share,
                                                   allowed_paths=[str(Path.home() / "tmp_comfy"), "/tmp"])


if __name__ == "__main__":
    main()
