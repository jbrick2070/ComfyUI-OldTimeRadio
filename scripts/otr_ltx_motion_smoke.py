"""otr_ltx_motion_smoke -- BARE-BONES isolated LTX i2v motion smoke (2026-06-12).

One scene still in -> one short clip out (SaveWEBM, so you can EYEBALL MOTION),
NO audio / Flux / HuMo / episode pipeline. ~1-2 min per render so we can sweep
the LTX motion knobs fast (operator directive: stop chasing motion through
24-min full episodes).

Topology MIRRORS eng_ltx_video._build_graph_i2v (the real engine path), i2v
ConditionOnly anchored on the still:
  CheckpointLoaderSimple(v0.9) + CLIPLoader(t5xxl, ltxv) -> CLIPTextEncode x2
  + LoadImage + EmptyLTXVLatentVideo -> LTXVImgToVideoConditionOnly(strength)
  -> LTXVConditioning -> KSampler(sampler_name, steps, cfg) -> VAEDecode -> SaveWEBM

Knobs (args) so we sweep ONE at a time and save each clip tagged:
  --sampler  euler | euler_cfg_pp | ...   (the 6/1+6/5 secret sauce = euler_cfg_pp)
  --length   frames (8n+1)                (6/1 used 257-305; today 169)
  --strength 1.0 | 0.75 | 0.6             (keep 1.0 -- stills IN the video)
  --cfg      3.0 | 4.0
  --steps    30
  --width/--height   default 768x448 (fast); use 1472x832 to match the episode

Run AFTER the server is up on :8000 with OTR_ENABLE_LTX_VIDEO=1.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import requests

URL = "http://127.0.0.1:8000"
CKPT = "ltx-video-2b-v0.9.safetensors"
T5 = "t5xxl_fp16.safetensors"
STILL = "otr_ltx_smoke_still.png"   # in the comfy input dir
# announcer motion template (BUG-LOCAL-112, <=188 chars), motion-only.
MOTION_PROMPT = ("Continuous shot, same console throughout. Tuning dial needle "
                 "sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille "
                 "trembles. Dust motes drift. Slow handheld dolly forward.")
NEG = "low quality, worst quality, blurry, distorted, watermark, text, static"


def build_workflow(a) -> dict:
    return {
        "1": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": a.ckpt}},
        "2": {"class_type": "CLIPLoader",
              "inputs": {"clip_name": a.t5, "type": "ltxv"}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": a.prompt, "clip": ["2", 0]}},
        "4": {"class_type": "CLIPTextEncode",
              "inputs": {"text": a.negative, "clip": ["2", 0]}},
        "5": {"class_type": "LoadImage", "inputs": {"image": a.still}},
        "6": {"class_type": "EmptyLTXVLatentVideo",
              "inputs": {"width": a.width, "height": a.height,
                         "length": a.length, "batch_size": 1}},
        "7": {"class_type": "LTXVImgToVideoConditionOnly",
              "inputs": {"vae": ["1", 2], "image": ["5", 0],
                         "latent": ["6", 0], "strength": a.strength}},
        "8": {"class_type": "LTXVConditioning",
              "inputs": {"positive": ["3", 0], "negative": ["4", 0],
                         "frame_rate": float(a.fps)}},
        "9": {"class_type": "KSampler",
              "inputs": {"seed": a.seed, "steps": a.steps, "cfg": a.cfg,
                         "sampler_name": a.sampler, "scheduler": a.scheduler,
                         "denoise": 1.0, "model": ["1", 0],
                         "positive": ["8", 0], "negative": ["8", 1],
                         "latent_image": ["7", 0]}},
        "10": {"class_type": "VAEDecode",
               "inputs": {"samples": ["9", 0], "vae": ["1", 2]}},
        "11": {"class_type": "SaveWEBM",
               "inputs": {"images": ["10", 0],
                          "filename_prefix": f"otr_ltxmotion/{a.tag}",
                          "codec": "vp9", "fps": float(a.fps), "crf": 18.0}},
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sampler", default="euler")
    p.add_argument("--scheduler", default="normal")
    p.add_argument("--length", type=int, default=97)
    p.add_argument("--strength", type=float, default=1.0)
    p.add_argument("--cfg", type=float, default=3.0)
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--height", type=int, default=448)
    p.add_argument("--fps", type=int, default=25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--ckpt", default=CKPT)
    p.add_argument("--t5", default=T5)
    p.add_argument("--still", default=STILL)
    p.add_argument("--prompt", default=MOTION_PROMPT)
    p.add_argument("--negative", default=NEG)
    p.add_argument("--tag", default="smoke")
    a = p.parse_args()

    try:
        requests.get(f"{URL}/system_stats", timeout=5)
    except Exception as e:
        print(f"FAIL: server not reachable: {e}")
        return 1

    wf = build_workflow(a)
    print(f"[smoke] tag={a.tag} sampler={a.sampler} steps={a.steps} cfg={a.cfg} "
          f"strength={a.strength} length={a.length} {a.width}x{a.height} "
          f"prompt_len={len(a.prompt)}", flush=True)
    r = requests.post(f"{URL}/prompt", json={"prompt": wf}, timeout=15).json()
    if "error" in r or r.get("node_errors"):
        print(f"FAIL: rejected: {json.dumps(r.get('node_errors', r.get('error')), indent=2)[:1200]}")
        return 1
    pid = r.get("prompt_id")
    print(f"[smoke] queued prompt_id={pid}", flush=True)
    t0 = time.time()
    deadline = t0 + 900
    while time.time() < deadline:
        try:
            entry = requests.get(f"{URL}/history/{pid}", timeout=10).json().get(pid)
        except Exception:
            entry = None
        if entry:
            st = entry.get("status", {})
            ss = st.get("status_str", "")
            errs = [m for m in st.get("messages", []) if m[0] == "execution_error"]
            if errs:
                print(f"FAIL: execution_error: {json.dumps(errs, indent=2)[:1500]}")
                return 1
            if ss in ("success", "error") or entry.get("outputs"):
                if ss == "error":
                    print(f"FAIL: status=error: {st.get('messages')}")
                    return 1
                outs = entry.get("outputs", {})
                files = []
                for nid, o in outs.items():
                    for key in ("images", "gifs", "video", "files"):
                        for f in (o.get(key) or []):
                            if isinstance(f, dict):
                                files.append(f"{f.get('subfolder','')}/{f.get('filename','')}")
                dt = time.time() - t0
                print(f"PASS: rendered in {dt:.0f}s -> {files}", flush=True)
                return 0
        time.sleep(3)
        sys.stdout.write("."); sys.stdout.flush()
    print("\nFAIL: timed out")
    return 1


if __name__ == "__main__":
    sys.exit(main())
