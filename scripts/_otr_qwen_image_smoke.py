"""_otr_qwen_image_smoke.py -- live GPU smoke for the Qwen-Image (GGUF) engine.

Submits the EXACT node recipe QwenImageEngine._build_qwen_graph emits
(UnetLoaderGGUF + CLIPLoader[type=qwen_image] + VAELoader -> ModelSamplingAuraFlow
-> CLIPTextEncode pos/neg + EmptySD3LatentImage -> KSampler -> VAEDecode) to a live
headless ComfyUI (:8000), with a SaveImage terminal, so the heavy forward runs in
the EXECUTOR THREAD. Proves the real-on-GPU truth the CPU tests cannot AND measures
the per-quant resident peak (qwen 20B is TIGHT under the 14.5 GB single-resident
ceiling -- the smoke is the gate that decides promotion).

The companion of _otr_lumina_image_smoke.py (same /prompt pattern). PRECONDITION:
the operator has downloaded a Qwen-Image GGUF + the qwen-2.5-vl TE + the qwen VAE,
and OTR_QWEN_IMAGE_GGUF points at the .gguf (this script fails loud if it is unset
or missing, since the weights are not yet on disk as of the build).

Usage (after the still-only z_image soak frees the GPU + the box is reset):
  set OTR_QWEN_IMAGE_GGUF=C:\ComfyUI-Models\unet\qwen-image-Q4_K_S.gguf
  python scripts/_otr_qwen_image_smoke.py [--width 1216] [--height 832]
      [--steps 20] [--cfg 2.5] [--shift 3.0] [--seed 7]

Watch the server log for the resident peak; PROMOTE only if it stays <= 14500 MB.
Exit 0 = a real PNG landed. UTF-8, no BOM, ASCII-only source.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from otr_api import COMFYUI_URL, submit_prompt, poll_history  # noqa: E402
from nodes import _otr_paths as P  # noqa: E402  (the one OTR path authority)

UNET = os.path.basename(os.environ.get("OTR_QWEN_IMAGE_GGUF", "") or "")
CLIP = os.path.basename(os.environ.get("OTR_QWEN_IMAGE_CLIP", "")
                        or "qwen_2.5_vl_7b_fp8_scaled.safetensors")
VAE = os.path.basename(os.environ.get("OTR_QWEN_IMAGE_VAE", "")
                       or "qwen_image_vae.safetensors")

SMOKE_EPISODE = "qwen_image_smoke"

_PROMPT = ("a 1940s science-fiction radio broadcast studio, brass microphone, "
           "warm tungsten key light, deep cinematic shadows, volumetric haze, "
           "35mm film still, rich saturated color")


def build_graph(*, width, height, steps, cfg, shift, seed, prefix):
    """The Qwen-Image GGUF /prompt graph -- a 1:1 mirror of the engine recipe + a
    SaveImage terminal so the still lands on disk for the file-check."""
    return {
        "unet": {"class_type": "UnetLoaderGGUF",
                 "inputs": {"unet_name": UNET}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "qwen_image"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": VAE}},
        "sampling": {"class_type": "ModelSamplingAuraFlow",
                     "inputs": {"model": ["unet", 0], "shift": float(shift)}},
        "pos": {"class_type": "CLIPTextEncode",
                "inputs": {"text": _PROMPT, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode",
                "inputs": {"text": "", "clip": ["clip", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": int(width), "height": int(height),
                              "batch_size": 1}},
        "ksampler": {"class_type": "KSampler",
                     "inputs": {"seed": int(seed), "steps": int(steps),
                                "cfg": float(cfg), "sampler_name": "euler",
                                "scheduler": "simple", "denoise": 1.0,
                                "model": ["sampling", 0], "positive": ["pos", 0],
                                "negative": ["neg", 0],
                                "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode",
                   "inputs": {"samples": ["ksampler", 0], "vae": ["vae", 0]}},
        "save": {"class_type": "SaveImage",
                 "inputs": {"filename_prefix": prefix, "images": ["decode", 0]}},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=1216)
    ap.add_argument("--height", type=int, default=832)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--cfg", type=float, default=2.5)
    ap.add_argument("--shift", type=float, default=3.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=1200)
    args = ap.parse_args(argv)

    ckpt = os.environ.get("OTR_QWEN_IMAGE_GGUF", "").strip()
    if not ckpt or not os.path.isfile(ckpt):
        print("[qwen-smoke] FAIL: OTR_QWEN_IMAGE_GGUF unset or not a file (%r). "
              "Download a Qwen-Image GGUF + set the env first." % ckpt, flush=True)
        return 2

    stills_dir = P.otr_stills_dir(SMOKE_EPISODE)
    glob_dir = str(stills_dir)
    prefix = (stills_dir / "qwen_smoke").relative_to(
        P.comfy_output_dir()).as_posix()
    started = time.time()

    print("[qwen-smoke] %s  %dx%d steps=%d cfg=%.2f shift=%.2f seed=%d"
          % (COMFYUI_URL, args.width, args.height, args.steps, args.cfg,
             args.shift, args.seed), flush=True)
    print("[qwen-smoke] unet=%s clip=%s vae=%s" % (UNET, CLIP, VAE), flush=True)

    graph = build_graph(width=args.width, height=args.height, steps=args.steps,
                        cfg=args.cfg, shift=args.shift, seed=args.seed,
                        prefix=prefix)
    prompt_id = submit_prompt(graph)
    print("[qwen-smoke] QUEUED prompt_id=%s" % prompt_id, flush=True)
    status, err = poll_history(prompt_id, timeout_s=args.timeout)
    wall = time.time() - started
    print("[qwen-smoke] history status=%s (%.1fs) %s"
          % (status, wall, err[:300]), flush=True)

    landed = []
    if os.path.isdir(glob_dir):
        landed = [f for f in os.listdir(glob_dir)
                  if f.startswith("qwen_smoke")
                  and f.endswith(".png")
                  and os.path.getmtime(os.path.join(glob_dir, f)) >= started - 2]
    if status == "SUCCESS" and landed:
        newest = max(landed,
                     key=lambda f: os.path.getmtime(os.path.join(glob_dir, f)))
        size = os.path.getsize(os.path.join(glob_dir, newest))
        print("[qwen-smoke] PASS: minted %s (%d bytes) in %.1fs -- now CONFIRM "
              "the server-log resident peak <= 14500 MB before promoting"
              % (newest, size, wall), flush=True)
        return 0
    print("[qwen-smoke] FAIL: status=%s landed=%s" % (status, landed), flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
