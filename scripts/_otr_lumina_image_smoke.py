"""_otr_lumina_image_smoke.py -- live GPU smoke for the Lumina-Image 2.0 engine.

Submits the EXACT node recipe LuminaImage2Engine._build_lumina_graph emits
(UNETLoader + CLIPLoader[type=lumina2] + VAELoader -> ModelSamplingAuraFlow ->
CLIPTextEncode pos/neg + EmptySD3LatentImage -> KSampler -> VAEDecode) to a live
headless ComfyUI (:8000), with a SaveImage terminal, so the heavy forward runs in
the EXECUTOR THREAD. Proves the real-on-GPU truth the CPU tests cannot: the lumina2
node classes exist, the three weight files load (lumina_2_model_bf16 + gemma_2_2b
text encoder + lumina2_ae VAE), and a real still mints at the OTR landscape canvas.

This mirrors the proven render_flux_batch /prompt pattern. The engine's own
render_image drives the same recipe in-process via wrapper_bridge; the recipe is
the only GPU-new surface (the adapter shape is byte-identical to the shipped
flux_gen1), so a green recipe render is the promotion evidence.

Usage:
  python scripts/_otr_lumina_image_smoke.py [--width 1216] [--height 832]
      [--steps 30] [--cfg 4.0] [--shift 6.0] [--seed 7]

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

# The default lumina split-file weights (same basenames the engine defaults to;
# resolved by ComfyUI folder_paths from C:/ComfyUI-Models/{diffusion_models,
# text_encoders,vae}). Env-overridable so the operator can point elsewhere.
UNET = os.path.basename(os.environ.get("OTR_LUMINA_CKPT", "")
                        or "lumina_2_model_bf16.safetensors")
CLIP = os.path.basename(os.environ.get("OTR_LUMINA_CLIP", "")
                        or "gemma_2_2b_fp16.safetensors")
VAE = os.path.basename(os.environ.get("OTR_LUMINA_VAE", "")
                       or "lumina2_ae.safetensors")

#: A dedicated smoke episode so the still lands inside the output-tree contract
#: (otr/episodes/<ep>/stills/), routed through the single _otr_paths authority.
SMOKE_EPISODE = "lumina_smoke"

_PROMPT = ("a 1940s science-fiction radio broadcast studio, brass microphone, "
           "warm tungsten key light, deep cinematic shadows, volumetric haze, "
           "35mm film still, rich saturated color")


def build_graph(*, width, height, steps, cfg, shift, seed, prefix):
    """The Lumina-2 /prompt graph -- a 1:1 mirror of the engine recipe + a
    SaveImage terminal so the still lands on disk for the file-check."""
    return {
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": UNET, "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": CLIP, "type": "lumina2"}},
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
                                "scheduler": "normal", "denoise": 1.0,
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
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--cfg", type=float, default=4.0)
    ap.add_argument("--shift", type=float, default=6.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args(argv)

    # Single path authority: the still lands in otr/episodes/<ep>/stills/. The
    # SaveImage filename_prefix is relative to ComfyUI's output root.
    stills_dir = P.otr_stills_dir(SMOKE_EPISODE)
    glob_dir = str(stills_dir)
    prefix = (stills_dir / "lumina_smoke").relative_to(
        P.comfy_output_dir()).as_posix()
    started = time.time()

    print("[lumina-smoke] %s  %dx%d steps=%d cfg=%.2f shift=%.2f seed=%d"
          % (COMFYUI_URL, args.width, args.height, args.steps, args.cfg,
             args.shift, args.seed), flush=True)
    print("[lumina-smoke] unet=%s clip=%s vae=%s" % (UNET, CLIP, VAE), flush=True)

    graph = build_graph(width=args.width, height=args.height, steps=args.steps,
                        cfg=args.cfg, shift=args.shift, seed=args.seed,
                        prefix=prefix)
    prompt_id = submit_prompt(graph)
    print("[lumina-smoke] QUEUED prompt_id=%s" % prompt_id, flush=True)
    status, err = poll_history(prompt_id, timeout_s=args.timeout)
    wall = time.time() - started
    print("[lumina-smoke] history status=%s (%.1fs) %s"
          % (status, wall, err), flush=True)

    landed = []
    if os.path.isdir(glob_dir):
        landed = [f for f in os.listdir(glob_dir)
                  if f.startswith("lumina_smoke")
                  and f.endswith(".png")
                  and os.path.getmtime(os.path.join(glob_dir, f)) >= started - 2]
    if status == "SUCCESS" and landed:
        newest = max(landed,
                     key=lambda f: os.path.getmtime(os.path.join(glob_dir, f)))
        size = os.path.getsize(os.path.join(glob_dir, newest))
        print("[lumina-smoke] PASS: minted %s (%d bytes) in %.1fs"
              % (newest, size, wall), flush=True)
        return 0
    print("[lumina-smoke] FAIL: status=%s landed=%s" % (status, landed),
          flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
