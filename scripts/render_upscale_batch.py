"""
render_upscale_batch.py -- post-HuMo SeedVR2 video upscale orchestrator.

After a HuMo episode mp4 has been stitched (CreateVideo + SaveVideo at
~480x832 / 25 fps), pipe the whole video through SeedVR2 to produce a
1080p delivery copy. SeedVR2 keeps temporal coherence across batches
(it's a video upscaler, not a per-frame ESRGAN), and it ships with
SageAttention support that runs fast on RTX 5080 Blackwell.

Pattern matches render_flux_batch.py:
  - build a single ComfyUI API graph
  - POST one /prompt
  - poll /history until complete
  - confirm the upscaled mp4 landed in ComfyUI's output dir

Usage (production path):
    python scripts/render_upscale_batch.py \\
        --input-mp4  C:/.../signal_lost_episode.mp4 \\
        --output-prefix signal_lost_episode_1080p \\
        --resolution 1080 \\
        --comfyui-url http://localhost:8000

If --output-prefix is omitted, derives <input_basename>_1080p.

Notes:
  - VRAM ceiling: SeedVR2 3B fp16 + VAE fp16 + a 480x832 input batch sits
    around 10-12 GB peak with blocks_to_swap=32 + tile_decoder. Should
    fit on 16 GB Blackwell. If OOM is observed, raise blocks_to_swap to
    48 (more aggressive offload).
  - Timing: ~3-5 min per 7s clip on a 5080. For a 5-min episode that's
    roughly 30-50 min wall-clock, vs 7.5h HuMo render. Cheap polish.
  - This is intentionally NOT an in-canvas workflow node. SeedVR2 +
    HuMo + FLUX cannot coexist in VRAM; running upscale post-hoc lets
    HuMo unload cleanly first.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Defaults (mirror SeedVR2_HD_video_upscale.json widget values)
# ---------------------------------------------------------------------------

DEFAULT_DIT_MODEL = "seedvr2_ema_3b_fp16.safetensors"
DEFAULT_VAE_MODEL = "ema_vae_fp16.safetensors"
DEFAULT_RESOLUTION = 1080
DEFAULT_BATCH_SIZE = 33          # 4n+1, matches example workflow
DEFAULT_TEMPORAL_OVERLAP = 3
DEFAULT_BLOCKS_TO_SWAP = 32      # aggressive offload for 16 GB ceiling
DEFAULT_COLOR_CORRECTION = "lab"
DEFAULT_SEED = 42


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-HuMo SeedVR2 1080p upscale orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python scripts/render_upscale_batch.py \\\n"
            "      --input-mp4 C:/.../signal_lost_long_goodbye.mp4 \\\n"
            "      --comfyui-url http://localhost:8000\n"
        ),
    )
    p.add_argument("--input-mp4", required=True, type=Path,
                   help="Path to the source mp4 (HuMo final output).")
    p.add_argument("--output-prefix", default=None,
                   help="filename_prefix passed to SaveVideo. "
                        "Defaults to '<input_basename>_1080p'.")
    p.add_argument("--comfyui-url", default="http://localhost:8000",
                   help="ComfyUI server URL.")
    p.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION,
                   help=f"Target shortest-edge resolution (default {DEFAULT_RESOLUTION}).")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
                   help=f"SeedVR2 batch_size, must be 4n+1 "
                        f"(default {DEFAULT_BATCH_SIZE}).")
    p.add_argument("--temporal-overlap", type=int,
                   default=DEFAULT_TEMPORAL_OVERLAP,
                   help=f"Frames overlapping between batches "
                        f"(default {DEFAULT_TEMPORAL_OVERLAP}).")
    p.add_argument("--blocks-to-swap", type=int,
                   default=DEFAULT_BLOCKS_TO_SWAP,
                   help=f"DiT block-swap count for VRAM offload "
                        f"(default {DEFAULT_BLOCKS_TO_SWAP}; raise to 48 "
                        f"if OOM).")
    p.add_argument("--dit-model", default=DEFAULT_DIT_MODEL,
                   help=f"SeedVR2 DiT checkpoint name "
                        f"(default {DEFAULT_DIT_MODEL!r}).")
    p.add_argument("--vae-model", default=DEFAULT_VAE_MODEL,
                   help=f"SeedVR2 VAE checkpoint name "
                        f"(default {DEFAULT_VAE_MODEL!r}).")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--color-correction", default=DEFAULT_COLOR_CORRECTION,
                   choices=["lab", "wavelet", "wavelet_adaptive", "hsv",
                            "adain", "none"],
                   help=f"Frame-batch color correction "
                        f"(default {DEFAULT_COLOR_CORRECTION!r}).")
    p.add_argument("--prompt-timeout", type=float, default=3600.0,
                   help="Max wall-clock per /prompt completion (default 1h).")
    p.add_argument("--poll-interval", type=float, default=5.0,
                   help="Seconds between /history polls.")
    p.add_argument("--comfy-output-dir", type=Path,
                   default=Path(r"C:/Users/jeffr/Documents/ComfyUI/output"),
                   help="ComfyUI output directory (used for promotion).")
    p.add_argument("--episodes-dir", type=Path,
                   default=Path(r"C:/Users/jeffr/Documents/ComfyUI/output/otr/episodes"),
                   help="Flat directory for final-good deliverables. After "
                        "SeedVR2 finishes, the 1080p mp4 (and any sibling "
                        ".vtt) is copied here with a clean filename so "
                        "media servers (Plex/Jellyfin/Emby) can loop on "
                        "one folder. Pass --no-promote to skip.")
    p.add_argument("--no-promote", action="store_true",
                   help="Skip the copy-to-episodes-dir step.")
    p.add_argument("--episode-name", default=None,
                   help="Filename stem in --episodes-dir. Defaults to "
                        "<input_basename> with '_framed' stripped.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the API graph and exit; do not submit.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_upscale_prompt(
    *,
    input_video_filename: str,
    output_prefix: str,
    resolution: int,
    batch_size: int,
    temporal_overlap: int,
    blocks_to_swap: int,
    dit_model: str,
    vae_model: str,
    seed: int,
    color_correction: str,
) -> dict[str, Any]:
    """ComfyUI API graph that mirrors SeedVR2_HD_video_upscale.json.

    Shape:
        load_video         → get_components (images, audio, fps)
        dit_loader         ─┐
        vae_loader         ─┤
        get_components.img ─┼→ seedvr_upscaler → create_video → save_video
        get_components.audio→ create_video.audio
        get_components.fps  → create_video.fps
    """
    # SeedVR2VideoUpscaler widgets order:
    #   [seed, "fixed", resolution, max_resolution, batch_size,
    #    uniform_batch_size, color_correction, temporal_overlap,
    #    prepend_frames, input_noise_scale, latent_noise_scale,
    #    offload_device, enable_debug]
    upscaler_widgets = [
        seed, "fixed", resolution, 0, batch_size,
        True, color_correction, temporal_overlap,
        0, 0, 0, "cpu", False,
    ]

    # SeedVR2LoadDiTModel widgets:
    #   [model, device, blocks_to_swap, swap_io_components,
    #    offload_device, cache_model, attention_mode]
    dit_widgets = [
        dit_model, "cuda:0", blocks_to_swap, False,
        "cpu", False, "sdpa",
    ]

    # SeedVR2LoadVAEModel widgets:
    #   [model, device, encode_tiled, encode_tile_size, encode_tile_overlap,
    #    decode_tiled, decode_tile_size, decode_tile_overlap, tile_debug,
    #    offload_device, cache_model]
    vae_widgets = [
        vae_model, "cuda:0", True, 1024, 128,
        True, 768, 128, "false", "cpu", False,
    ]

    return {
        "load_video": {
            "class_type": "LoadVideo",
            "inputs": {"file": input_video_filename},
            "_meta": {"title": "Load source mp4"},
        },
        "get_components": {
            "class_type": "GetVideoComponents",
            "inputs": {"video": ["load_video", 0]},
            "_meta": {"title": "Split images / audio / fps"},
        },
        "dit_loader": {
            "class_type": "SeedVR2LoadDiTModel",
            "inputs": {
                "model": dit_widgets[0],
                "device": dit_widgets[1],
                "blocks_to_swap": dit_widgets[2],
                "swap_io_components": dit_widgets[3],
                "offload_device": dit_widgets[4],
                "cache_model": dit_widgets[5],
                "attention_mode": dit_widgets[6],
            },
            "_meta": {"title": f"SeedVR2 DiT ({dit_model})"},
        },
        "vae_loader": {
            "class_type": "SeedVR2LoadVAEModel",
            "inputs": {
                "model": vae_widgets[0],
                "device": vae_widgets[1],
                "encode_tiled": vae_widgets[2],
                "encode_tile_size": vae_widgets[3],
                "encode_tile_overlap": vae_widgets[4],
                "decode_tiled": vae_widgets[5],
                "decode_tile_size": vae_widgets[6],
                "decode_tile_overlap": vae_widgets[7],
                "tile_debug": vae_widgets[8],
                "offload_device": vae_widgets[9],
                "cache_model": vae_widgets[10],
            },
            "_meta": {"title": f"SeedVR2 VAE ({vae_model})"},
        },
        "upscaler": {
            "class_type": "SeedVR2VideoUpscaler",
            "inputs": {
                "image": ["get_components", 0],
                "dit": ["dit_loader", 0],
                "vae": ["vae_loader", 0],
                "seed": upscaler_widgets[0],
                "resolution": upscaler_widgets[2],
                "max_resolution": upscaler_widgets[3],
                "batch_size": upscaler_widgets[4],
                "uniform_batch_size": upscaler_widgets[5],
                "color_correction": upscaler_widgets[6],
                "temporal_overlap": upscaler_widgets[7],
                "prepend_frames": upscaler_widgets[8],
                "input_noise_scale": upscaler_widgets[9],
                "latent_noise_scale": upscaler_widgets[10],
                "offload_device": upscaler_widgets[11],
                "enable_debug": upscaler_widgets[12],
            },
            "_meta": {"title": f"Upscale to {resolution}p"},
        },
        "create_video": {
            "class_type": "CreateVideo",
            "inputs": {
                "images": ["upscaler", 0],
                "audio": ["get_components", 1],
                "fps": ["get_components", 2],
            },
            "_meta": {"title": "Stitch frames + audio (preserve fps)"},
        },
        "save_video": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["create_video", 0],
                "filename_prefix": output_prefix,
                "format": "auto",
                "codec": "auto",
            },
            "_meta": {"title": f"Save {output_prefix}.mp4"},
        },
    }


# ---------------------------------------------------------------------------
# ComfyUI HTTP client (stdlib only)
# ---------------------------------------------------------------------------

class ComfyClient:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.client_id = str(uuid.uuid4())

    def submit_prompt(self, prompt: dict) -> str:
        payload = {"prompt": prompt, "client_id": self.client_id}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["prompt_id"]

    def wait_for_completion(self, prompt_id: str,
                            timeout_s: float, poll_interval_s: float) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            try:
                with urllib.request.urlopen(
                    f"{self.base_url}/history/{prompt_id}", timeout=30
                ) as resp:
                    hist = json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    time.sleep(poll_interval_s)
                    continue
                raise
            if hist and prompt_id in hist:
                return hist[prompt_id]
            time.sleep(poll_interval_s)
        raise TimeoutError(f"prompt_id={prompt_id} did not complete within "
                           f"{timeout_s:.0f}s")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    if not args.input_mp4.exists():
        print(f"FATAL: input mp4 not found: {args.input_mp4}", file=sys.stderr)
        return 2

    # ComfyUI's LoadVideo widget takes a filename (must already live in
    # ComfyUI's input directory) OR a relative path. Pass the basename;
    # if the user has the input mp4 elsewhere, ComfyUI 0.19+ accepts an
    # absolute path string in the file widget.
    input_filename = str(args.input_mp4)

    # 2026-04-26: outputs nested under output/otr/videos/. Single OTR
    # parent keeps the ComfyUI output/ root tidy.
    output_prefix = (
        args.output_prefix
        or f"otr/videos/{args.input_mp4.stem}_{args.resolution}p"
    )

    if (args.batch_size - 1) % 4 != 0:
        print(f"FATAL: --batch-size must be 4n+1 (got {args.batch_size}). "
              f"Try {args.batch_size + ((5 - args.batch_size) % 4)}.",
              file=sys.stderr)
        return 3

    api_prompt = build_upscale_prompt(
        input_video_filename=input_filename,
        output_prefix=output_prefix,
        resolution=args.resolution,
        batch_size=args.batch_size,
        temporal_overlap=args.temporal_overlap,
        blocks_to_swap=args.blocks_to_swap,
        dit_model=args.dit_model,
        vae_model=args.vae_model,
        seed=args.seed,
        color_correction=args.color_correction,
    )

    print(f"[plan] SeedVR2 upscale {args.input_mp4.name} -> "
          f"{output_prefix}.mp4 @ {args.resolution}p")
    print(f"  dit_model       = {args.dit_model}")
    print(f"  vae_model       = {args.vae_model}")
    print(f"  batch_size      = {args.batch_size} (4n+1)")
    print(f"  temporal_overlap= {args.temporal_overlap}")
    print(f"  blocks_to_swap  = {args.blocks_to_swap}")
    print(f"  color_correction= {args.color_correction}")
    print(f"  seed            = {args.seed}")

    if args.dry_run:
        print("[dry-run] graph not submitted.")
        print(json.dumps(api_prompt, indent=2))
        return 0

    client = ComfyClient(args.comfyui_url)
    started = time.time()
    try:
        prompt_id = client.submit_prompt(api_prompt)
    except urllib.error.HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        print(f"FATAL: SUBMIT FAILED: {e}", file=sys.stderr)
        if body:
            print(f"  response_body: {body[:1500]}", file=sys.stderr)
        return 4
    except Exception as e:
        print(f"FATAL: cannot reach ComfyUI at {args.comfyui_url}: {e}",
              file=sys.stderr)
        return 4

    print(f"[submit] prompt_id={prompt_id}")
    print(f"[wait] timeout={args.prompt_timeout:.0f}s, "
          f"poll_every={args.poll_interval:.0f}s")
    try:
        client.wait_for_completion(
            prompt_id,
            timeout_s=args.prompt_timeout,
            poll_interval_s=args.poll_interval,
        )
    except TimeoutError as e:
        print(f"TIMEOUT: {e}", file=sys.stderr)
        return 5

    wall = time.time() - started
    print(f"[done] upscale complete in {wall:.1f}s "
          f"({wall / 60.0:.1f} min)")
    print(f"[hint] check ComfyUI output dir for "
          f"{output_prefix}_NNNNN.mp4")

    # ---- promote: copy final mp4 (and sibling .vtt) to flat episodes dir ----
    if args.no_promote:
        return 0
    import shutil  # local import keeps stdlib-only top-level
    # Locate the actual upscaled mp4 (ComfyUI appends _NNNNN_ before .mp4)
    expected_dir = (
        args.comfy_output_dir / Path(output_prefix).parent
    )
    expected_stem = Path(output_prefix).name
    candidates = sorted(
        expected_dir.glob(f"{expected_stem}_*.mp4"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        print(f"WARN: no upscaled mp4 found under {expected_dir} matching "
              f"{expected_stem}_*.mp4 -- promotion skipped", file=sys.stderr)
        return 0
    src_mp4 = candidates[0]

    # Clean filename: strip "_framed" + "_<resolution>p" + ComfyUI's "_NNNNN_"
    clean_stem = (
        args.episode_name
        or args.input_mp4.stem.replace("_framed", "").replace("_FRAMED", "")
    )
    args.episodes_dir.mkdir(parents=True, exist_ok=True)
    dst_mp4 = args.episodes_dir / f"{clean_stem}.mp4"
    shutil.copy2(src_mp4, dst_mp4)
    print(f"[promote] {dst_mp4}")

    # Look for the .vtt sidecar that render_episode_concat.py wrote next
    # to the un-framed source. Canonical path pattern (post BUG-LOCAL-075):
    #   <comfy-out>/otr/episodes/<episode_id>/<episode_id>.vtt
    # Legacy: <comfy-out>/otr_videos/<episode_id>/<episode_id>.vtt
    # The input mp4 is most likely <episode_id>_framed.mp4 in the same dir,
    # so the sibling-derived candidates below resolve either layout without
    # hardcoding the parent dir name.
    src_vtt_candidates = [
        args.input_mp4.with_suffix(".vtt"),
        args.input_mp4.parent / f"{args.input_mp4.stem.replace('_framed', '')}.vtt",
    ]
    for src_vtt in src_vtt_candidates:
        if src_vtt.exists():
            dst_vtt = args.episodes_dir / f"{clean_stem}.vtt"
            shutil.copy2(src_vtt, dst_vtt)
            print(f"[promote] {dst_vtt}")
            break
    else:
        print(f"[promote] note: no sibling .vtt found; mov_text track in "
              f"the mp4 itself still carries embedded subs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
