"""ltx_motion_smoke.py -- BUG-LOCAL-112 verification smoke.

Purpose: render ONE LTX clip from a single radio_bookend still + one
motion prompt, using ONLY stock LTX nodes (no OTR_BatchLTXRender, no
ledger plumbing, no HuMo, no FLUX, no audio mux). Submits the workflow
graph directly to ComfyUI's HTTP /prompt endpoint and polls /history
until it lands. Total iteration: ~30-60s per test.

Output: one MP4 + one animated WEBP. The WEBP plays inline in the
ComfyUI node UI so motion is visible without external tools. The MP4
is the canonical artifact for MAD-based motion measurement against
the BUG-112 baseline.

Usage:
    python scripts\\ltx_motion_smoke.py
    python scripts\\ltx_motion_smoke.py --role announcer
    python scripts\\ltx_motion_smoke.py --prompt "Custom prompt here"
    python scripts\\ltx_motion_smoke.py --still C:\\path\\to\\radio_bookend.png
    python scripts\\ltx_motion_smoke.py --seed 1234 --length 121

Prereqs:
    ComfyUI Desktop running on http://127.0.0.1:8000.
    LTX 2B v0.9 .safetensors + T5-XXL fp16 already in models/checkpoints.
    A radio_bookend.png already on disk -- by default the script picks
    the most recent one from output/otr/episodes/.

License: MIT (matches OTR repo). No GPL nodes used (no VHS).

Stock-only node set:
    LowVRAMCheckpointLoader, CLIPLoader (T5), LoadImage,
    CLIPTextEncode, LTXVConditioning, EmptyLTXVLatentVideo,
    LTXVImgToVideoConditionOnly, ManualSigmas, KSamplerSelect,
    RandomNoise, CFGGuider, SamplerCustomAdvanced, VAEDecode,
    SaveAnimatedWEBP, CreateVideo, SaveVideo.

If your ComfyUI install lacks one of these, the script returns the
HTTP error verbatim so the missing node is obvious.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

COMFYUI_URL = os.environ.get("COMFYUI_URL", "http://127.0.0.1:8000")

# BUG-LOCAL-112 motion-centric prompts, mirroring nodes/batch_ltx_render._PROMPT_BY_ROLE.
PROMPTS = {
    "announcer": (
        "Continuous shot, same console throughout. Tuning dial needle "
        "sweeps rhythmically. Vacuum tubes pulse. Brass speaker grille "
        "trembles. Dust motes drift. Slow handheld dolly forward."
    ),
    "music_open": (
        "Continuous shot, same console throughout. Dial whip-pans across "
        "frequencies. Tube filaments ignite from cold to white-hot. "
        "Speaker grille vibrates aggressively. Dynamic dolly push forward."
    ),
    "music_close": (
        "Continuous shot, same console throughout. Dial settles. Tube "
        "filaments cool from white through deep amber. Smoke trails "
        "from cooling tubes. Slow dolly pull back."
    ),
    "music_inter": (
        "Continuous shot, same console throughout. Dial steady, glowing. "
        "Oscilloscope dances to the rhythm. VU meters bounce. Tubes "
        "pulse with the bass. Slow orbit around the speaker."
    ),
    "sfx": (
        "Continuous shot, same console throughout. Snap zoom on the "
        "dial as needle spikes hard. Tubes surge with electric arcs. "
        "Speaker grille rattles violently. Quick whip-pan to the dial."
    ),
}

# OTR LTX render constants (mirrors nodes/batch_ltx_render).
LTX_FPS = 25
LTX_WIDTH = 832
LTX_HEIGHT = 480
LTX_CFG = 1.0
LTX_I2V_STRENGTH = 0.75
LTX_DISTILLED_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_CHECKPOINT = "ltx-video-2b-v0.9.safetensors"
T5_CLIP = "t5xxl_fp16.safetensors"


def _find_default_still() -> str:
    """Pick the most recent radio_bookend.png from output/otr/episodes/."""
    eps_root = Path(
        os.path.expanduser("~"), "Documents", "ComfyUI",
        "output", "otr", "episodes",
    )
    if not eps_root.exists():
        raise FileNotFoundError(f"No episodes dir at {eps_root}")
    candidates = list(eps_root.glob("**/stills/radio_bookend_*.png"))
    if not candidates:
        raise FileNotFoundError(
            f"No radio_bookend_*.png found under {eps_root}. Run a "
            "full episode first, or pass --still PATH."
        )
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return str(candidates[0])


def _ensure_still_in_input_dir(still_path: str) -> str:
    """Copy the still into ComfyUI input/ so LoadImage can find it.
    Returns the basename (what LoadImage's image widget expects).
    """
    src = Path(still_path)
    if not src.exists():
        raise FileNotFoundError(f"still not found: {src}")
    input_dir = Path(
        os.path.expanduser("~"), "Documents", "ComfyUI", "input",
    )
    input_dir.mkdir(parents=True, exist_ok=True)
    dst = input_dir / src.name
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    return dst.name


def build_workflow(
    *,
    image_basename: str,
    prompt: str,
    seed: int,
    length: int,
    width: int = LTX_WIDTH,
    height: int = LTX_HEIGHT,
    fps: int = LTX_FPS,
    cfg: float = LTX_CFG,
    i2v_strength: float = LTX_I2V_STRENGTH,
    sigmas: str = LTX_DISTILLED_SIGMAS,
    filename_prefix: str = "ltx_motion_smoke",
) -> dict:
    """Build the /prompt-shape workflow dict for one LTX render."""
    return {
        "1": {
            "class_type": "LowVRAMCheckpointLoader",
            "inputs": {"ckpt_name": LTX_CHECKPOINT},
        },
        "2": {
            "class_type": "CLIPLoader",
            "inputs": {
                "clip_name": T5_CLIP,
                "type": "ltxv",
                "device": "default",
            },
        },
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": image_basename, "upload": "image"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": prompt, "clip": ["2", 0]},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "", "clip": ["2", 0]},
        },
        "6": {
            "class_type": "LTXVConditioning",
            "inputs": {
                "positive": ["4", 0],
                "negative": ["5", 0],
                "frame_rate": fps,
            },
        },
        "7": {
            "class_type": "EmptyLTXVLatentVideo",
            "inputs": {
                "width": width,
                "height": height,
                "length": length,
                "batch_size": 1,
            },
        },
        "8": {
            "class_type": "LTXVImgToVideoConditionOnly",
            "inputs": {
                "vae": ["1", 2],
                "image": ["3", 0],
                "latent": ["7", 0],
                "strength": i2v_strength,
            },
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": "euler"},
        },
        "10": {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": sigmas},
        },
        "11": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
        },
        "12": {
            "class_type": "CFGGuider",
            "inputs": {
                "model": ["1", 0],
                "positive": ["6", 0],
                "negative": ["6", 1],
                "cfg": cfg,
            },
        },
        "13": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {
                "noise": ["11", 0],
                "guider": ["12", 0],
                "sampler": ["9", 0],
                "sigmas": ["10", 0],
                "latent_image": ["8", 0],
            },
        },
        "14": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["13", 0], "vae": ["1", 2]},
        },
        # SaveAnimatedWEBP is stock ComfyUI core (MIT). Plays inline in
        # the node UI so motion is visible without leaving ComfyUI.
        "15": {
            "class_type": "SaveAnimatedWEBP",
            "inputs": {
                "images": ["14", 0],
                "filename_prefix": filename_prefix,
                "fps": fps,
                "lossless": False,
                "quality": 85,
                "method": "default",
            },
        },
        # CreateVideo + SaveVideo are stock ComfyUI core. Produces the
        # canonical mp4 for MAD-based motion measurement.
        "16": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["14", 0], "fps": fps},
        },
        "17": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["16", 0],
                "filename_prefix": filename_prefix,
                "format": "mp4",
                "codec": "h264",
            },
        },
    }


def submit(prompt_dict: dict) -> str:
    """POST to ComfyUI /prompt and return prompt_id."""
    body = json.dumps({
        "prompt": prompt_dict,
        "client_id": str(uuid.uuid4()),
    }).encode("utf-8")
    req = urllib.request.Request(
        f"{COMFYUI_URL}/prompt",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body_txt = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ComfyUI rejected the prompt (HTTP {exc.code}):\n{body_txt}"
        ) from exc
    pid = data.get("prompt_id")
    if not pid:
        raise RuntimeError(f"No prompt_id in response: {data!r}")
    return pid


def wait_for_completion(prompt_id: str, *, timeout_s: int = 600) -> dict:
    """Poll /history/<prompt_id> until the prompt completes or times out."""
    deadline = time.time() + timeout_s
    last_msg = ""
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(
                f"{COMFYUI_URL}/history/{prompt_id}",
                timeout=10,
            ) as resp:
                hist = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError):
            time.sleep(2.0)
            continue
        entry = hist.get(prompt_id)
        if entry:
            status = entry.get("status", {})
            if status.get("completed"):
                return entry
            if status.get("status_str") == "error":
                raise RuntimeError(
                    f"Render failed: {json.dumps(status, indent=2)}"
                )
        else:
            msg = "queued"
            if msg != last_msg:
                print(f"  status: {msg}", flush=True)
                last_msg = msg
        time.sleep(2.0)
    raise TimeoutError(f"Render did not complete within {timeout_s}s")


def find_outputs(history_entry: dict) -> list[Path]:
    """Walk the history entry's outputs block and return saved file paths."""
    outputs = history_entry.get("outputs", {})
    out_root = Path(
        os.path.expanduser("~"), "Documents", "ComfyUI", "output",
    )
    found = []
    for node_id, payload in outputs.items():
        for key in ("images", "videos", "gifs"):
            for item in payload.get(key, []) or []:
                rel = item.get("filename")
                subfolder = item.get("subfolder", "")
                folder_type = item.get("type", "output")
                if rel:
                    base = out_root if folder_type == "output" else (
                        Path(os.path.expanduser("~")) / "Documents" / "ComfyUI" / folder_type
                    )
                    p = base / subfolder / rel if subfolder else base / rel
                    if p.exists():
                        found.append(p)
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--role", choices=list(PROMPTS.keys()), default="announcer",
                        help="Pre-baked motion prompt by role (default: announcer)")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Override with a custom prompt (skips --role)")
    parser.add_argument("--still", type=str, default=None,
                        help="Path to radio_bookend.png (default: most recent)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--length", type=int, default=121,
                        help="Frame count (8n+1 valid; LTX native is 121 = 4.84s @ 25fps)")
    parser.add_argument("--filename-prefix", type=str, default="ltx_motion_smoke")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    still_path = args.still or _find_default_still()
    prompt = args.prompt or PROMPTS[args.role]
    image_basename = _ensure_still_in_input_dir(still_path)

    print(f"COMFYUI_URL    = {COMFYUI_URL}", flush=True)
    print(f"still          = {still_path}", flush=True)
    print(f"  -> input/    = {image_basename}", flush=True)
    print(f"prompt ({len(prompt)} chars):", flush=True)
    print(f"  {prompt}", flush=True)
    print(f"seed={args.seed} length={args.length} ({args.length/LTX_FPS:.2f}s @ {LTX_FPS}fps)", flush=True)
    print(f"i2v_strength={LTX_I2V_STRENGTH}  cfg={LTX_CFG}  sigmas=distilled-8step", flush=True)
    print(flush=True)

    workflow = build_workflow(
        image_basename=image_basename,
        prompt=prompt,
        seed=args.seed,
        length=args.length,
        filename_prefix=args.filename_prefix,
    )

    print("submitting workflow...", flush=True)
    t0 = time.time()
    prompt_id = submit(workflow)
    print(f"  prompt_id = {prompt_id}", flush=True)
    print("polling for completion (this will take ~20-60s)...", flush=True)
    history = wait_for_completion(prompt_id, timeout_s=args.timeout)
    elapsed = time.time() - t0
    print(f"  done in {elapsed:.1f}s", flush=True)

    outputs = find_outputs(history)
    if not outputs:
        print("  WARNING: no output files reported in history", flush=True)
        return 2

    print(flush=True)
    print("=== outputs ===", flush=True)
    for p in outputs:
        size_mb = p.stat().st_size / (1024 * 1024)
        print(f"  {p}  ({size_mb:.1f} MB)", flush=True)
    print(flush=True)
    print("To measure motion (MAD between consecutive frames), run:", flush=True)
    print(f"  python scripts\\ltx_motion_mad.py --mp4 \"{outputs[0]}\"", flush=True)
    print("Real motion = sustained 15-30 MAD. Static = <5 MAD.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
