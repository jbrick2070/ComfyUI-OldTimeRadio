"""
SceneAnimator — LTX-Video Image-to-Video wrapper for OldTimeRadio v2.0.

Takes SD3.5-generated anchor images and animates each one into a 2-4 second
video clip using LTX-Video 22B distilled FP8 in I2V (image-to-video) mode.

Pipeline position:
    CharacterForge -> ScenePainter -> VisualCompositor -> MemoryBoundary
    -> [LTX checkpoint load] -> SceneAnimator -> ProductionBus

VRAM contract (V2_actionplan.md Section 5):
    - Calls memory_boundary() on entry to guarantee clean slate
    - LTX-Video 22B FP8 peaks at ~14.1 GB on RTX 5080
    - Must stay under 14.5 GB ceiling at all times
    - fp8_e4m3fn ONLY — e5m2 causes RT core artifacts on Blackwell

Dependencies:
    - ComfyUI-LTXVideo (Kosinkadink) must be installed
    - ltx-2.3-22b-distilled-fp8.safetensors in models/checkpoints/
"""

import gc
import json
import logging
import os
import subprocess
import tempfile
import time

import numpy as np
import torch

log = logging.getLogger("OTR")

_V2_CATEGORY = "OldTimeRadio/v2.0 Visual Drama Engine"

# LTX-Video latent dimensions
_LTX_LATENT_CHANNELS = 128
_LTX_TEMPORAL_COMPRESS = 8
_LTX_SPATIAL_COMPRESS = 32

# Default generation parameters from action plan Section 3
_DEFAULT_WIDTH = 768
_DEFAULT_HEIGHT = 512
_DEFAULT_NUM_FRAMES = 65      # 2.6s at 25fps (must be 8*N+1)
_DEFAULT_FPS = 25
_DEFAULT_STEPS = 20
_DEFAULT_CFG = 3.5
_DEFAULT_STRENGTH = 0.9       # I2V conditioning strength


def _runtime_log(msg):
    """Append a timestamped line to otr_runtime.log."""
    try:
        from datetime import datetime
        ts = datetime.now().strftime("%H:%M:%S")
        log_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "otr_runtime.log"
        )
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"[{ts}] [v2] {msg}\n")
    except Exception:
        pass


def _log_vram_peak(phase, peak_gb, current_gb):
    """Append a JSON line to the VRAM drift-detection log."""
    try:
        from datetime import datetime
        log_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "logs"
        )
        os.makedirs(log_dir, exist_ok=True)
        entry = {
            "ts": datetime.now().isoformat(),
            "phase": phase,
            "peak_gb": round(peak_gb, 3),
            "current_gb": round(current_gb, 3),
        }
        with open(os.path.join(log_dir, "vram_peaks.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception:
        pass


def _vram_snapshot(label):
    """Log current VRAM usage."""
    if torch.cuda.is_available():
        curr = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        log.info("[SceneAnimator] VRAM %s: current=%.2fGB peak=%.2fGB", label, curr, peak)
        _log_vram_peak(f"scene_animator_{label}", peak, curr)


def _valid_frame_count(target_frames):
    """Round to nearest valid LTX frame count (8*N + 1)."""
    n = max(1, round((target_frames - 1) / 8))
    return 8 * n + 1


def _encode_prompt(clip, text):
    """Encode text prompt using the CLIP/text encoder model.

    Returns ComfyUI conditioning format: [[cond_tensor, {extras}]]
    """
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    output = dict(output)
    cond = output.pop("cond")
    return [[cond, output]]


def _frames_to_mp4(frames_tensor, output_path, fps=25):
    """Save a batch of frames tensor to MP4 using FFmpeg.

    Args:
        frames_tensor: [num_frames, H, W, 3] float32 tensor in [0, 1]
        output_path: Destination .mp4 path
        fps: Frame rate
    """
    num_frames = frames_tensor.shape[0]
    h, w = frames_tensor.shape[1], frames_tensor.shape[2]

    # Write raw frames via pipe to FFmpeg (no temp PNGs needed)
    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-vcodec", "rawvideo",
        "-s", f"{w}x{h}",
        "-pix_fmt", "rgb24",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "20",
        "-pix_fmt", "yuv420p",
        output_path,
    ]

    proc = subprocess.Popen(
        ffmpeg_cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )

    # Write frame data
    for i in range(num_frames):
        frame_np = (frames_tensor[i].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
        proc.stdin.write(frame_np.tobytes())

    proc.stdin.close()
    proc.wait(timeout=120)

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"FFmpeg failed (rc={proc.returncode}): {stderr[:500]}")

    log.info("[SceneAnimator] Saved clip: %s (%d frames, %dx%d, %dfps)",
             os.path.basename(output_path), num_frames, w, h, fps)


# ===================================================================
# COMFYUI NODE: SceneAnimator
# ===================================================================

class SceneAnimator:
    """v2.0 Scene Animator — LTX-Video I2V animation per scene.

    Receives anchor images (from ScenePainter/VisualCompositor) and
    animates each into a short video clip using LTX-Video in I2V mode.

    The node expects the LTX model to be provided via workflow wiring
    (from a LowVRAMCheckpointLoader or CheckpointLoaderSimple node).
    A MemoryBoundary node MUST be placed between the SD3.5 visual
    generation nodes and the LTX checkpoint loader.

    Each scene produces one MP4 clip stored in the output directory.
    The clip paths are returned as a JSON array for ProductionBus
    to concat with the episode audio.
    """

    CATEGORY = _V2_CATEGORY
    FUNCTION = "animate"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_clips_json", "animator_log")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "v2.0 Scene Animator — Animates SD3.5 anchor images into "
        "short LTX-Video I2V clips. One clip per scene. "
        "Requires MemoryBoundary between SD3.5 and LTX phases."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ltx_model": ("MODEL", {
                    "tooltip": "LTX-Video model from LowVRAMCheckpointLoader",
                }),
                "ltx_vae": ("VAE", {
                    "tooltip": "LTX-Video VAE from checkpoint loader",
                }),
                "scene_prompts_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": (
                        "JSON array of scene prompt dicts from PromptBuilder. "
                        "Each dict has: scene_id, anchor_prompt, motion_prompt, "
                        "motion, duration_s."
                    ),
                }),
                "seed": ("INT", {
                    "default": 1337,
                    "min": 0,
                    "max": 0xffffffffffffffff,
                }),
                "steps": ("INT", {
                    "default": _DEFAULT_STEPS,
                    "min": 1, "max": 100,
                    "tooltip": "Sampling steps. 20 is good for distilled LTX.",
                }),
                "cfg": ("FLOAT", {
                    "default": _DEFAULT_CFG,
                    "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "CFG scale. 3.5 works well for LTX-Video distilled.",
                }),
                "width": ("INT", {
                    "default": _DEFAULT_WIDTH,
                    "min": 256, "max": 1280, "step": 32,
                    "tooltip": "Output video width. 768 for VRAM safety.",
                }),
                "height": ("INT", {
                    "default": _DEFAULT_HEIGHT,
                    "min": 256, "max": 1024, "step": 32,
                    "tooltip": "Output video height. 512 for VRAM safety.",
                }),
                "num_frames": ("INT", {
                    "default": _DEFAULT_NUM_FRAMES,
                    "min": 9, "max": 257, "step": 8,
                    "tooltip": (
                        "Frames per clip. Must be 8*N+1. "
                        "65=2.6s, 97=3.9s at 25fps."
                    ),
                }),
                "fps": ("INT", {
                    "default": _DEFAULT_FPS,
                    "min": 1, "max": 60,
                }),
                "i2v_strength": ("FLOAT", {
                    "default": _DEFAULT_STRENGTH,
                    "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": (
                        "I2V conditioning strength. 0.9 = mostly preserve anchor, "
                        "0.5 = more creative motion. Higher = more faithful to anchor."
                    ),
                }),
            },
            "optional": {
                "ltx_clip": ("CLIP", {
                    "tooltip": (
                        "LTX text encoder. If not provided, motion_prompt "
                        "conditioning will use empty positive."
                    ),
                }),
                "anchor_images": ("IMAGE", {
                    "tooltip": (
                        "Anchor images from VisualCompositor or ScenePainter. "
                        "One per scene. If fewer than scenes, last image is reused."
                    ),
                }),
                "output_subdir": ("STRING", {
                    "default": "old_time_radio",
                    "tooltip": "Subdirectory under ComfyUI output root for clips.",
                }),
                "fallback_clip_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Path to a static fallback clip. Used on OOM or generation "
                        "failure instead of crashing the pipeline."
                    ),
                }),
            },
        }

    def animate(self, ltx_model, ltx_vae, scene_prompts_json="[]",
                seed=1337, steps=_DEFAULT_STEPS, cfg=_DEFAULT_CFG,
                width=_DEFAULT_WIDTH, height=_DEFAULT_HEIGHT,
                num_frames=_DEFAULT_NUM_FRAMES, fps=_DEFAULT_FPS,
                i2v_strength=_DEFAULT_STRENGTH,
                ltx_clip=None, anchor_images=None,
                output_subdir="old_time_radio", fallback_clip_path=""):

        import comfy.sample
        import comfy.model_management

        _runtime_log("SceneAnimator: starting")
        _vram_snapshot("entry")

        # Resolve output directory
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except ImportError:
            output_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "..", "..", "output"
            ))

        _subdir = str(output_subdir).strip() if output_subdir else ""
        if _subdir and _subdir.lower() not in ("false", "none", "0", ""):
            output_dir = os.path.join(output_dir, _subdir)
        os.makedirs(output_dir, exist_ok=True)

        clips_dir = os.path.join(output_dir, "v2_clips")
        os.makedirs(clips_dir, exist_ok=True)

        # Parse scene prompts
        try:
            scene_prompts = json.loads(scene_prompts_json) if isinstance(
                scene_prompts_json, str) else scene_prompts_json
        except (json.JSONDecodeError, TypeError):
            scene_prompts = []

        if not scene_prompts:
            _runtime_log("SceneAnimator: no scene prompts — returning empty")
            return ("[]", "SceneAnimator: No scene prompts provided.")

        # Validate frame count
        num_frames = _valid_frame_count(num_frames)
        temporal_len = ((num_frames - 1) // _LTX_TEMPORAL_COMPRESS) + 1
        lat_h = height // _LTX_SPATIAL_COMPRESS
        lat_w = width // _LTX_SPATIAL_COMPRESS

        log.info(
            "[SceneAnimator] %d scenes, %d frames/clip (%dx%d), "
            "latent=[128, %d, %d, %d], steps=%d, cfg=%.1f",
            len(scene_prompts), num_frames, width, height,
            temporal_len, lat_h, lat_w, steps, cfg
        )
        _runtime_log(
            f"SceneAnimator: {len(scene_prompts)} scenes, "
            f"{num_frames} frames @ {width}x{height}, "
            f"latent=[128,{temporal_len},{lat_h},{lat_w}]"
        )

        # Determine anchor image count
        num_anchors = 0
        if anchor_images is not None:
            num_anchors = anchor_images.shape[0]

        clip_paths = []
        log_lines = [
            f"SceneAnimator: {len(scene_prompts)} scenes, "
            f"{num_frames} frames/clip ({width}x{height})"
        ]

        for s_idx, sp in enumerate(scene_prompts):
            scene_id = sp.get("scene_id", f"s{s_idx + 1:02d}")
            motion_prompt = sp.get("motion_prompt", "cinematic scene")
            neg_prompt = sp.get("negative_motion", "blurry, low quality")
            clip_filename = f"scene_{scene_id}_{seed + s_idx:08d}.mp4"
            clip_path = os.path.join(clips_dir, clip_filename)

            _runtime_log(
                f"SceneAnimator: scene {s_idx + 1}/{len(scene_prompts)} "
                f"[{scene_id}] starting"
            )

            try:
                # --- Encode conditioning ---
                if ltx_clip is not None:
                    positive = _encode_prompt(ltx_clip, motion_prompt)
                    negative = _encode_prompt(ltx_clip, neg_prompt)
                else:
                    # No text encoder — use unconditional generation
                    positive = [[torch.zeros([1, 1, 4096]), {}]]
                    negative = [[torch.zeros([1, 1, 4096]), {}]]

                # --- Create video latent ---
                latent_image = torch.zeros(
                    [1, _LTX_LATENT_CHANNELS, temporal_len, lat_h, lat_w],
                    device="cpu"
                )

                # Apply I2V conditioning: encode anchor and place at frame 0
                if anchor_images is not None and num_anchors > 0:
                    anchor_idx = min(s_idx, num_anchors - 1)
                    anchor_pixel = anchor_images[anchor_idx:anchor_idx + 1, :, :, :3]

                    # Resize anchor to match target if needed
                    if (anchor_pixel.shape[1] != height or
                            anchor_pixel.shape[2] != width):
                        anchor_pixel = torch.nn.functional.interpolate(
                            anchor_pixel.permute(0, 3, 1, 2),
                            size=(height, width),
                            mode="bilinear",
                            align_corners=False,
                        ).permute(0, 2, 3, 1)

                    # Encode anchor image with LTX VAE
                    anchor_latent = ltx_vae.encode(anchor_pixel)

                    # Place encoded anchor at temporal position 0
                    t_len = min(anchor_latent.shape[2], temporal_len)
                    latent_image[:, :, :t_len] = anchor_latent[:, :, :t_len]

                    log.info(
                        "[SceneAnimator] I2V anchor applied: "
                        "anchor_latent=%s, strength=%.2f",
                        list(anchor_latent.shape), i2v_strength
                    )

                # Fix latent channels if model expects different count
                latent_image = comfy.sample.fix_empty_latent_channels(
                    ltx_model, latent_image
                )

                # --- Create noise mask for I2V strength ---
                # Where mask=0: keep original (anchor), mask=1: full denoise
                denoise_mask = None
                if anchor_images is not None and num_anchors > 0:
                    denoise_mask = torch.ones_like(latent_image[:, :1])
                    # First frame(s): reduce noise to preserve anchor
                    denoise_mask[:, :, :t_len] = i2v_strength

                # --- Prepare noise ---
                scene_seed = seed + s_idx
                noise = comfy.sample.prepare_noise(latent_image, scene_seed)

                # --- Sample ---
                _vram_snapshot(f"scene_{scene_id}_pre_sample")
                t_start = time.time()

                samples = comfy.sample.sample(
                    ltx_model, noise, steps, cfg,
                    sampler_name="euler",
                    scheduler="normal",
                    positive=positive,
                    negative=negative,
                    latent_image=latent_image,
                    denoise=1.0,
                    seed=scene_seed,
                )

                sample_time = time.time() - t_start
                _vram_snapshot(f"scene_{scene_id}_post_sample")
                _runtime_log(
                    f"SceneAnimator: scene [{scene_id}] sampled in "
                    f"{sample_time:.1f}s"
                )

                # --- Decode video frames ---
                t_decode = time.time()
                video_frames = ltx_vae.decode(samples)
                decode_time = time.time() - t_decode

                # video_frames shape: [B*F, H, W, C] or [F, H, W, C]
                # Ensure we have [F, H, W, C]
                if video_frames.dim() == 4:
                    frames = video_frames
                elif video_frames.dim() == 5:
                    frames = video_frames.squeeze(0)
                else:
                    frames = video_frames.reshape(-1, height, width, 3)

                log.info(
                    "[SceneAnimator] Decoded %d frames in %.1fs, shape=%s",
                    frames.shape[0], decode_time, list(frames.shape)
                )

                # --- Save clip as MP4 ---
                _frames_to_mp4(frames, clip_path, fps=fps)
                clip_paths.append(clip_path)

                log_lines.append(
                    f"  + {scene_id}: {frames.shape[0]} frames, "
                    f"{sample_time:.1f}s sample, {decode_time:.1f}s decode"
                )

                # Flush between scenes to prevent VRAM accumulation
                if s_idx < len(scene_prompts) - 1:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Extended boundary every 10 scenes (driver fragmentation)
                if (s_idx + 1) % 10 == 0 and s_idx < len(scene_prompts) - 1:
                    from .memory_boundary import memory_boundary
                    memory_boundary(sleep_s=3.0, phase_label=f"defrag_after_{scene_id}")
                    _runtime_log(
                        f"SceneAnimator: extended boundary after scene {s_idx + 1}"
                    )

            except torch.cuda.OutOfMemoryError as oom:
                log.error(
                    "[SceneAnimator] OOM on scene %s: %s", scene_id, oom
                )
                _runtime_log(f"SceneAnimator: OOM on [{scene_id}] - using fallback")
                _handle_fallback(clip_path, fallback_clip_path, clip_paths, log_lines,
                                 scene_id, str(oom))

            except Exception as exc:
                log.error(
                    "[SceneAnimator] Error on scene %s: %s", scene_id, exc,
                    exc_info=True
                )
                _runtime_log(f"SceneAnimator: ERROR on [{scene_id}]: {exc}")
                _handle_fallback(clip_path, fallback_clip_path, clip_paths, log_lines,
                                 scene_id, str(exc))

        # --- Final summary ---
        total_clips = len(clip_paths)
        log_lines.insert(1, f"  Completed: {total_clips}/{len(scene_prompts)} clips")
        animator_log = "\n".join(log_lines)

        _runtime_log(
            f"SceneAnimator: DONE - {total_clips}/{len(scene_prompts)} clips"
        )
        _vram_snapshot("exit")

        return (json.dumps(clip_paths), animator_log)


def _handle_fallback(clip_path, fallback_clip_path, clip_paths, log_lines,
                     scene_id, error_msg):
    """Insert fallback clip on failure. Never instantiate a model in fallback."""
    import shutil

    if fallback_clip_path and os.path.isfile(fallback_clip_path):
        shutil.copy2(fallback_clip_path, clip_path)
        clip_paths.append(clip_path)
        log_lines.append(f"  ! {scene_id}: FALLBACK (from {os.path.basename(fallback_clip_path)})")
    else:
        # Generate a 2-second black clip as last resort
        try:
            black_frames = torch.zeros([50, 512, 768, 3], dtype=torch.float32)
            _frames_to_mp4(black_frames, clip_path, fps=25)
            clip_paths.append(clip_path)
            log_lines.append(f"  ! {scene_id}: BLACK FALLBACK ({error_msg[:60]})")
        except Exception as fb_err:
            log_lines.append(f"  X {scene_id}: FAILED entirely ({fb_err})")
