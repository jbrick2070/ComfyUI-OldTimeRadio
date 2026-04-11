"""
ComfyUI-OldTimeRadio - v2.0 Visual Drama Engine
=================================================

Production inference implementation for the Visual Drama Engine.
Transforms the Director's visual_plan JSON into rendered imagery.

Pipeline:
  8.  CharacterForge    - Flux/SD portrait generation per cast member
  9.  ScenePainter      - LTX 2.3 / Flux scene background generation
  10. VisualCompositor  - PIL compositing of characters over scenes + CRT
  11. ProductionBus     - FFmpeg timeline assembly synced to audio

Architecture follows the v2 Research Report:
  - Audio-first: all visual work happens AFTER audio pipeline completes.
  - Sequential VRAM handoff: Forge unloads before Painter loads.
  - JSON state machine: visual_plan drives all rendering deterministically.

See docs/v2_research_report.md for the full architectural analysis.
"""

import gc
import json
import logging
import math
import os
import re
import shutil
import struct
import tempfile

import numpy as np
import torch

log = logging.getLogger("OTR")

_V2_CATEGORY = "OldTimeRadio/v2.0 Visual Drama Engine"

# Negative prompts tuned for each use case
_NEG_PORTRAIT = (
    "blurry, low quality, deformed, ugly, bad anatomy, extra fingers, "
    "extra limbs, disfigured, watermark, text, signature, cropped, "
    "out of frame, worst quality, jpeg artifacts"
)
_NEG_SCENE = (
    "blurry, low quality, watermark, text, signature, ugly, deformed, "
    "worst quality, jpeg artifacts, cartoon, anime, illustration"
)


# ---------------------------------------------------------------------------
# STARTUP JANITOR — clean orphaned preview_tmp dirs from interrupted renders
# (try/finally cannot clean up after kill -9)
# ---------------------------------------------------------------------------

def _cleanup_orphaned_preview_tmp():
    """Remove stale otr_v2_* temp dirs left by crashed/killed renders."""
    try:
        import folder_paths
        output_dir = folder_paths.get_output_directory()
    except ImportError:
        return  # Outside ComfyUI, nothing to clean
    preview_root = os.path.join(output_dir, "preview_tmp")
    if not os.path.isdir(preview_root):
        return
    cleaned = 0
    for entry in os.listdir(preview_root):
        entry_path = os.path.join(preview_root, entry)
        if os.path.isdir(entry_path) and entry.startswith("otr_v2_"):
            try:
                shutil.rmtree(entry_path, ignore_errors=True)
                cleaned += 1
            except Exception:
                pass
    if cleaned:
        log.info("[v2.0] Startup janitor: removed %d orphaned preview_tmp dirs", cleaned)

_cleanup_orphaned_preview_tmp()


# ---------------------------------------------------------------------------
# FILENAME SANITIZATION
# ---------------------------------------------------------------------------

def _safe_name(name, max_len=80):
    """Strip filesystem-unsafe characters from an output filename."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return (name or "untitled")[:max_len]


# ---------------------------------------------------------------------------
# VRAM MANAGEMENT - follows OTR v1.5 sequential handoff pattern
# ---------------------------------------------------------------------------

def _flush_vram(phase=""):
    """Aggressive VRAM flush between heavy visual operations."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if phase:
        log.info("[v2.0] VRAM flushed after %s", phase)


def _vram_snapshot(label=""):
    """Log current VRAM usage for debugging."""
    if torch.cuda.is_available():
        curr = torch.cuda.memory_allocated() / (1024 ** 3)
        peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        log.info("[v2.0] VRAM_SNAPSHOT %s: current=%.2fGB peak=%.2fGB",
                 label, curr, peak)


# ---------------------------------------------------------------------------
# CORE SAMPLING - wraps ComfyUI internals for image generation
# ---------------------------------------------------------------------------

def _encode_prompt(clip, text):
    """Encode a text prompt using the CLIP model.

    Returns ComfyUI conditioning format: [[cond_tensor, {extras}]]
    Compatible with SD 1.5, SDXL, SD 3.5, and Flux CLIP encoders.
    """
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    output = dict(output)  # shallow copy — do not mutate the encoder's cached dict
    cond = output.pop("cond")
    # Remaining keys (pooled_output, etc.) become the extras dict
    return [[cond, output]]


def _generate_image(model, clip, vae, prompt, negative_prompt,
                    width, height, seed, steps=20, cfg=7.0,
                    sampler_name="euler", scheduler="normal"):
    """Generate a single image using the full ComfyUI sampling pipeline.

    Uses comfy.sample internals with keyword args for version safety.
    Handles SD 1.5, SDXL, SD 3.5, and Flux latent channel differences
    automatically via fix_empty_latent_channels.
    """
    from nodes.safety_filter import classify_prompt

    import comfy.sample
    import comfy.model_management
    import comfy.utils

    # Content safety pre-check — block before any GPU work
    safety = classify_prompt(prompt)
    if safety["result"] == "block":
        log.warning("[v2.0] Prompt BLOCKED by safety filter: %s", safety["reason"])
        # Return a dim gray placeholder frame — no GPU resources consumed
        placeholder = torch.full([1, height, width, 3], 0.15, dtype=torch.float32)
        return placeholder

    # Encode positive and negative prompts
    positive = _encode_prompt(clip, prompt)
    negative = _encode_prompt(clip, negative_prompt)

    # Create empty latent (4 channels is the default; fix_empty_latent_channels
    # will adjust to 16 channels for Flux/SD3 automatically)
    latent_h = height // 8
    latent_w = width // 8
    latent_image = torch.zeros([1, 4, latent_h, latent_w], device="cpu")
    latent_image = comfy.sample.fix_empty_latent_channels(model, latent_image)

    # Prepare noise from seed
    noise = comfy.sample.prepare_noise(latent_image, seed)

    # Sample - this is what KSampler does internally
    log.info("[v2.0] Sampling: %dx%d, %d steps, cfg=%.1f, seed=%d",
             width, height, steps, cfg, seed)

    samples = comfy.sample.sample(
        model, noise, steps, cfg,
        sampler_name=sampler_name,
        scheduler=scheduler,
        positive=positive,
        negative=negative,
        latent_image=latent_image,
        denoise=1.0,
        seed=seed
    )

    # Decode latent to pixel space
    # VAE.decode returns [B, H, W, C] float32 tensor in [0, 1]
    images = vae.decode(samples)

    return images


def _generate_batch(model, clip, vae, prompts, negative_prompt,
                    width, height, base_seed, steps=20, cfg=7.0):
    """Generate multiple images sequentially with unique seeds.

    Each image gets base_seed + index for reproducibility.
    Flushes VRAM between images to stay within 14.5GB ceiling.
    """
    results = []
    for i, prompt in enumerate(prompts):
        seed = base_seed + i
        log.info("[v2.0] Generating image %d/%d (seed=%d): %s",
                 i + 1, len(prompts), seed, prompt[:60])
        try:
            img = _generate_image(
                model, clip, vae, prompt, negative_prompt,
                width, height, seed, steps, cfg
            )
            results.append(img)
        except Exception as e:
            log.error("[v2.0] Image generation failed for prompt %d: %s", i, e)
            # Return a red error frame instead of crashing
            err_img = torch.zeros([1, height, width, 3], dtype=torch.float32)
            err_img[:, :, :, 0] = 0.3  # Dim red tint to signal error
            results.append(err_img)

        # Flush between images to prevent VRAM accumulation
        if i < len(prompts) - 1:
            _flush_vram()

    if results:
        return torch.cat(results, dim=0)
    else:
        return torch.zeros([1, height, width, 3], dtype=torch.float32)


# ===================================================================
# NODE 8: CHARACTER FORGE
# ===================================================================

class CharacterForge:
    """v2.0 Character Forge - Flux/SD portrait generation per cast member.

    Reads the visual_plan.characters section from the Director's
    production_plan_json and generates one portrait per character
    using the provided Flux/SD model.

    Portraits use consistent seeds derived from the episode seed,
    ensuring reproducible character appearance across re-runs.

    Output is a batch IMAGE tensor with one frame per character,
    suitable for downstream compositing or TripoSR 3D conversion.

    VRAM: Sequential handoff. Runs AFTER audio pipeline completes.
    Flushes between character generations to stay under 14.5GB.
    """

    CATEGORY = _V2_CATEGORY
    FUNCTION = "forge"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("character_portraits", "forge_log")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "v2.0 Character Forge - Generates consistent character portraits "
        "from the Director's visual_plan using Flux or SD. "
        "One portrait per cast member, batched output."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Production plan JSON with visual_plan.characters section",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                }),
                "portrait_size": (["512x512", "768x768", "1024x1024"], {
                    "default": "512x512",
                    "tooltip": "Portrait resolution. 512 for speed, 1024 for quality.",
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 100,
                    "tooltip": "Sampling steps. 20 is good for Flux, 30+ for SD 3.5.",
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5,
                    "tooltip": "CFG scale. 3.5 for Flux, 7.0 for SD/SDXL.",
                }),
            },
            "optional": {
                "reference_image": ("IMAGE", {
                    "tooltip": "Reference image for IP-Adapter style consistency",
                }),
                "vram_power_wash": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Unload all models and flush VRAM before visual phase. "
                               "Recommended True on 16GB cards to reclaim audio-phase memory.",
                }),
                "debug_vram_snapshots": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Emit VRAM_SNAPSHOT markers at entry/exit for QA.",
                }),
            },
        }

    def forge(self, model, clip, vae, production_plan_json="{}",
             seed=42, portrait_size="512x512", steps=20, cfg=3.5,
             reference_image=None, vram_power_wash=True,
             debug_vram_snapshots=False):

        # VRAM power-wash: reclaim memory from audio phase before visual work
        if vram_power_wash:
            import comfy.model_management as mm
            log.info("[CharacterForge] VRAM power-wash: unloading all models + flushing cache")
            mm.unload_all_models()
            mm.soft_empty_cache()
            _flush_vram("power-wash")

        if debug_vram_snapshots:
            _vram_snapshot("CharacterForge-entry")

        # Parse dimensions
        w, h = [int(x) for x in portrait_size.split("x")]

        # Parse the visual plan
        try:
            plan = json.loads(production_plan_json)
        except json.JSONDecodeError as e:
            log.error("[CharacterForge] Invalid JSON: %s", e)
            dummy = torch.zeros([1, h, w, 3], dtype=torch.float32)
            return (dummy, f"ERROR: Invalid production plan JSON - {e}")

        visual_plan = plan.get("visual_plan", {})
        characters = visual_plan.get("characters", {})

        if not characters:
            log.info("[CharacterForge] No characters in visual_plan - returning placeholder")
            dummy = torch.zeros([1, h, w, 3], dtype=torch.float32)
            return (dummy, "No characters found in visual_plan. "
                          "Ensure the Director is generating a visual_plan section.")

        # Build prompt list from character data
        prompts = []
        char_names = []
        for name, data in characters.items():
            prompt = data.get("portrait_prompt", f"Portrait photograph of {name}")
            # Enhance prompt for portrait quality
            enhanced = (
                f"Professional studio portrait photograph, {prompt}, "
                f"clean background, dramatic lighting, sharp focus, "
                f"high detail, cinematic color grading"
            )
            prompts.append(enhanced)
            char_names.append(name)

        log.info("[CharacterForge] Generating %d character portraits at %s",
                 len(prompts), portrait_size)

        # Generate all portraits
        portraits = _generate_batch(
            model, clip, vae, prompts, _NEG_PORTRAIT,
            w, h, seed, steps, cfg
        )

        # Build status log
        log_lines = [f"Character Forge: {len(char_names)} portraits generated ({portrait_size})"]
        for i, name in enumerate(char_names):
            log_lines.append(f"  + {name} (seed={seed + i})")
        forge_log = "\n".join(log_lines)

        _flush_vram("CharacterForge-complete")
        if debug_vram_snapshots:
            _vram_snapshot("CharacterForge-exit")

        log.info("[CharacterForge] Complete: %d portraits", len(char_names))
        return (portraits, forge_log)


# ===================================================================
# NODE 9: SCENE PAINTER
# ===================================================================

class ScenePainter:
    """v2.0 Scene Painter - Cinematic background generation per scene.

    Reads the visual_plan.scenes section from the Director's
    production_plan_json and generates one establishing shot
    per scene using the provided model (Flux/SD for stills,
    LTX 2.3 for video when available).

    For the alpha, generates high-quality static backgrounds.
    These serve as keyframes for future AnimateDiff/LTX video
    integration via IP-Adapter anchoring.

    VRAM: Sequential handoff after CharacterForge completes.
    """

    CATEGORY = _V2_CATEGORY
    FUNCTION = "paint"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("scene_backgrounds", "painter_log")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "v2.0 Scene Painter - Generates cinematic establishing shots "
        "from the Director's visual_plan using Flux, SD, or LTX 2.3. "
        "One background per scene, batched output."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "vae": ("VAE",),
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Production plan JSON with visual_plan.scenes section",
                }),
                "width": ("INT", {
                    "default": 1280, "min": 256, "max": 2048,
                    "tooltip": "Scene width. 1280 for 720p, 1920 for 1080p.",
                }),
                "height": ("INT", {
                    "default": 720, "min": 256, "max": 2048,
                    "tooltip": "Scene height. 720 for 720p, 1080 for 1080p.",
                }),
                "seed": ("INT", {
                    "default": 42, "min": 0, "max": 0xffffffffffffffff,
                }),
                "steps": ("INT", {
                    "default": 20, "min": 1, "max": 100,
                }),
                "cfg": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 20.0, "step": 0.5,
                }),
            },
            "optional": {
                "anchor_image": ("IMAGE", {
                    "tooltip": "Anchor frame for IP-Adapter visual consistency",
                }),
                "debug_vram_snapshots": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Emit VRAM_SNAPSHOT markers at entry/exit for QA.",
                }),
            },
        }

    def paint(self, model, clip, vae, production_plan_json="{}",
             width=1280, height=720, seed=42, steps=20, cfg=3.5,
             anchor_image=None, debug_vram_snapshots=False):

        if debug_vram_snapshots:
            _vram_snapshot("ScenePainter-entry")

        # Parse the visual plan
        try:
            plan = json.loads(production_plan_json)
        except json.JSONDecodeError as e:
            log.error("[ScenePainter] Invalid JSON: %s", e)
            dummy = torch.zeros([1, height, width, 3], dtype=torch.float32)
            return (dummy, f"ERROR: Invalid production plan JSON - {e}")

        visual_plan = plan.get("visual_plan", {})
        scenes = visual_plan.get("scenes", [])

        if not scenes:
            log.info("[ScenePainter] No scenes in visual_plan - returning placeholder")
            dummy = torch.zeros([1, height, width, 3], dtype=torch.float32)
            return (dummy, "No scenes found in visual_plan. "
                          "Ensure the Director is generating a visual_plan section.")

        # Build prompt list from scene data
        prompts = []
        scene_ids = []
        for scene_data in scenes:
            scene_id = scene_data.get("scene_id", "unknown")
            prompt = scene_data.get("visual_prompt",
                                    scene_data.get("shot_description", "A cinematic scene"))
            # Enhance for cinematic quality
            enhanced = (
                f"Cinematic establishing shot, {prompt}, "
                f"professional cinematography, dramatic lighting, "
                f"film grain, anamorphic lens, photorealistic, 8K detail"
            )
            prompts.append(enhanced)
            scene_ids.append(scene_id)

        log.info("[ScenePainter] Generating %d scene backgrounds at %dx%d",
                 len(prompts), width, height)

        # Offset seed from portraits to avoid visual correlation
        scene_seed = seed + 10000

        # Generate all backgrounds
        backgrounds = _generate_batch(
            model, clip, vae, prompts, _NEG_SCENE,
            width, height, scene_seed, steps, cfg
        )

        # Build status log
        log_lines = [f"Scene Painter: {len(scene_ids)} backgrounds generated ({width}x{height})"]
        for i, sid in enumerate(scene_ids):
            log_lines.append(f"  + {sid}: {prompts[i][:60]}... (seed={scene_seed + i})")
        painter_log = "\n".join(log_lines)

        _flush_vram("ScenePainter-complete")
        if debug_vram_snapshots:
            _vram_snapshot("ScenePainter-exit")

        log.info("[ScenePainter] Complete: %d scene backgrounds", len(scene_ids))
        return (backgrounds, painter_log)


# ===================================================================
# NODE 10: VISUAL COMPOSITOR
# ===================================================================

class VisualCompositor:
    """v2.0 Visual Compositor - Layers characters over scene backgrounds.

    Composites CharacterForge portraits over ScenePainter backgrounds
    using PIL. Applies optional CRT overlay for retro aesthetic
    (reusing the phosphor-green scanline look from SignalLostVideo).

    This is a CPU-only node - no GPU needed. It runs after both
    visual generators have completed and released VRAM.

    Input: character_renders (batch IMAGE), scene_backgrounds (batch IMAGE)
    Output: composited_frames (batch IMAGE), one per scene
    """

    CATEGORY = _V2_CATEGORY
    FUNCTION = "composite"
    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("composited_frames", "compositor_log")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "v2.0 Visual Compositor - Layers character portraits over "
        "scene backgrounds with optional CRT post-processing. "
        "CPU-only, no GPU required."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Production plan for scene-character mapping",
                }),
                "crt_overlay": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply retro CRT scanline aesthetic",
                }),
                "character_scale": ("FLOAT", {
                    "default": 0.3, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Scale of character portrait relative to scene",
                }),
            },
            "optional": {
                "character_renders": ("IMAGE", {
                    "tooltip": "Character portraits from CharacterForge",
                }),
                "scene_backgrounds": ("IMAGE", {
                    "tooltip": "Scene backgrounds from ScenePainter",
                }),
                "episode_audio": ("AUDIO", {
                    "tooltip": "Mastered episode audio for timeline sync",
                }),
            },
        }

    def composite(self, production_plan_json="{}", crt_overlay=True,
                  character_scale=0.3, character_renders=None,
                  scene_backgrounds=None, episode_audio=None):

        from PIL import Image, ImageDraw, ImageFilter

        log.info("[VisualCompositor] Starting compositing pass")

        # Parse plan to understand scene-character mapping
        try:
            plan = json.loads(production_plan_json)
        except json.JSONDecodeError:
            plan = {}

        visual_plan = plan.get("visual_plan", {})
        scenes = visual_plan.get("scenes", [])
        characters = visual_plan.get("characters", {})

        # If no scene backgrounds provided, return a status-only placeholder
        if scene_backgrounds is None:
            log.info("[VisualCompositor] No scene backgrounds - returning placeholder")
            dummy = torch.zeros([1, 720, 1280, 3], dtype=torch.float32)
            return (dummy, "Waiting for ScenePainter input...")

        num_scenes = scene_backgrounds.shape[0]
        scene_h, scene_w = scene_backgrounds.shape[1], scene_backgrounds.shape[2]

        composited = []

        for s_idx in range(num_scenes):
            # Convert scene background to PIL
            bg_np = (scene_backgrounds[s_idx].cpu().numpy() * 255).astype(np.uint8)
            bg_pil = Image.fromarray(bg_np, "RGB")

            # Overlay character portraits in the lower-third
            if character_renders is not None:
                num_chars = character_renders.shape[0]
                # Calculate portrait size
                p_size = int(min(scene_h, scene_w) * character_scale)

                # Position characters evenly across the bottom
                total_width = num_chars * p_size + (num_chars - 1) * 10
                start_x = max(10, (scene_w - total_width) // 2)
                y_pos = scene_h - p_size - 20  # 20px from bottom

                for c_idx in range(num_chars):
                    # Get character portrait
                    char_np = (character_renders[c_idx].cpu().numpy() * 255).astype(np.uint8)
                    char_pil = Image.fromarray(char_np, "RGB")
                    char_pil = char_pil.resize((p_size, p_size), Image.LANCZOS)

                    # Add a subtle border/shadow
                    shadow = Image.new("RGBA", (p_size + 4, p_size + 4), (0, 0, 0, 128))
                    bg_pil.paste(Image.new("RGB", (p_size + 4, p_size + 4), (0, 0, 0)),
                                 (start_x - 2, y_pos - 2))
                    bg_pil.paste(char_pil, (start_x, y_pos))

                    start_x += p_size + 10

            # Apply CRT overlay if enabled
            if crt_overlay:
                bg_pil = self._apply_crt(bg_pil)

            # Convert back to tensor
            result_np = np.array(bg_pil).astype(np.float32) / 255.0
            composited.append(torch.from_numpy(result_np).unsqueeze(0))

        result = torch.cat(composited, dim=0)

        log_lines = [
            f"Visual Compositor: {num_scenes} frames composited ({scene_w}x{scene_h})",
            f"  Characters overlaid: {character_renders.shape[0] if character_renders is not None else 0}",
            f"  CRT overlay: {'ON' if crt_overlay else 'OFF'}",
            f"  Character scale: {character_scale:.0%}",
        ]
        compositor_log = "\n".join(log_lines)

        log.info("[VisualCompositor] Complete: %d composited frames", num_scenes)
        return (result, compositor_log)

    @staticmethod
    def _apply_crt(img):
        """Apply a subtle CRT scanline + vignette effect.

        Replicates the phosphor-green aesthetic from SignalLostVideo
        but as a lighter overlay suitable for cinematic backgrounds.
        """
        from PIL import Image, ImageDraw, ImageEnhance

        w, h = img.size

        # Slight green tint (CRT phosphor)
        r, g, b = img.split()
        g = ImageEnhance.Brightness(g).enhance(1.05)
        img = Image.merge("RGB", (r, g, b))

        # Scanlines - darken every third row subtly
        scanline_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        scan_draw = ImageDraw.Draw(scanline_overlay)
        for y in range(0, h, 3):
            scan_draw.line([(0, y), (w, y)], fill=(0, 0, 0, 25), width=1)
        img = Image.alpha_composite(img.convert("RGBA"), scanline_overlay).convert("RGB")

        # Vignette - darken corners
        vignette = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        vig_draw = ImageDraw.Draw(vignette)
        cx, cy = w // 2, h // 2
        max_dist = math.sqrt(cx * cx + cy * cy)
        for ring in range(0, int(max_dist), 4):
            alpha = int(min(60, (ring / max_dist) * 80))
            if alpha > 0:
                vig_draw.ellipse(
                    [cx - ring, cy - ring, cx + ring, cy + ring],
                    outline=(0, 0, 0, alpha)
                )
        img = Image.alpha_composite(img.convert("RGBA"), vignette).convert("RGB")

        return img


# ===================================================================
# NODE 11: PRODUCTION BUS
# ===================================================================

class ProductionBus:
    """v2.0 Production Bus - Unified audio-visual timeline assembly.

    Merges composited visual frames with the mastered episode audio
    to produce the final video. Uses FFmpeg for encoding.

    This extends the existing SignalLostVideo pattern:
    - SignalLostVideo: procedural CRT art + audio -> MP4
    - ProductionBus: AI-generated visuals + audio -> MP4

    Both can coexist - SignalLostVideo remains the v1.5 fallback,
    ProductionBus is the v2.0 upgrade path.

    CPU-only for compositing; FFmpeg handles encoding.
    """

    CATEGORY = _V2_CATEGORY
    FUNCTION = "assemble"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "bus_log")
    OUTPUT_NODE = True
    DESCRIPTION = (
        "v2.0 Production Bus - Assembles composited visual frames "
        "with mastered audio into a final MP4 video. "
        "CPU-only, uses FFmpeg for encoding."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "script_json": ("STRING", {
                    "multiline": True,
                    "default": "[]",
                    "tooltip": "Script data for timeline scene boundaries",
                }),
                "production_plan_json": ("STRING", {
                    "multiline": True,
                    "default": "{}",
                    "tooltip": "Production plan with visual_plan + voice_assignments",
                }),
                "fps": ("INT", {
                    "default": 24, "min": 1, "max": 60,
                    "tooltip": "Output video frame rate",
                }),
                "output_name": ("STRING", {
                    "default": "otr_v2_episode",
                    "tooltip": "Output filename (without extension)",
                }),
                "preview_mode": (["none", "keyframes", "full"], {
                    "default": "keyframes",
                    "tooltip": "none=skip video, keyframes=1 frame/scene (fast), full=all frames",
                }),
                "encoding_profile": (["preview", "balanced", "quality"], {
                    "default": "preview",
                    "tooltip": "preview=fast/crf23, balanced=medium/crf20, quality=slow/crf18. "
                               "All use yuv420p for maximum playback compatibility.",
                }),
            },
            "optional": {
                "composited_frames": ("IMAGE", {
                    "tooltip": "Visual frames from VisualCompositor",
                }),
                "episode_audio": ("AUDIO", {
                    "tooltip": "Mastered episode audio from EpisodeAssembler",
                }),
                "compositor_log": ("STRING", {
                    "tooltip": "Log from VisualCompositor for diagnostics",
                }),
                "output_subdir": ("STRING", {
                    "default": "",
                    "tooltip": "Subdirectory under ComfyUI output root. "
                               "Empty = write to output root. e.g. 'otr_v2' -> output/otr_v2/",
                }),
                "debug_vram_snapshots": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Emit VRAM_SNAPSHOT markers at entry/exit of every phase. "
                               "Enable for repeatable QA on 16 GB hardware.",
                }),
            },
        }

    def assemble(self, script_json="[]", production_plan_json="{}",
                 fps=24, output_name="otr_v2_episode",
                 preview_mode="keyframes", encoding_profile="preview",
                 composited_frames=None, episode_audio=None,
                 compositor_log=None, output_subdir="",
                 debug_vram_snapshots=False):

        import subprocess
        from PIL import Image

        log.info("[ProductionBus] Starting v2.0 assembly")

        if debug_vram_snapshots:
            _vram_snapshot("ProductionBus-entry")

        # Use ComfyUI-native output resolution (Bug Bible 01.02/01.03)
        try:
            import folder_paths
            output_dir = folder_paths.get_output_directory()
        except ImportError:
            # Fallback for standalone testing outside ComfyUI
            output_dir = os.path.normpath(os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "..", "..", "output"
            ))
        if output_subdir:
            output_dir = os.path.join(output_dir, _safe_name(output_subdir))
        os.makedirs(output_dir, exist_ok=True)

        # If we have no visual frames, return early with status
        if composited_frames is None:
            return ("", "ProductionBus: Waiting for composited frames from VisualCompositor...")

        num_frames = composited_frames.shape[0]
        frame_h = composited_frames.shape[1]
        frame_w = composited_frames.shape[2]

        # Parse scene data to calculate frame timing
        try:
            script_lines = json.loads(script_json)
        except json.JSONDecodeError:
            script_lines = []

        # Calculate audio duration if available
        audio_duration_s = 0.0
        audio_path = None
        if episode_audio is not None:
            try:
                # ComfyUI AUDIO type is a dict with "waveform" and "sample_rate"
                waveform = episode_audio["waveform"]
                sample_rate = episode_audio["sample_rate"]
                audio_duration_s = waveform.shape[-1] / sample_rate
                log.info("[ProductionBus] Audio: %.1fs at %dHz", audio_duration_s, sample_rate)

                # Save audio to temp WAV for FFmpeg
                import torchaudio
                audio_path = os.path.join(output_dir, f"{_safe_name(output_name)}_temp_audio.wav")
                # waveform shape: [batch, channels, samples] or [channels, samples]
                if waveform.dim() == 3:
                    waveform_save = waveform[0]
                else:
                    waveform_save = waveform
                torchaudio.save(audio_path, waveform_save.cpu(), sample_rate)
            except Exception as e:
                log.warning("[ProductionBus] Audio processing failed: %s", e)
                audio_duration_s = num_frames / fps  # Fallback

        if audio_duration_s == 0:
            audio_duration_s = max(10.0, num_frames * 5.0)  # 5s per scene minimum

        # Calculate how many output frames we need
        total_frames = int(audio_duration_s * fps)
        frames_per_scene = max(1, total_frames // num_frames)

        # --- preview_mode gate ---
        if preview_mode == "none":
            log.info("[ProductionBus] preview_mode=none, skipping video render")
            return ("", "ProductionBus: preview_mode=none, video skipped.")

        # In keyframes mode, write 1 frame per scene (fast preview).
        # In full mode, write every frame (production quality).
        if preview_mode == "keyframes":
            effective_fps = max(1, num_frames / max(1.0, audio_duration_s))
            frames_per_scene = 1
            total_frames = num_frames
            log.info("[ProductionBus] keyframes mode: %d frames (1 per scene)", total_frames)
        else:
            log.info("[ProductionBus] full mode: %d total frames (%d per scene, %.1fs)",
                     total_frames, frames_per_scene, audio_duration_s)

        # Hard cap to prevent disk bombs (~12 min at 24 fps)
        MAX_FRAMES = 18000
        if total_frames > MAX_FRAMES:
            raise RuntimeError(
                f"Refusing to write {total_frames} frames (cap {MAX_FRAMES}). "
                f"Use preview_mode=keyframes for long episodes."
            )

        # Use a project-local preview_tmp/ so leftovers are obvious, not hidden in %TEMP%
        preview_root = os.path.join(output_dir, "preview_tmp")
        os.makedirs(preview_root, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix="otr_v2_", dir=preview_root)
        video_path = ""

        try:
            frame_idx = 0
            for scene_idx in range(num_frames):
                # Get scene frame
                frame_np = (composited_frames[scene_idx].cpu().numpy() * 255).astype(np.uint8)
                frame_pil = Image.fromarray(frame_np, "RGB")

                # Repeat this frame for its duration
                for _ in range(frames_per_scene):
                    if frame_idx >= total_frames:
                        break
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                    frame_pil.save(frame_path)
                    frame_idx += 1

            # Pad remaining frames with last scene (full mode only)
            if frame_idx < total_frames and num_frames > 0:
                last_np = (composited_frames[-1].cpu().numpy() * 255).astype(np.uint8)
                last_pil = Image.fromarray(last_np, "RGB")
                while frame_idx < total_frames:
                    frame_path = os.path.join(temp_dir, f"frame_{frame_idx:06d}.png")
                    last_pil.save(frame_path)
                    frame_idx += 1

            # Encode with FFmpeg — encoding_profile controls preset/crf,
            # yuv420p is always used for maximum playback compatibility.
            _ENC_PROFILES = {
                "preview":  ("fast",   "23"),
                "balanced": ("medium", "20"),
                "quality":  ("slow",   "18"),
            }
            enc_preset, enc_crf = _ENC_PROFILES.get(encoding_profile, ("fast", "23"))

            video_path = os.path.join(output_dir, f"{_safe_name(output_name)}.mp4")
            render_fps = fps if preview_mode == "full" else max(1, num_frames // max(1, int(audio_duration_s)))
            ffmpeg_cmd = [
                "ffmpeg", "-y",
                "-framerate", str(render_fps if preview_mode == "keyframes" else fps),
                "-i", os.path.join(temp_dir, "frame_%06d.png"),
            ]

            if audio_path and os.path.exists(audio_path):
                ffmpeg_cmd.extend(["-i", audio_path])
                ffmpeg_cmd.extend([
                    "-c:v", "libx264",
                    "-preset", enc_preset,
                    "-crf", enc_crf,
                    "-pix_fmt", "yuv420p",
                    "-c:a", "aac",
                    "-b:a", "192k",
                    "-shortest",
                    video_path
                ])
            else:
                ffmpeg_cmd.extend([
                    "-c:v", "libx264",
                    "-preset", enc_preset,
                    "-crf", enc_crf,
                    "-pix_fmt", "yuv420p",
                    video_path
                ])

            log.info("[ProductionBus] FFmpeg encoding: %s", " ".join(ffmpeg_cmd[-4:]))
            try:
                result = subprocess.run(
                    ffmpeg_cmd, capture_output=True, text=True, timeout=300
                )
                if result.returncode != 0:
                    log.error("[ProductionBus] FFmpeg failed: %s", result.stderr[-500:])
                    video_path = ""
            except Exception as e:
                log.error("[ProductionBus] FFmpeg error: %s", e)
                video_path = ""

        finally:
            # Crash-safe cleanup: always runs, even on OOM or kill
            shutil.rmtree(temp_dir, ignore_errors=True)
            if audio_path and os.path.exists(audio_path):
                os.remove(audio_path)

        # Build log
        bus_lines = [
            f"Production Bus: v2.0 Assembly Complete",
            f"  Output: {video_path}" if video_path else "  Output: FAILED",
            f"  Duration: {audio_duration_s:.1f}s ({total_frames} frames at {fps}fps)",
            f"  Scenes: {num_frames}",
            f"  Resolution: {frame_w}x{frame_h}",
        ]
        if compositor_log:
            bus_lines.append(f"\n--- Compositor Report ---\n{compositor_log}")

        # Machine-readable QA metrics line for regression parsing
        qa_metrics = {
            "preview_mode": preview_mode,
            "audio_duration_s": round(audio_duration_s, 2),
            "total_frames": total_frames,
            "resolution": f"{frame_w}x{frame_h}",
            "video_path_status": "ok" if video_path else "failed",
            "fps": fps,
        }
        bus_lines.append(f"\nQA_METRICS: {json.dumps(qa_metrics)}")
        bus_log = "\n".join(bus_lines)

        if debug_vram_snapshots:
            _vram_snapshot("ProductionBus-exit")

        log.info("[ProductionBus] Complete: %s", video_path or "FAILED")
        return (video_path, bus_log)
