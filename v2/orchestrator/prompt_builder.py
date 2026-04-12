"""
PromptBuilder — Constructs the full prompt matrix before any GPU work.

V2_actionplan.md Section 2 requires all prompt assembly to complete
BEFORE the ComfyUI queue runs. A stalled API call during the memory
boundary will drop the CUDA context. So every prompt is a plain string
by the time sampling starts.

Not a ComfyUI node — called by SceneAnimator during its setup phase.
"""

import json
import logging
from typing import Any

log = logging.getLogger("OTR")

# Negative prompt tuned for LTX-Video I2V output quality
_NEG_VIDEO = (
    "blurry, low quality, distorted, artifacts, flickering, "
    "watermark, text, static image, frozen, jittery, "
    "worst quality, jpeg artifacts"
)

# Anchor negative (SD3.5 still generation)
_NEG_ANCHOR = (
    "blurry, low quality, deformed, ugly, watermark, text, "
    "signature, worst quality, jpeg artifacts, cartoon, "
    "anime, illustration"
)


def build_prompt_matrix(scenes, production_plan=None, characters=None):
    """Build anchor_prompt + motion_prompt for every scene upfront.

    Args:
        scenes: List of scene dicts from SceneSegmenter.
        production_plan: Parsed production plan dict (optional).
        characters: Dict of character name -> character data (optional).

    Returns:
        List of prompt dicts, one per scene:
        {
            "scene_id": "s01",
            "anchor_prompt": "...",      # For SD3.5 still generation
            "motion_prompt": "...",      # For LTX-Video I2V animation
            "negative_anchor": "...",
            "negative_motion": "...",
            "motion": "high|medium|low|static",
            "duration_s": 3.0,
            "character_tokens": []       # Future: textual inversion tokens
        }
    """
    if production_plan is None:
        production_plan = {}
    if characters is None:
        characters = production_plan.get("visual_plan", {}).get("characters", {})

    # Extract episode-level context
    episode_title = production_plan.get("episode_title", "Untitled")
    vintage = production_plan.get("vintage_settings", {})
    era_hint = vintage.get("era", "1950s")

    prompts = []
    for scene in scenes:
        scene_id = scene.get("scene_id", "s00")
        lines = scene.get("dialogue_lines", [])
        visual_prompt = scene.get("visual_prompt", "")
        motion_tag = scene.get("motion", "medium")
        duration = scene.get("duration_hint_s", 3.0)

        # Build scene context from dialogue content
        speakers = []
        dialogue_summary = []
        for dl in lines[:4]:  # Cap at 4 lines for prompt length
            name = dl.get("character_name", "UNKNOWN")
            text = dl.get("line", "")[:80]
            if name not in speakers:
                speakers.append(name)
            dialogue_summary.append(f"{name}: {text}")

        scene_context = "; ".join(dialogue_summary) if dialogue_summary else "establishing shot"

        # Character appearance hints
        char_hints = []
        for name in speakers:
            if name in characters:
                desc = characters[name].get("portrait_prompt", "")
                if desc:
                    char_hints.append(f"{name}: {desc}")

        char_block = ". ".join(char_hints) if char_hints else ""

        # --- Anchor prompt (SD3.5 still image) ---
        if visual_prompt:
            anchor_base = visual_prompt
        else:
            anchor_base = _infer_scene_visual(scene_context, era_hint)

        anchor_prompt = (
            f"Cinematic establishing shot, {anchor_base}, "
            f"professional cinematography, dramatic lighting, "
            f"film grain, anamorphic lens, photorealistic, 8K detail"
        )
        if char_block:
            anchor_prompt = f"{anchor_prompt}. Characters: {char_block}"

        # --- Motion prompt (LTX-Video I2V animation) ---
        motion_desc = _motion_descriptor(motion_tag)
        motion_prompt = (
            f"{motion_desc} cinematic scene, {anchor_base}, "
            f"{_era_style(era_hint)}, "
            f"smooth camera movement, professional film quality"
        )

        prompts.append({
            "scene_id": scene_id,
            "anchor_prompt": anchor_prompt,
            "motion_prompt": motion_prompt,
            "negative_anchor": _NEG_ANCHOR,
            "negative_motion": _NEG_VIDEO,
            "motion": motion_tag,
            "duration_s": duration,
            "character_tokens": [],  # Future: textual inversion refs
        })

    log.info("[PromptBuilder] Built %d prompt pairs (anchor + motion)", len(prompts))
    return prompts


def _infer_scene_visual(dialogue_context, era_hint):
    """Generate a visual description from dialogue content when the
    Director did not provide an explicit visual_prompt."""
    # Simple keyword-based inference — good enough for RC1
    ctx = dialogue_context.lower()
    if any(w in ctx for w in ("station", "relay", "console", "terminal")):
        return f"dimly lit {era_hint} space station control room with blinking consoles"
    elif any(w in ctx for w in ("ship", "bridge", "cockpit", "pilot")):
        return f"interior of a {era_hint} spacecraft bridge with instrument panels"
    elif any(w in ctx for w in ("outside", "surface", "planet", "landscape")):
        return f"alien planet surface with dramatic sky, {era_hint} sci-fi aesthetic"
    elif any(w in ctx for w in ("lab", "science", "experiment", "research")):
        return f"{era_hint} science laboratory with vintage equipment and green CRT screens"
    else:
        return f"atmospheric {era_hint} science fiction interior with moody lighting"


def _motion_descriptor(motion_tag):
    """Map motion budget tag to a prompt-friendly motion description."""
    descriptors = {
        "high": "Dynamic action sequence with significant movement,",
        "medium": "Subtle atmospheric motion with gentle camera drift,",
        "low": "Near-static scene with barely perceptible ambient movement,",
        "static": "Still photograph with minimal parallax effect,",
    }
    return descriptors.get(motion_tag, descriptors["medium"])


def _era_style(era):
    """Return era-appropriate style keywords for the motion prompt."""
    era_lower = str(era).lower()
    if "1940" in era_lower or "1950" in era_lower:
        return "vintage film noir aesthetic, high contrast black and white"
    elif "1960" in era_lower or "1970" in era_lower:
        return "retro technicolor look, warm film grain"
    else:
        return "classic radio drama aesthetic, warm amber tones, film grain"
