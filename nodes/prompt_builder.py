"""
PromptBuilder — Fully-materialized prompt matrix before any queue submission.

All network I/O and prompt assembly completes BEFORE ComfyUI queue starts.
A stalled API call during the memory boundary will drop the CUDA context.

Assertions at exit: every field populated, all strings, no None except
director_hint, no coroutines/futures.
"""

import json
import logging
from dataclasses import dataclass
from typing import Optional

log = logging.getLogger("OTR")


@dataclass
class ScenePrompt:
    """Fully-materialized prompt pair for one scene."""
    scene_id: str
    anchor_prompt: str
    motion_prompt: str
    motion: str                     # static | low | medium | high
    duration_s: float
    character_tokens: list          # Future: textual inversion refs
    director_hint: Optional[str]    # None is OK here only


_NEG_ANCHOR = (
    "blurry, low quality, deformed, ugly, watermark, text, "
    "signature, worst quality, jpeg artifacts, cartoon, anime, illustration"
)

_NEG_VIDEO = (
    "blurry, low quality, distorted, artifacts, flickering, "
    "watermark, text, static image, frozen, jittery, worst quality"
)


def build(scenes: list, director_json: dict,
          audio_timings: dict = None) -> list:
    """Build fully-materialized prompt matrix.

    Args:
        scenes: List of reconciled scene dicts (from director_reconciler).
            Each has scene_id, dialogue, visual_prompt, motion, line_indices.
        director_json: Full Director production plan dict.
        audio_timings: Optional dict mapping scene_id -> duration_s from audio.

    Returns:
        List of ScenePrompt dataclass instances.
    """
    if audio_timings is None:
        audio_timings = {}

    characters = director_json.get("visual_plan", {}).get("characters", {})
    vintage = director_json.get("vintage_settings", {})
    era_hint = vintage.get("era", "1950s")

    prompts = []
    for scene in scenes:
        scene_id = scene.get("scene_id", "s00")
        dialogue = scene.get("dialogue", [])
        visual_prompt = scene.get("visual_prompt", "")
        motion_tag = scene.get("motion", "medium")
        director_hint = visual_prompt or None

        # Duration from audio timings or estimate from dialogue count
        duration_s = audio_timings.get(scene_id, len(dialogue) * 4.5)

        # Build scene context from dialogue
        context = "; ".join(d[:80] for d in dialogue[:4]) if dialogue else "establishing shot"

        # Anchor prompt (SD3.5 still)
        if visual_prompt:
            anchor_base = visual_prompt
        else:
            anchor_base = _infer_visual(context, era_hint)

        anchor_prompt = (
            f"Cinematic establishing shot, {anchor_base}, "
            f"professional cinematography, dramatic lighting, "
            f"film grain, anamorphic lens, photorealistic, 8K detail"
        )

        # Motion prompt (LTX-Video I2V)
        motion_desc = _MOTION_MAP.get(motion_tag, _MOTION_MAP["medium"])
        motion_prompt = (
            f"{motion_desc} cinematic scene, {anchor_base}, "
            f"{_era_style(era_hint)}, "
            f"smooth camera movement, professional film quality"
        )

        sp = ScenePrompt(
            scene_id=scene_id,
            anchor_prompt=anchor_prompt,
            motion_prompt=motion_prompt,
            motion=motion_tag,
            duration_s=round(duration_s, 1),
            character_tokens=[],
            director_hint=director_hint,
        )

        # Assertion: every field populated, all strings where expected
        assert isinstance(sp.scene_id, str) and sp.scene_id
        assert isinstance(sp.anchor_prompt, str) and sp.anchor_prompt
        assert isinstance(sp.motion_prompt, str) and sp.motion_prompt
        assert isinstance(sp.motion, str) and sp.motion in ("static", "low", "medium", "high")
        assert isinstance(sp.duration_s, float) and sp.duration_s > 0
        assert isinstance(sp.character_tokens, list)

        prompts.append(sp)

    log.info("[PromptBuilder] Built %d scene prompts", len(prompts))
    return prompts


_MOTION_MAP = {
    "high": "Dynamic action sequence with significant movement,",
    "medium": "Subtle atmospheric motion with gentle camera drift,",
    "low": "Near-static scene with barely perceptible ambient movement,",
    "static": "Still photograph with minimal parallax effect,",
}


def _infer_visual(context: str, era: str) -> str:
    """Generate visual description from dialogue context."""
    ctx = context.lower()
    if any(w in ctx for w in ("station", "relay", "console", "terminal")):
        return f"dimly lit {era} space station control room with blinking consoles"
    elif any(w in ctx for w in ("ship", "bridge", "cockpit", "pilot")):
        return f"interior of a {era} spacecraft bridge with instrument panels"
    elif any(w in ctx for w in ("outside", "surface", "planet", "landscape")):
        return f"alien planet surface with dramatic sky, {era} sci-fi aesthetic"
    elif any(w in ctx for w in ("lab", "science", "experiment")):
        return f"{era} science laboratory with vintage equipment and green CRT screens"
    else:
        return f"atmospheric {era} science fiction interior with moody lighting"


def _era_style(era: str) -> str:
    """Era-appropriate style keywords."""
    e = str(era).lower()
    if "1940" in e or "1950" in e:
        return "vintage film noir aesthetic, high contrast black and white"
    elif "1960" in e or "1970" in e:
        return "retro technicolor look, warm film grain"
    else:
        return "classic radio drama aesthetic, warm amber tones, film grain"
