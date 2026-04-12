"""
SceneSegmenter — Chunks dialogue lines into visual scenes.

Rule from V2_actionplan.md Section 7:
  - If total dialogue lines > TRIGGER_THRESHOLD (8), force a scene
    break every MAX_LINES_PER_SCENE (4) lines.
  - Otherwise, treat the entire script as one scene.

This prevents talking-head loop syndrome where a single static
frame holds for 2+ minutes of dialogue.

Not a ComfyUI node — called by SceneAnimator and PromptBuilder.
"""

import json
import logging
from typing import Any

log = logging.getLogger("OTR")

MAX_LINES_PER_SCENE = 4
TRIGGER_THRESHOLD = 8


def segment_scenes(script_json_str, production_plan_json_str="{}"):
    """Break a script into visual scenes for animation.

    Args:
        script_json_str: JSON string of script lines (list of dicts,
            each with at minimum 'type' and optionally 'character_name',
            'line', 'voice_traits').
        production_plan_json_str: JSON string of production plan. If it
            contains visual_plan.scenes, those scene boundaries are used
            as hints (but dialogue chunking still applies within them).

    Returns:
        List of scene dicts, each containing:
            scene_id: "s01", "s02", ...
            dialogue_lines: list of dialogue line dicts in this scene
            line_range: (start_idx, end_idx) in the original script
            duration_hint_s: estimated duration based on line count
    """
    # Parse inputs
    try:
        script_lines = json.loads(script_json_str) if isinstance(script_json_str, str) else script_json_str
    except (json.JSONDecodeError, TypeError):
        script_lines = []

    try:
        plan = json.loads(production_plan_json_str) if isinstance(production_plan_json_str, str) else production_plan_json_str
    except (json.JSONDecodeError, TypeError):
        plan = {}

    # Extract only dialogue lines (skip stage directions, SFX, etc.)
    dialogue_lines = []
    for i, line in enumerate(script_lines):
        if not isinstance(line, dict):
            continue
        if line.get("type") == "dialogue":
            dialogue_lines.append({"index": i, **line})

    total = len(dialogue_lines)
    log.info("[SceneSegmenter] %d dialogue lines from %d total script entries",
             total, len(script_lines))

    # Apply chunking rule
    if total > TRIGGER_THRESHOLD:
        chunks = _chunk_lines(dialogue_lines, MAX_LINES_PER_SCENE)
    else:
        chunks = [dialogue_lines] if dialogue_lines else []

    # Build scene list
    scenes = []
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        scene_id = f"s{idx + 1:02d}"
        start = chunk[0]["index"]
        end = chunk[-1]["index"]
        # Rough duration estimate: ~4.5 seconds per dialogue line (Bark TTS average)
        duration_hint = len(chunk) * 4.5
        scenes.append({
            "scene_id": scene_id,
            "dialogue_lines": chunk,
            "line_range": [start, end],
            "duration_hint_s": round(duration_hint, 1),
            "num_lines": len(chunk),
        })

    # Merge with Director's scene data if available
    visual_scenes = plan.get("visual_plan", {}).get("scenes", [])
    if visual_scenes and len(visual_scenes) == len(scenes):
        for scene, vs in zip(scenes, visual_scenes):
            scene["visual_prompt"] = vs.get("visual_prompt",
                                             vs.get("shot_description", ""))
            scene["motion"] = vs.get("motion", "medium")
    elif visual_scenes:
        # Scene count mismatch — redistribute visual prompts
        for i, scene in enumerate(scenes):
            vs_idx = min(i, len(visual_scenes) - 1)
            scene["visual_prompt"] = visual_scenes[vs_idx].get(
                "visual_prompt", visual_scenes[vs_idx].get("shot_description", ""))
            scene["motion"] = visual_scenes[vs_idx].get("motion", "medium")

    log.info("[SceneSegmenter] Produced %d scenes (rule: %d lines/scene, threshold: %d)",
             len(scenes), MAX_LINES_PER_SCENE, TRIGGER_THRESHOLD)

    return scenes


def _chunk_lines(lines, chunk_size):
    """Split a list into chunks of at most chunk_size."""
    return [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
