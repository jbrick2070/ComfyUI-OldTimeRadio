"""
SceneSegmenter — Chunks dialogue lines into visual scenes.

Rule: if len(lines) > 8, chunk every 4 lines. Else single scene.
Scene IDs are zero-padded and stable across runs.

Segmenter owns scene *count*. Director is *advisory* for scene *meaning*.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("OTR")

MAX_LINES_PER_SCENE = 4
TRIGGER_THRESHOLD = 8


@dataclass
class Scene:
    """A visual scene derived from dialogue segmentation."""
    scene_id: str                  # s01, s02, ...
    line_indices: list             # Indices into the original script_json
    dialogue: list                 # List of dialogue text strings


def segment(lines: list) -> list:
    """Segment dialogue lines into scenes.

    Args:
        lines: List of dialogue line strings, or list of dicts with
               'type' and 'line' keys (script_json format).

    Returns:
        List of Scene dataclass instances.
    """
    # Normalize input: extract dialogue-only lines
    dialogue_entries = []
    for i, entry in enumerate(lines):
        if isinstance(entry, dict):
            if entry.get("type") != "dialogue":
                continue
            text = entry.get("line", "")
        elif isinstance(entry, str):
            text = entry
        else:
            continue
        dialogue_entries.append({"index": i, "text": text})

    total = len(dialogue_entries)
    log.info("[SceneSegmenter] %d dialogue lines from %d total entries",
             total, len(lines))

    # Apply chunking rule
    if total > TRIGGER_THRESHOLD:
        chunks = [
            dialogue_entries[i:i + MAX_LINES_PER_SCENE]
            for i in range(0, total, MAX_LINES_PER_SCENE)
        ]
    else:
        chunks = [dialogue_entries] if dialogue_entries else []

    # Build Scene list
    scenes = []
    for idx, chunk in enumerate(chunks):
        if not chunk:
            continue
        scenes.append(Scene(
            scene_id=f"s{idx + 1:02d}",
            line_indices=[e["index"] for e in chunk],
            dialogue=[e["text"] for e in chunk],
        ))

    log.info("[SceneSegmenter] Produced %d scenes", len(scenes))
    return scenes


def segment_from_json(script_json_str: str) -> list:
    """Convenience: parse JSON string, then segment.

    Args:
        script_json_str: JSON string of script lines.

    Returns:
        List of Scene dataclass instances.
    """
    try:
        lines = json.loads(script_json_str) if isinstance(
            script_json_str, str) else script_json_str
    except (json.JSONDecodeError, TypeError):
        lines = []
    return segment(lines)
