"""
shotlist.py  --  Deterministic OTR script_lines -> shots[] mapper
=================================================================
Reads the Canonical Audio Tokens produced by _parse_script() and
produces a shots[] array suitable for driving the HyWorld sidecar
(or any interim stand-in: Diffusion360, SPAG4D, ComfyUI-Sharp).

Design doc: docs/superpowers/specs/2026-04-15-otr-to-hyworld-narrative-mapping.md
    Sections 4 (per-token mapping), 5 (deterministic vs LLM), 12.2 (Lane 1)

Rules:
    - Pure functions.  No GPU, no model loads, no network.
    - Deterministic.  Same script_lines in -> same shots[] out.
    - No v1.7 imports.  Reads JSON, returns JSON.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from typing import Any

log = logging.getLogger("OTR.hyworld.shotlist")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_SHOT_DURATION_SEC = 12   # C4 ceiling
MIN_SHOT_DURATION_SEC = 3    # minimum viable clip
DEFAULT_SHOT_DURATION_SEC = 9
MAX_SHOTS_PER_SCENE = 4      # VRAM predictability

# Voice-traits -> camera adjective lookup  (Section 4.5)
_TRAIT_CAMERA_MAP: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(?:weary|tired|old|exhausted)\b", re.I), "slow handheld, close"),
    (re.compile(r"\b(?:angry|hostile|sharp|furious)\b", re.I), "fast dolly, canted"),
    (re.compile(r"\b(?:calm|warm|gentle|serene)\b", re.I), "locked off, wide"),
    (re.compile(r"\b(?:frantic|panicked|manic)\b", re.I), "whip-pan, short focal length"),
    (re.compile(r"\b(?:announcer|formal|narrator)\b", re.I), "clean push-in, centered"),
    (re.compile(r"\b(?:child|young|teen)\b", re.I), "low angle, looking up"),
    (re.compile(r"\b(?:whisper|hushed|quiet)\b", re.I), "macro detail, shallow focus"),
]
_DEFAULT_CAMERA = "slow drift, medium lens"


def _camera_from_traits(voice_traits: str) -> str:
    """Map voice_traits string to a camera adjective via first-match lookup."""
    for pattern, camera in _TRAIT_CAMERA_MAP:
        if pattern.search(voice_traits):
            return camera
    return _DEFAULT_CAMERA


def _mood_from_traits(voice_traits: str) -> str:
    """Extract the dominant mood keyword from voice_traits."""
    for pattern, _ in _TRAIT_CAMERA_MAP:
        m = pattern.search(voice_traits)
        if m:
            return m.group(0).lower()
    return "neutral"


def _style_anchor_hash(episode_title: str, first_env: str) -> str:
    """12-char hex hash for episode-wide visual consistency."""
    payload = f"{episode_title}:{first_env}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Scene splitter
# ---------------------------------------------------------------------------

def _split_into_scenes(script_lines: list[dict]) -> list[dict]:
    """
    Split flat script_lines into scenes.

    Returns a list of scene dicts:
        {
            "scene_id": str,
            "tokens": [token, ...],
            "env_descriptions": [str, ...],
            "dialogue_lines": [{character_name, voice_traits, line, index}, ...],
            "sfx_cues": [{description, index}, ...],
            "beat_count": int,
        }
    """
    scenes: list[dict] = []
    current: dict | None = None

    def _new_scene(label: str) -> dict:
        return {
            "scene_id": label,
            "tokens": [],
            "env_descriptions": [],
            "dialogue_lines": [],
            "sfx_cues": [],
            "beat_count": 0,
        }

    for idx, token in enumerate(script_lines):
        ttype = token.get("type", "")

        if ttype == "title":
            continue  # skip title token, not scene-relevant

        if ttype == "scene_break":
            if current is not None:
                scenes.append(current)
            current = _new_scene(token.get("scene", str(len(scenes) + 1)))
            continue

        # If we haven't seen a scene_break yet, start an implicit Scene 1
        if current is None:
            current = _new_scene("1")

        current["tokens"].append(token)

        if ttype == "environment":
            current["env_descriptions"].append(token.get("description", ""))
        elif ttype == "dialogue":
            current["dialogue_lines"].append({
                "character_name": token.get("character_name", "UNKNOWN"),
                "voice_traits": token.get("voice_traits", ""),
                "line": token.get("line", ""),
                "index": idx,
            })
        elif ttype == "sfx":
            current["sfx_cues"].append({
                "description": token.get("description", ""),
                "index": idx,
            })
        elif ttype == "pause":
            current["beat_count"] += 1

    if current is not None:
        scenes.append(current)

    return scenes


# ---------------------------------------------------------------------------
# Shot builder
# ---------------------------------------------------------------------------

def _build_shots_for_scene(scene: dict, scene_idx: int) -> list[dict]:
    """
    Build 1-4 shots for a single scene using deterministic rules.

    Each shot gets:
        shot_id, scene_ref, duration_sec, camera, env_prompt,
        sfx_accents, dialogue_line_ids, mood, visual_backend
    """
    env_prompt = scene["env_descriptions"][0] if scene["env_descriptions"] else "empty room, dim light"
    dialogue = scene["dialogue_lines"]
    sfx_cues = scene["sfx_cues"]

    # Estimate scene audio duration from token count
    # Rough heuristic: 2.5s per dialogue line, 1s per SFX, 0.2s per beat
    est_duration = (
        len(dialogue) * 2.5
        + len(sfx_cues) * 1.0
        + scene["beat_count"] * 0.2
    )
    est_duration = max(MIN_SHOT_DURATION_SEC, est_duration)

    # Determine dominant mood from first dialogue's voice_traits
    dominant_traits = dialogue[0]["voice_traits"] if dialogue else ""
    camera = _camera_from_traits(dominant_traits)
    mood = _mood_from_traits(dominant_traits)

    # Split scene into shots (each capped at MAX_SHOT_DURATION_SEC)
    num_shots = min(
        MAX_SHOTS_PER_SCENE,
        max(1, int(est_duration / DEFAULT_SHOT_DURATION_SEC + 0.5)),
    )
    shot_duration = min(MAX_SHOT_DURATION_SEC, max(MIN_SHOT_DURATION_SEC, est_duration / num_shots))

    shots: list[dict] = []
    dl_cursor = 0
    sfx_cursor = 0

    for shot_num in range(num_shots):
        shot_id = f"s{scene_idx + 1:02d}_{shot_num + 1:02d}"

        # Distribute dialogue lines across shots
        dl_per_shot = max(1, len(dialogue) // num_shots) if dialogue else 0
        dl_start = dl_cursor
        dl_end = min(len(dialogue), dl_cursor + dl_per_shot) if shot_num < num_shots - 1 else len(dialogue)
        shot_dialogue_ids = [f"line_{d['index']}" for d in dialogue[dl_start:dl_end]]
        dl_cursor = dl_end

        # Update camera if this chunk has a different mood
        if dl_start < len(dialogue):
            chunk_traits = dialogue[dl_start]["voice_traits"]
            shot_camera = _camera_from_traits(chunk_traits)
            shot_mood = _mood_from_traits(chunk_traits)
        else:
            shot_camera = camera
            shot_mood = mood

        # Distribute SFX accents
        sfx_per_shot = max(1, len(sfx_cues) // num_shots) if sfx_cues else 0
        sfx_start = sfx_cursor
        sfx_end = min(len(sfx_cues), sfx_cursor + sfx_per_shot) if shot_num < num_shots - 1 else len(sfx_cues)
        shot_sfx = [
            {"at": round((i - sfx_start) * (shot_duration / max(1, sfx_end - sfx_start)), 1),
             "desc": s["description"]}
            for i, s in enumerate(sfx_cues[sfx_start:sfx_end])
        ]
        sfx_cursor = sfx_end

        # Use secondary env descriptions for later shots if available
        shot_env = scene["env_descriptions"][min(shot_num, len(scene["env_descriptions"]) - 1)] if scene["env_descriptions"] else env_prompt

        shots.append({
            "shot_id": shot_id,
            "scene_ref": scene["scene_id"],
            "duration_sec": round(shot_duration, 1),
            "camera": shot_camera,
            "env_prompt": shot_env,
            "sfx_accents": shot_sfx,
            "dialogue_line_ids": shot_dialogue_ids,
            "mood": shot_mood,
            "visual_backend": {
                "pano": "diffusion360",
                "stereo": "spag4d_da360",
                "nav": "hand_authored",
            },
        })

    return shots


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_shotlist(
    script_lines: list[dict],
    episode_title: str = "Untitled Episode",
) -> dict[str, Any]:
    """
    Main entry point.  Deterministic Lane 1 mapping.

    Args:
        script_lines: the Canonical Audio Token array from _parse_script().
        episode_title: used for style_anchor_hash generation.

    Returns:
        A dict with keys:
            "shots": list of shot dicts
            "style_anchor_hash": 12-char hex
            "scene_count": int
            "total_shots": int
    """
    scenes = _split_into_scenes(script_lines)

    all_shots: list[dict] = []
    for idx, scene in enumerate(scenes):
        scene_shots = _build_shots_for_scene(scene, idx)
        all_shots.extend(scene_shots)

    # Style anchor from first environment in Act 1
    first_env = ""
    for scene in scenes:
        if scene["env_descriptions"]:
            first_env = scene["env_descriptions"][0]
            break

    return {
        "shots": all_shots,
        "style_anchor_hash": _style_anchor_hash(episode_title, first_env),
        "scene_count": len(scenes),
        "total_shots": len(all_shots),
    }


def generate_shotlist_json(
    script_lines_json: str,
    episode_title: str = "Untitled Episode",
) -> str:
    """Convenience wrapper: JSON string in, JSON string out."""
    lines = json.loads(script_lines_json)
    result = generate_shotlist(lines, episode_title)
    return json.dumps(result, indent=2)
