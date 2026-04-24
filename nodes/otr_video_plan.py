"""
nodes.otr_video_plan  --  OTR_VideoPlan read-only adapter
==========================================================

MIT-licensed, OTR-original.  Read-only adapter: parses the Director
JSON (``_DIRECTOR_SCHEMA`` per nodes/story_orchestrator.py:6814) plus
the raw script text into a structured list of FLUX render prompts
suitable for OTR_BatchFluxRender.

This is the "Ideas / multi-pass FLUX" machinery from the
otr-video-node-anatomy artifact, implemented in its thinnest form:

    PASS 1: one character's portrait_prompt (rendered once, reused)
    PASS 2: each scene's visual_prompt (env render, reused)
    PASS 3: composed start/end frames per shot, sharing frames at
            scene boundaries (shot_N_end IS shot_N+1_start)

Output mode
-----------
Emits a JSON envelope identical to what ``visual/prompt_coercion.py``
produces for OTR_BatchFluxRender, so the existing env-token parser
eats it as-is with no BatchFluxRender changes:

    {
      "tokens": [
        {"type": "environment", "description": "<composed prompt>",
         "role": "char_scene_composite",
         "shot_id": "scene_01_start_0", ...},
        ...
      ],
      "source": "OTR_VideoPlan",
      "focus_character": "BABA",
      ...
    }

Zero torch / diffusers / GPU dependencies.  Pure Python stdlib.
Unit-testable without ComfyUI installed.

Scope notes (2026-04-23)
------------------------
This first cut implements Jeffrey's "one character, enough start/end
frames for a whole episode" thin slice:

* Takes ONE focus_character name (e.g. "BABA")
* Generates shots_per_scene renders per scene (default 3 per scene)
* For S scenes × K shots per scene, emits S*K start frames + 1 final
  end frame = S*K+1 prompts total
* Pre-composes each prompt as:
      portrait_prompt + ", " + scene.visual_prompt + ", "
      + era_tail[genre_flavor] + ", " + style_tail
* Each prompt carries a ``shot_id`` so downstream save can name
  output files meaningfully when that's wired in

Later additions (not in this cut):
* Multi-character scene assembly (derive chars_in_shot from
  [VOICE:] tags per beat boundary)
* Audio-aligned beat cutting (needs Bark durations; runs after audio)
* FLUX Kontext / OminiControl / ACE++ integration for richer compose

License
-------
MIT.  Part of ComfyUI-OldTimeRadio.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

log = logging.getLogger("OTR.nodes.otr_video_plan")


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_STYLE_TAIL = (
    "cinematic, 35mm film look, 1980s broadcast aesthetic, "
    "subtle film grain, volumetric lighting"
)

# Era-appropriate tails keyed by OTR's genre_flavor dropdown options.
# Fed into every composed prompt so FLUX dresses characters for the
# right era without requiring per-scene costume prose.
_ERA_TAIL_BY_GENRE: dict[str, str] = {
    "hard_sci_fi":       "near-future industrial sci-fi, clean lines, utilitarian",
    "space_opera":       "grand operatic space-opera scale, ornate costumes, vibrant",
    "dystopian":         "oppressive dystopian regime, drab uniforms, concrete",
    "time_travel":       "anachronistic mix, period-appropriate to the scene",
    "first_contact":     "scientific sublime, clean institutional aesthetic",
    "cosmic_horror":     "1920s Lovecraftian era, weathered, shadowed",
    "cyberpunk":         "1980s neon cyberpunk aesthetic, rain-slick streets, chrome",
    "post_apocalyptic":  "post-apocalyptic scavenger chic, worn gear, sun-bleached",
}

# Fallback era tail used when genre_flavor is empty, unknown, or
# when the caller explicitly passes an override.
_DEFAULT_ERA_TAIL = "timeless cinematic aesthetic"


# ---------------------------------------------------------------------------
# Pure helpers (unit-testable without ComfyUI)
# ---------------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str, max_len: int = 40) -> str:
    """Turn a name into a filesystem-safe slug.

    "BABA"          -> "baba"
    "KENJI CROSS"   -> "kenji_cross"
    "Agent 47 / B"  -> "agent_47_b"
    """
    if not name:
        return "unnamed"
    slug = _SLUG_RE.sub("_", name.lower()).strip("_")
    if not slug:
        return "unnamed"
    return slug[:max_len]


def resolve_era_tail(genre_flavor: str) -> str:
    """Return the era tail for a given genre_flavor, or the default."""
    if not genre_flavor:
        return _DEFAULT_ERA_TAIL
    key = genre_flavor.strip().lower().replace(" ", "_").replace("-", "_")
    return _ERA_TAIL_BY_GENRE.get(key, _DEFAULT_ERA_TAIL)


def resolve_character_portrait(
    director: dict,
    character_name: str,
    style_tail: str,
) -> str:
    """Fallback chain for a character's visual description.

    1. director.visual_plan.characters[NAME].portrait_prompt  (canonical)
    2. Synthesized from voice_assignments[NAME].notes         (fallback)
    3. Generic template using just the name                   (last resort)

    Always returns a non-empty string.
    """
    if not character_name:
        return f"Cinematic portrait of a mysterious figure, {style_tail}"

    visual_plan = director.get("visual_plan") or {}
    characters = visual_plan.get("characters") or {}
    entry = characters.get(character_name) or {}
    portrait = (entry.get("portrait_prompt") or "").strip()
    if portrait:
        return portrait

    # Fallback 2: synthesize from voice_assignments notes
    voice_assignments = director.get("voice_assignments") or {}
    va_entry = voice_assignments.get(character_name) or {}
    notes = (va_entry.get("notes") or "").strip()
    if notes:
        return (
            f"Cinematic portrait of {character_name}, {notes}, {style_tail}"
        )

    # Fallback 3: generic
    log.warning(
        "OTR_VideoPlan: no portrait data for %r; falling back to generic",
        character_name,
    )
    return f"Cinematic portrait of {character_name}, {style_tail}"


def extract_scenes(director: dict) -> list[dict]:
    """Return director.visual_plan.scenes as a list of dicts.

    Each dict has ``scene_id``, ``shot_description``, ``visual_prompt``
    fields (may be missing; callers must handle).  Returns empty list
    if no scenes are present.
    """
    visual_plan = director.get("visual_plan") or {}
    scenes = visual_plan.get("scenes") or []
    if not isinstance(scenes, list):
        log.warning(
            "OTR_VideoPlan: visual_plan.scenes is not a list (%s); ignoring",
            type(scenes).__name__,
        )
        return []
    return [s for s in scenes if isinstance(s, dict)]


def compose_shot_prompt(
    portrait: str,
    scene_visual: str,
    era_tail: str,
    style_tail: str,
    shot_hint: str = "",
) -> str:
    """Concatenate the five prompt layers for PASS 3 composite.

    Order matters: subject (character) first, scene context next,
    then shot-specific framing hint, era tail, style tail.  FLUX
    tends to weight earlier tokens more heavily.
    """
    parts: list[str] = []
    for piece in (portrait, scene_visual, shot_hint, era_tail, style_tail):
        if piece:
            cleaned = piece.strip().rstrip(",").strip()
            if cleaned:
                parts.append(cleaned)
    return ", ".join(parts)


def build_shot_plan(
    director_json: str,
    focus_character: str,
    *,
    shots_per_scene: int = 3,
    genre_flavor: str = "",
    style_tail: str = "",
    include_final_end_frame: bool = True,
) -> dict:
    """Build the per-shot FLUX prompt plan for one character across
    all scenes in the Director JSON.

    Output envelope:

        {
          "tokens": [
            {"type": "environment",
             "description": "<composed prompt>",
             "role": "char_scene_composite",
             "shot_id": "scene_01_s0_start",
             "scene_id": "scene_1",
             "focus_character": "BABA",
             "kind": "start"},
            ...
          ],
          "source": "OTR_VideoPlan",
          "focus_character": "BABA",
          "shots_per_scene": 3,
          "scenes_covered": 5,
          "total_prompts": 16,
          "genre_flavor": "hard_sci_fi",
          "era_tail": "near-future industrial sci-fi, clean lines, utilitarian",
          "style_tail": "cinematic, 35mm film look, ..."
        }

    For ``include_final_end_frame=True``, emits S*K+1 prompts (so the
    FLF chain has an end for the last shot of the last scene).  For
    ``False``, emits exactly S*K prompts.
    """
    if shots_per_scene < 1:
        raise ValueError(
            f"shots_per_scene must be >= 1, got {shots_per_scene}"
        )

    try:
        director = json.loads(director_json) if director_json else {}
    except json.JSONDecodeError as exc:
        log.warning(
            "OTR_VideoPlan: director_json JSONDecodeError (%s); using empty",
            exc,
        )
        director = {}

    if not isinstance(director, dict):
        log.warning(
            "OTR_VideoPlan: director root is %s not dict; using empty",
            type(director).__name__,
        )
        director = {}

    resolved_style_tail = (style_tail or _DEFAULT_STYLE_TAIL).strip()
    era_tail = resolve_era_tail(genre_flavor)
    portrait = resolve_character_portrait(
        director, focus_character, resolved_style_tail
    )

    scenes = extract_scenes(director)
    tokens: list[dict[str, Any]] = []

    char_slug = slugify(focus_character)

    for scene in scenes:
        scene_id = (scene.get("scene_id") or "").strip()
        if not scene_id:
            # Synthesize an id so downstream filenames don't collide
            scene_id = f"scene_{len(tokens)+1:02d}"
        scene_visual = (scene.get("visual_prompt") or "").strip()
        if not scene_visual:
            log.warning(
                "OTR_VideoPlan: scene %r missing visual_prompt; using name only",
                scene_id,
            )
            scene_visual = (
                scene.get("shot_description") or f"{scene_id} environment"
            ).strip()

        for shot_idx in range(shots_per_scene):
            # Simple shot-progression hint: "early", "mid", "late"
            if shots_per_scene == 1:
                shot_hint = ""
            elif shot_idx == 0:
                shot_hint = "establishing framing"
            elif shot_idx == shots_per_scene - 1:
                shot_hint = "closing framing of the beat"
            else:
                shot_hint = "medium framing, action in progress"

            composed = compose_shot_prompt(
                portrait=portrait,
                scene_visual=scene_visual,
                era_tail=era_tail,
                style_tail=resolved_style_tail,
                shot_hint=shot_hint,
            )

            shot_id = (
                f"{slugify(scene_id)}_s{shot_idx:02d}_start_{char_slug}"
            )
            tokens.append({
                "type": "environment",
                "description": composed,
                "role": "char_scene_composite",
                "shot_id": shot_id,
                "scene_id": scene_id,
                "focus_character": focus_character,
                "shot_index_in_scene": shot_idx,
                "kind": "start",
            })

    if include_final_end_frame and tokens:
        # One extra frame at the very end so the FLF chain closes.
        last_scene = scenes[-1] if scenes else {}
        last_scene_id = (last_scene.get("scene_id") or "scene_final").strip()
        last_visual = (last_scene.get("visual_prompt") or "").strip() or (
            last_scene.get("shot_description") or "final environment"
        )
        composed = compose_shot_prompt(
            portrait=portrait,
            scene_visual=last_visual,
            era_tail=era_tail,
            style_tail=resolved_style_tail,
            shot_hint="final closing framing, end of episode",
        )
        tokens.append({
            "type": "environment",
            "description": composed,
            "role": "char_scene_composite",
            "shot_id": f"{slugify(last_scene_id)}_final_end_{char_slug}",
            "scene_id": last_scene_id,
            "focus_character": focus_character,
            "shot_index_in_scene": -1,
            "kind": "end",
        })

    return {
        "tokens": tokens,
        "source": "OTR_VideoPlan",
        "focus_character": focus_character,
        "shots_per_scene": shots_per_scene,
        "scenes_covered": len(scenes),
        "total_prompts": len(tokens),
        "genre_flavor": genre_flavor,
        "era_tail": era_tail,
        "style_tail": resolved_style_tail,
    }


# ---------------------------------------------------------------------------
# ComfyUI node class
# ---------------------------------------------------------------------------


class OTRVideoPlan:
    """OTR_VideoPlan  --  read-only Director/script adapter."""

    @classmethod
    def INPUT_TYPES(cls):
        # Genre flavors matched to OTR_LLMScriptWriter dropdown
        genre_choices = list(_ERA_TAIL_BY_GENRE.keys()) + ["(none)"]
        return {
            "required": {
                "director_json": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": (
                            "Full Director JSON from OTR_LLMDirector "
                            "(contains visual_plan.characters + scenes)"
                        ),
                    },
                ),
                "focus_character": (
                    "STRING",
                    {
                        "default": "BABA",
                        "tooltip": (
                            "Name of the one character to anchor "
                            "across every shot in this render pass"
                        ),
                    },
                ),
                "shots_per_scene": (
                    "INT",
                    {"default": 3, "min": 1, "max": 40, "step": 1},
                ),
                "genre_flavor": (
                    genre_choices,
                    {"default": "hard_sci_fi"},
                ),
            },
            "optional": {
                "style_tail": (
                    "STRING",
                    {"default": _DEFAULT_STYLE_TAIL},
                ),
                "include_final_end_frame": (
                    "BOOLEAN",
                    {"default": True},
                ),
            },
        }

    RETURN_TYPES = ("STRING", "INT", "STRING")
    RETURN_NAMES = (
        "prompt_list_json",   # consumed by OTR_BatchFluxRender as env tokens
        "prompt_count",       # count int for UI display
        "debug_summary",      # human-readable summary
    )
    FUNCTION = "plan"
    CATEGORY = "OldTimeRadio/video"

    def plan(
        self,
        director_json: str,
        focus_character: str,
        shots_per_scene: int,
        genre_flavor: str,
        style_tail: str = "",
        include_final_end_frame: bool = True,
    ):
        if genre_flavor == "(none)":
            genre_flavor = ""

        plan = build_shot_plan(
            director_json=director_json,
            focus_character=focus_character,
            shots_per_scene=shots_per_scene,
            genre_flavor=genre_flavor,
            style_tail=style_tail,
            include_final_end_frame=include_final_end_frame,
        )

        summary_lines = [
            f"focus character: {plan['focus_character']}",
            f"genre flavor:    {plan['genre_flavor'] or '(none)'}",
            f"scenes covered:  {plan['scenes_covered']}",
            f"shots per scene: {plan['shots_per_scene']}",
            f"total prompts:   {plan['total_prompts']}",
            f"era tail:        {plan['era_tail']}",
            "",
            "first 3 prompts:",
        ]
        for tok in plan["tokens"][:3]:
            summary_lines.append(
                f"  [{tok['shot_id']}] "
                f"{tok['description'][:120]}"
                f"{'...' if len(tok['description']) > 120 else ''}"
            )
        summary = "\n".join(summary_lines)

        log.info(
            "OTR_VideoPlan READY: focus=%r shots=%d (%d scenes x %d + %d end)",
            focus_character,
            plan["total_prompts"],
            plan["scenes_covered"],
            shots_per_scene,
            1 if include_final_end_frame and plan["tokens"] else 0,
        )

        return (
            json.dumps(plan, indent=2),
            plan["total_prompts"],
            summary,
        )


# ---------------------------------------------------------------------------
# Module-level registration hook
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "OTR_VideoPlan": OTRVideoPlan,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "OTR_VideoPlan": " OTR Video Plan",
}

__all__ = [
    "OTRVideoPlan",
    "NODE_CLASS_MAPPINGS",
    "NODE_DISPLAY_NAME_MAPPINGS",
    "build_shot_plan",
    "compose_shot_prompt",
    "extract_scenes",
    "resolve_character_portrait",
    "resolve_era_tail",
    "slugify",
]
