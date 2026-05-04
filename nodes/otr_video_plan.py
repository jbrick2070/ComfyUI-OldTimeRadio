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
      + era_tail[style] + ", " + style_tail
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

# Era-appropriate tails keyed by OTR's style dropdown options.
# Fed into every composed prompt so FLUX dresses characters for the
# right era without requiring per-scene costume prose.
_ERA_TAIL_BY_STYLE: dict[str, str] = {
    "hard_sci_fi":       "near-future industrial sci-fi, clean lines, utilitarian",
    "space_opera":       "grand operatic space-opera scale, ornate costumes, vibrant",
    "dystopian":         "oppressive dystopian regime, drab uniforms, concrete",
    "time_travel":       "anachronistic mix, period-appropriate to the scene",
    "first_contact":     "scientific sublime, clean institutional aesthetic",
    "cosmic_horror":     "1920s Lovecraftian era, weathered, shadowed",
    "cyberpunk":         "1980s neon cyberpunk aesthetic, rain-slick streets, chrome",
    "post_apocalyptic":  "post-apocalyptic scavenger chic, worn gear, sun-bleached",
}

# Fallback era tail used when style is empty, unknown, or
# when the caller explicitly passes an override.
_DEFAULT_ERA_TAIL = "timeless cinematic aesthetic"

# Vocabulary
# ----------
# SCENE    - story setting (e.g. "comms bay"). One per scene_id.
# SHOT     - narrative unit belonging to one scene. Duration is TBD
#            (computed by a downstream calculator from Bark audio
#            durations once audio is rendered; not available at plan
#            time). Can be longer than one Wan call.
# SEGMENT  - Wan-call-sized sub-unit of a shot. Target <= 9 seconds
#            (under the 10s C4 hard cap with buffer). One Wan render
#            per segment, bounded by two FLUX frames.
# FRAME    - one PASS 3 FLUX render. N segments need N+1 frames
#            (adjacent segments share boundary frames, FLF style).
#
# Default behavior when the duration calculator is unavailable:
# ONE segment per shot (we assume the shot fits in one Wan call).
# Multi-segment splitting happens later when real durations are known.

SEGMENT_TARGET_DURATION_S = 9.0
SEGMENT_HARD_CAP_S = 10.0   # C4 constraint
SEGMENT_TARGET_FPS = 16
SEGMENT_MAX_FRAMES = 161    # 10.0s at 16fps, proven on 5080 Blackwell


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


def resolve_era_tail(style: str) -> str:
    """Return the era tail for a given style, or the default."""
    if not style:
        return _DEFAULT_ERA_TAIL
    key = style.strip().lower().replace(" ", "_").replace("-", "_")
    return _ERA_TAIL_BY_STYLE.get(key, _DEFAULT_ERA_TAIL)


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
    # BUG-LOCAL-080 (2026-05-03 EVENING): defensive coercion for
    # LLMDirector output shape drift. The expected shape is
    # ``visual_plan.characters[NAME] = {"portrait_prompt": "...", ...}``
    # but high-temperature LLMDirector occasionally emits one of:
    #   (a) characters as a top-level LIST instead of dict (rare)
    #   (b) characters[NAME] = [{"portrait_prompt": "..."}, {"voice":
    #       "..."}] -- list-of-dicts instead of single merged dict
    #   (c) characters[NAME] = "portrait prompt string" -- bare string
    # Without this coercion the line below raises:
    #   AttributeError: 'list' object has no attribute 'get'
    # which kills the entire FLUX/Plan phase (~8 min wasted before
    # crash). Coerce all three variants back to a single dict.
    if isinstance(characters, list):
        # (a) characters is a list -- try to find an entry with name match
        merged_chars = {}
        for item in characters:
            if isinstance(item, dict):
                key = (item.get("name") or item.get("character") or "").strip()
                if key:
                    merged_chars[key] = item
        characters = merged_chars
    elif not isinstance(characters, dict):
        characters = {}
    entry = characters.get(character_name)
    if isinstance(entry, list):
        # (b) list-of-dicts -- merge into single dict (later wins)
        merged_entry: dict = {}
        for item in entry:
            if isinstance(item, dict):
                merged_entry.update(item)
        entry = merged_entry
    elif isinstance(entry, str):
        # (c) bare string -- treat as the portrait_prompt directly
        entry = {"portrait_prompt": entry}
    elif not isinstance(entry, dict):
        entry = {}
    portrait = (entry.get("portrait_prompt") or "").strip()
    if portrait:
        return portrait

    # Fallback 2: synthesize from voice_assignments notes
    voice_assignments = director.get("voice_assignments") or {}
    if not isinstance(voice_assignments, dict):
        voice_assignments = {}
    va_entry = voice_assignments.get(character_name)
    if isinstance(va_entry, list):
        merged_va: dict = {}
        for item in va_entry:
            if isinstance(item, dict):
                merged_va.update(item)
        va_entry = merged_va
    elif not isinstance(va_entry, dict):
        va_entry = {}
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


def build_pass1_char_prompts(
    director_json: str,
    focus_character: str = "",
    *,
    style_tail: str = "",
) -> dict:
    """PASS 1 envelope: one token for each character's portrait.

    Multi-character mode (default): emits one token per character found
    in ``director.visual_plan.characters``. Use this to render the whole
    cast's portraits up front as FLUX images that can be reused across
    every shot they appear in.

    Single-character mode: set ``focus_character`` to a specific name to
    limit output to just that one portrait. Useful for thin-slice tests.

    Output envelope matches BatchFluxRender's env-token schema so it
    consumes the list as-is.
    """
    try:
        director = json.loads(director_json) if director_json else {}
    except json.JSONDecodeError:
        director = {}
    if not isinstance(director, dict):
        director = {}

    resolved_style_tail = (style_tail or _DEFAULT_STYLE_TAIL).strip()

    # Collect all character names that have portrait data
    visual_plan = director.get("visual_plan") or {}
    chars_dict = visual_plan.get("characters") or {}
    all_char_names = [
        n for n in chars_dict.keys() if isinstance(n, str) and n.strip()
    ]

    # If focus_character is set (non-empty, not the sentinel "(all)"),
    # limit to just that one. Otherwise emit tokens for every character.
    focus_is_limit = bool(focus_character) and focus_character != "(all)"
    if focus_is_limit:
        target_names = [focus_character]
    else:
        target_names = all_char_names if all_char_names else [focus_character]
        # If neither the cast nor a focus char is usable, fall back to
        # a single un-named portrait so the envelope is still valid.
        target_names = [n for n in target_names if n]
        if not target_names:
            target_names = [""]

    tokens: list[dict[str, Any]] = []
    for name in target_names:
        portrait = resolve_character_portrait(
            director, name, resolved_style_tail
        )
        char_slug = slugify(name) if name else "unnamed"
        tokens.append({
            "type": "environment",
            "description": portrait,
            "role": "char_portrait",
            "shot_id": f"char_{char_slug}_portrait",
            "character": name,
            "focus_character": focus_character,
            "kind": "pass1_char",
        })

    return {
        "tokens": tokens,
        "source": "OTR_VideoPlan.pass1",
        "focus_character": focus_character,
        "character_count": len(tokens),
        "characters": [t["character"] for t in tokens],
        "total_prompts": len(tokens),
        "style_tail": resolved_style_tail,
    }


def build_pass2_scene_prompts(
    director_json: str,
    *,
    style_tail: str = "",
) -> dict:
    """PASS 2 envelope: one token per scene environment.

    Each scene's visual_prompt is wrapped with the style tail. No
    character, no era, no action — just the empty scene.

    Output envelope matches BatchFluxRender's env-token schema.
    """
    try:
        director = json.loads(director_json) if director_json else {}
    except json.JSONDecodeError:
        director = {}
    if not isinstance(director, dict):
        director = {}

    resolved_style_tail = (style_tail or _DEFAULT_STYLE_TAIL).strip()
    scenes = extract_scenes(director)

    tokens: list[dict[str, Any]] = []
    for scene in scenes:
        scene_id = (scene.get("scene_id") or "").strip()
        if not scene_id:
            scene_id = f"scene_{len(tokens)+1:02d}"
        scene_visual = (scene.get("visual_prompt") or "").strip()
        if not scene_visual:
            scene_visual = (
                scene.get("shot_description") or f"{scene_id} environment"
            ).strip()

        # Compose without character, without era, without action —
        # just the scene visual + style tail.
        composed = compose_shot_prompt(
            portrait="",
            scene_visual=scene_visual,
            era_tail="",
            style_tail=resolved_style_tail,
            shot_hint="empty environment establishing shot",
        )
        tokens.append({
            "type": "environment",
            "description": composed,
            "role": "scene_env",
            "shot_id": f"{slugify(scene_id)}_env",
            "scene_id": scene_id,
            "kind": "pass2_scene",
        })

    return {
        "tokens": tokens,
        "source": "OTR_VideoPlan.pass2",
        "scenes_covered": len(scenes),
        "total_prompts": len(tokens),
        "style_tail": resolved_style_tail,
    }


def build_shot_plan(
    director_json: str,
    focus_character: str,
    *,
    shots_per_scene: int = 3,
    style: str = "",
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
          "style": "hard_sci_fi",
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
    era_tail = resolve_era_tail(style)

    # Multi-character compose: concatenate portraits for every character
    # in visual_plan.characters (the whole cast). Single-character mode
    # activates when ``focus_character`` is set to a specific non-empty
    # name (not "(all)"), emitting only that character's portrait.
    visual_plan = director.get("visual_plan") or {}
    chars_dict = visual_plan.get("characters") or {}

    # BUG-LOCAL-023 (Phase H, 2026-05-03): EXCLUDE non-visual roles
    # from portrait composition. ANNOUNCER (and any narrator-style
    # role that only voices over without appearing on screen) is
    # always represented by the static radio bookend image (per
    # BUG-LOCAL-129b: VideoComposite static-radio fill, BatchHumoRender
    # skip). Including their portrait_prompt in scene visuals wastes
    # FLUX context AND skews composition by forcing every shot to
    # accommodate an extra character. Belt-and-suspenders: the
    # LLMDirector prompt also instructs the LLM to skip these roles
    # in its visual_plan.characters output (story_orchestrator.py
    # VISUAL PLAN RULES), but a future LLM regression could re-include
    # them, so the filter here catches that case too.
    NON_VISUAL_ROLES = {"ANNOUNCER", "NARRATOR"}
    raw_char_names = [
        n for n in chars_dict.keys() if isinstance(n, str) and n.strip()
    ]
    all_char_names = [
        n for n in raw_char_names
        if n.strip().upper() not in NON_VISUAL_ROLES
    ]
    _skipped_non_visual = [
        n for n in raw_char_names if n.strip().upper() in NON_VISUAL_ROLES
    ]
    if _skipped_non_visual:
        log.info(
            "OTR_VideoPlan: skipped non-visual role(s) from portrait "
            "composition: %s (radio bookend handles their visuals)",
            ", ".join(_skipped_non_visual),
        )
    focus_is_limit = bool(focus_character) and focus_character != "(all)"
    if focus_is_limit:
        # Honor explicit focus_character even if it's a non-visual role
        # (lets a user manually request an "ANNOUNCER" portrait for
        # debugging or special workflows). The filter only applies to
        # the all-characters compose path.
        target_char_names = [focus_character]
    else:
        target_char_names = all_char_names if all_char_names else [focus_character]
        target_char_names = [n for n in target_char_names if n]

    # Collect portrait strings for the target characters and join.
    portrait_parts = [
        resolve_character_portrait(director, n, resolved_style_tail)
        for n in target_char_names
    ]
    portrait = ", ".join(p.strip().rstrip(",").strip() for p in portrait_parts if p) \
        if portrait_parts else resolve_character_portrait(
            director, focus_character, resolved_style_tail
        )

    scenes = extract_scenes(director)
    tokens: list[dict[str, Any]] = []

    char_slug = slugify(focus_character)

    # Flat list of all shots across all scenes, with their scene context.
    # Globally-numbered shot_ids ("shot_001", "shot_002", ...) for at-a-glance
    # cross-referencing. Parent scene pointer is carried separately.
    shots: list[dict[str, Any]] = []
    shot_global_idx = 0
    for scene in scenes:
        scene_id = (scene.get("scene_id") or "").strip()
        if not scene_id:
            scene_id = f"scene_{len(shots)+1:02d}"
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
            shot_global_idx += 1
            shots.append({
                "shot_id": f"shot_{shot_global_idx:03d}",
                "scene_id": scene_id,
                "scene_visual": scene_visual,
                "shot_index_in_scene": shot_idx,
            })

    # Render N+1 frames. Frame k bounds shot k (as its start) and shot k-1
    # (as its end). Edge cases: frame 0 is only a start; frame N is only an
    # end. If there are zero shots, emit zero frames (empty director case).
    #
    # For N shots, we emit len(shots)+1 frames when include_final_end_frame
    # is True, else len(shots) frames (dropping the final cap).
    frame_count = (
        len(shots) + (1 if include_final_end_frame and shots else 0)
    )
    for frame_idx in range(frame_count):
        # Which shot does this frame BOUND?
        # Frame k is the START of shot k (if k < N) and the END of shot k-1 (if k > 0).
        starts_shot = shots[frame_idx] if frame_idx < len(shots) else None
        ends_shot   = shots[frame_idx - 1] if frame_idx > 0 else None

        # Pick the scene context: prefer the START shot's scene; fall back to the
        # ending shot's scene for the final frame. Defensive skip if neither is
        # set (only possible with shots=[] which we already guard above).
        bound_shot = starts_shot or ends_shot
        if bound_shot is None:
            continue
        scene_id = bound_shot["scene_id"]
        scene_visual = bound_shot["scene_visual"]

        # Shot-progression hint — "opening", "mid", "closing":
        # first frame of the episode = opening establishing shot
        # last frame of the episode = final closing
        # bridge frames = "transition frame" (end of one, start of next)
        if starts_shot and not ends_shot:
            shot_hint = "establishing framing, opening of the scene"
        elif ends_shot and not starts_shot:
            shot_hint = "final closing framing, end of episode"
        elif starts_shot and ends_shot:
            # Bridge: same scene means transition within; different scene means cut.
            if starts_shot["scene_id"] == ends_shot["scene_id"]:
                shot_hint = "transition within scene, action continues"
            else:
                shot_hint = "scene transition, new setting establishing"
        else:
            shot_hint = ""

        composed = compose_shot_prompt(
            portrait=portrait,
            scene_visual=scene_visual,
            era_tail=era_tail,
            style_tail=resolved_style_tail,
            shot_hint=shot_hint,
        )

        frame_id = f"frame_{frame_idx:04d}"

        tokens.append({
            "type": "environment",
            "description": composed,
            "role": "char_scene_composite",
            "frame_id": frame_id,
            "shot_id": frame_id,  # back-compat: some callers still key on shot_id
            "scene_id": scene_id,
            "focus_character": focus_character,
            "frame_index": frame_idx,
            "starts_shot": starts_shot["shot_id"] if starts_shot else None,
            "ends_shot":   ends_shot["shot_id"]   if ends_shot   else None,
            "kind": (
                "start" if starts_shot and not ends_shot
                else "end" if ends_shot and not starts_shot
                else "bridge"
            ),
        })

    # Emit a shots[] index with a segments[] sub-structure per shot.
    #
    # A SHOT is a narrative unit (6-10s of audio-aligned story).
    # A CLIP is a single Wan call, bounded by 2 FLUX frames.
    #
    # When shot_duration_s <= wan_single_call_budget_s (~10s at 16fps
    # on 5080): one clip per shot, bounded by 2 frames.
    #
    # When shot_duration_s > budget: the "magical duration calculator"
    # (TBA - depends on Bark audio timing, not available at plan time)
    # splits the shot into M segments, needing M+1 frames. That expansion
    # is DEFERRED - for now we emit 1 clip per shot, and the structure
    # is ready to receive more segments later.
    shots_index: list[dict[str, Any]] = []
    for i, shot in enumerate(shots):
        start_frame_id = f"frame_{i:04d}"
        end_frame_idx = i + 1
        if end_frame_idx < len(shots) or include_final_end_frame:
            end_frame_id = f"frame_{end_frame_idx:04d}"
        else:
            end_frame_id = None  # no end frame available

        # Default: ONE clip per shot. Duration calculator (TBA) expands
        # to N segments when shot_duration_s > SEGMENT_HARD_CAP_S.
        # Clip IDs are 1-indexed: shot_003_c1, shot_003_c2, etc.
        segments = [{
            "clip_id": f"{shot['shot_id']}_c1",
            "start_frame_id": start_frame_id,
            "end_frame_id":   end_frame_id,
        }]

        shots_index.append({
            "shot_id": shot["shot_id"],
            "scene_id": shot["scene_id"],
            "shot_duration_s": None,  # TBA: filled by duration calculator
            "segments": segments,
            # convenience top-level pointers (valid for 1-clip-per-shot case)
            "start_frame_id": start_frame_id,
            "end_frame_id":   end_frame_id,
        })

    total_segments = sum(len(s["segments"]) for s in shots_index)

    return {
        "tokens": tokens,
        "shots": shots_index,
        "source": "OTR_VideoPlan",
        "focus_character": focus_character,
        "shots_per_scene": shots_per_scene,
        "scenes_covered": len(scenes),
        "total_shots": len(shots),
        "total_segments": total_segments,
        "total_prompts": len(tokens),
        "style": style,
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
        # Style values matched to OTR_LLMScriptWriter dropdown
        style_choices = list(_ERA_TAIL_BY_STYLE.keys()) + ["(none)"]
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
                        "default": "(all)",
                        "tooltip": (
                            "'(all)' or empty = multi-character mode: "
                            "render every character in visual_plan.characters. "
                            "Set to a specific name (e.g. 'LEMMY') to limit to "
                            "one character for a thin-slice test."
                        ),
                    },
                ),
                "shots_per_scene": (
                    "INT",
                    {"default": 3, "min": 1, "max": 40, "step": 1},
                ),
                "style": (
                    style_choices,
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
                "audio_gate": (
                    "STRING",
                    {
                        "default": "",
                        "forceInput": True,
                        "tooltip": (
                            "Optional sequencing gate. Wire a downstream audio "
                            "node's output here (e.g. OTR_VisualRenderer."
                            "final_mp4_path) to force ComfyUI's topsort to run "
                            "audio + POC video + ffmpeg chain to completion "
                            "BEFORE starting the FLUX video branch. "
                            "The value itself is ignored — presence of the link "
                            "is what creates the dependency. Prevents VRAM "
                            "contention when Bark / MusicGen / FLUX compete."
                        ),
                    },
                ),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = (
        "pass1_char_prompts_json",    # PASS 1: character portraits (one per cast member)
        "pass2_scene_prompts_json",   # PASS 2: empty scene envs (one per scene)
        "pass3_compose_prompts_json", # PASS 3: per-shot char-in-scene composites
        "pass3_prompt_count",         # count of PASS 3 shots (feeds Wan N+1 chain)
        "debug_summary",              # human-readable summary of all 3 passes
    )
    FUNCTION = "plan"
    CATEGORY = "OldTimeRadio/video"

    def plan(
        self,
        director_json: str,
        focus_character: str,
        shots_per_scene: int,
        style: str,
        style_tail: str = "",
        include_final_end_frame: bool = True,
        audio_gate: str = "",
    ):
        # audio_gate is deliberately ignored — its only purpose is to create
        # a topsort dependency so this node fires AFTER the upstream audio
        # pipeline completes. Never read or log the value.
        _ = audio_gate

        if style == "(none)":
            style = ""

        # Treat the "(all)" sentinel as unset so downstream helpers
        # switch into multi-character mode.
        resolved_focus = focus_character
        if focus_character == "(all)":
            resolved_focus = ""

        # PASS 1: char portraits (one per character, or just the
        # focus character if focus_character is a specific name)
        pass1 = build_pass1_char_prompts(
            director_json=director_json,
            focus_character=resolved_focus,
            style_tail=style_tail,
        )

        # PASS 2: scene envs (one per scene)
        pass2 = build_pass2_scene_prompts(
            director_json=director_json,
            style_tail=style_tail,
        )

        # PASS 3: composite shot frames (S scenes x K shots + 1 end)
        pass3 = build_shot_plan(
            director_json=director_json,
            focus_character=resolved_focus,
            shots_per_scene=shots_per_scene,
            style=style,
            style_tail=style_tail,
            include_final_end_frame=include_final_end_frame,
        )

        # Describe cast (for multi-char mode) or focus
        pass1_chars = pass1.get("characters", [])
        if resolved_focus:
            cast_line = f"focus character: {resolved_focus}"
        else:
            cast_line = (
                f"cast ({pass1['character_count']}): "
                f"{', '.join(pass1_chars) if pass1_chars else '(none)'}"
            )

        summary_lines = [
            cast_line,
            f"style:           {pass3['style'] or '(none)'}",
            f"scenes covered:  {pass3['scenes_covered']}",
            "",
            f"PASS 1 (char portraits):   {pass1['total_prompts']} prompt(s)"
            f"{' — all cast' if not resolved_focus else ' — focus only'}",
            f"PASS 2 (scene envs):       {pass2['total_prompts']} prompt(s)",
            f"PASS 3 (composite shots):  {pass3['total_prompts']} prompt(s) "
            f"({pass3['shots_per_scene']} per scene + "
            f"{1 if include_final_end_frame and pass3['tokens'] else 0} end)",
            "",
            "first PASS 3 composite:",
        ]
        for tok in pass3["tokens"][:1]:
            summary_lines.append(
                f"  [{tok['shot_id']}] "
                f"{tok['description'][:160]}"
                f"{'...' if len(tok['description']) > 160 else ''}"
            )
        summary = "\n".join(summary_lines)

        log.info(
            "OTR_VideoPlan READY: PASS1=%d PASS2=%d PASS3=%d focus=%r",
            pass1["total_prompts"],
            pass2["total_prompts"],
            pass3["total_prompts"],
            focus_character,
        )

        # ------------------------------------------------------------------
        # Production Ledger (L2 prep). VideoPlan emits the canonical PASS3
        # shot list. Write it to the ledger so:
        #   - FULL run: shots[] gets the detailed visual_prompt + per-shot
        #     scene_id (richer than the Director-stage one-shot-per-scene
        #     placeholder).
        #   - TEST run: VideoPlan is typically the FIRST OTR node. If no
        #     ledger exists yet, get_ledger() auto-creates a placeholder
        #     so subsequent FLUX renders can attach png_paths.
        # ------------------------------------------------------------------
        try:
            from .production_ledger import get_ledger
            led = get_ledger()
            shot_rows = []
            for idx, tok in enumerate(pass3.get("tokens") or []):
                if not isinstance(tok, dict):
                    continue
                shot_rows.append({
                    "shot_id":       str(tok.get("shot_id") or f"sh{idx+1:02d}"),
                    "scene_id":      str(tok.get("scene_id") or "") or None,
                    "description":   str(tok.get("description") or "")[:200],
                    "visual_prompt": str(tok.get("visual_prompt")
                                          or tok.get("compose_prompt")
                                          or tok.get("description") or ""),
                })
            if shot_rows:
                led.set_shots(shot_rows)
                led.save()
        except Exception as _e:  # noqa: BLE001
            log.warning("[Ledger] OTR_VideoPlan snapshot failed: %s", _e)

        return (
            json.dumps(pass1, indent=2),
            json.dumps(pass2, indent=2),
            json.dumps(pass3, indent=2),
            pass3["total_prompts"],
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
    "build_pass1_char_prompts",
    "build_pass2_scene_prompts",
    "build_shot_plan",
    "compose_shot_prompt",
    "extract_scenes",
    "resolve_character_portrait",
    "resolve_era_tail",
    "slugify",
]
