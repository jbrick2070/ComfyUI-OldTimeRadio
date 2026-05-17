"""
test_otr_video_plan.py  --  OTR_VideoPlan adapter unit tests
=============================================================

Validates ``nodes.otr_video_plan`` without torch, diffusers, or
ComfyUI.  Covers the pure helpers + node surface area + the
fallback chain for missing legacy Director fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ledger_wrap(director_dict):
    """Voice-path-cleanbreak Sprint 2 helper: wrap a director-shape
    dict (the legacy OTR_LLMDirector output shape) inside an L3 ledger
    JSON so OTRVideoPlan.plan() can read it via the new script_json
    socket. Mirrors the writer's K.5 stamp pattern: meta.visual_plan +
    meta.voice_assignments + meta.style.
    """
    return json.dumps({
        "schema_version": "l3-2026-05-14",
        "cast":  [],
        "lines": [],
        "meta": {
            "visual_plan":       director_dict.get("visual_plan") or {},
            "voice_assignments": director_dict.get("voice_assignments") or {},
            "style":             director_dict.get("style") or "",
        },
    })


# ------------------------------------------------------------------
# Import smoke
# ------------------------------------------------------------------


def test_module_imports_cleanly():
    import importlib
    mod = importlib.import_module("nodes.otr_video_plan")
    for name in (
        "OTRVideoPlan",
        "build_shot_plan",
        "compose_shot_prompt",
        "extract_scenes",
        "resolve_character_portrait",
        "resolve_era_tail",
        "slugify",
        "NODE_CLASS_MAPPINGS",
        "NODE_DISPLAY_NAME_MAPPINGS",
    ):
        assert hasattr(mod, name), f"missing attribute: {name}"


def test_node_class_mappings_present():
    from nodes.otr_video_plan import (
        NODE_CLASS_MAPPINGS,
        OTRVideoPlan,
    )
    assert "OTR_VideoPlan" in NODE_CLASS_MAPPINGS
    assert NODE_CLASS_MAPPINGS["OTR_VideoPlan"] is OTRVideoPlan


# ------------------------------------------------------------------
# slugify
# ------------------------------------------------------------------


def test_slugify_simple():
    from nodes.otr_video_plan import slugify
    assert slugify("BABA") == "baba"
    assert slugify("Kenji Cross") == "kenji_cross"
    assert slugify("Agent 47 / B") == "agent_47_b"


def test_slugify_edge_cases():
    from nodes.otr_video_plan import slugify
    assert slugify("") == "unnamed"
    assert slugify("   ") == "unnamed"
    assert slugify("!!!???") == "unnamed"
    assert slugify("A") == "a"


def test_slugify_length_limit():
    from nodes.otr_video_plan import slugify
    long_name = "A" * 200
    result = slugify(long_name, max_len=40)
    assert len(result) == 40


# ------------------------------------------------------------------
# resolve_era_tail
# ------------------------------------------------------------------


def test_resolve_era_tail_known_genres():
    from nodes.otr_video_plan import resolve_era_tail
    # Each assertion picks one distinctive keyword from the era-tail
    # descriptor so the test catches both the lookup AND the
    # descriptor wording in one shot. New 10-preset set landed
    # 2026-05-10; era-tail strings are visual aesthetics only --
    # no era literals.
    assert "cathode-tube" in resolve_era_tail("haunted_broadcast_signal").lower()
    assert "vessel" in resolve_era_tail("deep_space_distress_call").lower()
    assert "smoke-filtered" in resolve_era_tail("noir_interrogation").lower()
    assert "instrument-panel" in resolve_era_tail("mission_control_procedural").lower()


def test_resolve_era_tail_empty_and_unknown():
    from nodes.otr_video_plan import resolve_era_tail
    default = resolve_era_tail("")
    assert default
    assert resolve_era_tail("gothic_romance") == default
    assert resolve_era_tail("nonexistent_genre") == default


def test_resolve_era_tail_normalizes_case_and_punctuation():
    from nodes.otr_video_plan import resolve_era_tail
    # All five spellings of the same preset must hit the same era tail.
    r1 = resolve_era_tail("mission control procedural")
    r2 = resolve_era_tail("MISSION CONTROL PROCEDURAL")
    r3 = resolve_era_tail("Mission Control Procedural")
    r4 = resolve_era_tail("mission_control_procedural")
    r5 = resolve_era_tail("mission-control-procedural")
    assert r1 == r2 == r3 == r4 == r5


# ------------------------------------------------------------------
# resolve_character_portrait
# ------------------------------------------------------------------


def test_portrait_canonical_path():
    from nodes.otr_video_plan import resolve_character_portrait
    derived = {
        "characters": {
            "BABA": {
                "portrait_prompt": "Cinematic portrait of elderly spacer, silver braids"
            }
        }
    }
    result = resolve_character_portrait(derived, "BABA", "style_tail")
    assert "silver braids" in result
    assert "style_tail" not in result  # canonical path doesn't append tail


def test_portrait_tier_2_notes_fallback_is_retired():
    """Voice-path-cleanbreak Sprint 6.2 (2026-05-12): the Tier-2
    fallback that synthesized portraits from
    voice_assignments[NAME].notes was retired. The notes field no
    longer exists in the post-Sprint-6.2 voice_assignments shape
    (only voice_preset). resolve_character_portrait now falls
    straight from Tier 1 (portrait_prompt) to the generic template.
    """
    from nodes.otr_video_plan import resolve_character_portrait
    derived = {
        "characters": {"BABA": {}},
        "voice_assignments": {
            "BABA": {"voice_preset": "v2/en_speaker_8",
                     "notes": "Female, 60s, weary, low"}
        },
    }
    result = resolve_character_portrait(derived, "BABA", "cinematic tail")
    # Tier 2 retired: the notes string MUST NOT appear in output.
    assert "Female, 60s, weary, low" not in result
    # Falls to the generic template (Tier-3-now-Tier-2).
    assert "BABA" in result
    assert "cinematic tail" in result


def test_portrait_fallback_generic():
    """Empty projection -> generic template."""
    from nodes.otr_video_plan import resolve_character_portrait
    derived = {}
    result = resolve_character_portrait(derived, "BOOEY", "the tail")
    assert "BOOEY" in result
    assert "the tail" in result


def test_portrait_empty_character_name():
    from nodes.otr_video_plan import resolve_character_portrait
    result = resolve_character_portrait({}, "", "style")
    assert result  # non-empty
    assert "style" in result


def test_portrait_empty_portrait_prompt_falls_to_generic():
    """If derived["characters"][NAME].portrait_prompt is '', fall through
    to the generic template (Sprint 6.2 retired the notes-based Tier-2)."""
    from nodes.otr_video_plan import resolve_character_portrait
    derived = {
        "characters": {"BABA": {"portrait_prompt": ""}},
        "voice_assignments": {"BABA": {"notes": "old pilot"}},
    }
    result = resolve_character_portrait(derived, "BABA", "x")
    # Falls past empty Tier 1, past the retired Tier 2 (notes ignored
    # entirely now), to the generic Tier-3 template.
    assert "BABA" in result
    assert "x" in result
    assert "old pilot" not in result


# ------------------------------------------------------------------
# extract_scenes
# ------------------------------------------------------------------


def test_extract_scenes_basic():
    from nodes.otr_video_plan import extract_scenes
    derived = {
        "scenes": [
            {"scene_id": "scene_1", "visual_prompt": "bridge"},
            {"scene_id": "scene_2", "visual_prompt": "corridor"},
        ]
    }
    scenes = extract_scenes(derived)
    assert len(scenes) == 2
    assert scenes[0]["scene_id"] == "scene_1"


def test_extract_scenes_empty():
    from nodes.otr_video_plan import extract_scenes
    assert extract_scenes({}) == []
    assert extract_scenes({"scenes": []}) == []


def test_extract_scenes_malformed_skips_non_dicts():
    from nodes.otr_video_plan import extract_scenes
    derived = {
        "scenes": [
            {"scene_id": "good", "visual_prompt": "ok"},
            "bogus_string_entry",
            42,
            {"scene_id": "also_good"},
        ]
    }
    scenes = extract_scenes(derived)
    assert len(scenes) == 2


def test_extract_scenes_malformed_root_returns_empty():
    from nodes.otr_video_plan import extract_scenes
    derived = {"scenes": "not a list"}
    assert extract_scenes(derived) == []


# ------------------------------------------------------------------
# compose_shot_prompt
# ------------------------------------------------------------------


def test_compose_shot_prompt_order_and_joining():
    from nodes.otr_video_plan import compose_shot_prompt
    result = compose_shot_prompt(
        portrait="elderly spacer",
        scene_visual="dim bridge",
        era_tail="sci-fi",
        style_tail="cinematic",
        shot_hint="establishing",
    )
    # Subject first, then scene, shot_hint, era_tail, style_tail
    assert result.startswith("elderly spacer")
    assert result.endswith("cinematic")
    # Comma-joined
    parts = [p.strip() for p in result.split(",")]
    assert "elderly spacer" in parts
    assert "dim bridge" in parts
    assert "establishing" in parts
    assert "sci-fi" in parts
    assert "cinematic" in parts


def test_compose_shot_prompt_skips_empty_pieces():
    from nodes.otr_video_plan import compose_shot_prompt
    result = compose_shot_prompt(
        portrait="char",
        scene_visual="",
        era_tail="era",
        style_tail="",
        shot_hint="",
    )
    parts = [p.strip() for p in result.split(",")]
    assert "char" in parts
    assert "era" in parts
    # No empty sections
    assert all(p for p in parts)


def test_compose_shot_prompt_cleans_trailing_commas():
    from nodes.otr_video_plan import compose_shot_prompt
    result = compose_shot_prompt(
        portrait="char, ",       # trailing comma
        scene_visual=" scene,",  # trailing comma + leading space
        era_tail="",
        style_tail="style",
        shot_hint="",
    )
    # Should NOT produce ",," or "char,  ,"
    assert ",," not in result
    assert result.count(",") == 2  # char, scene, style


# ------------------------------------------------------------------
# build_shot_plan -- the core integration
# ------------------------------------------------------------------


def _sample_director() -> dict:
    return {
        "episode_title": "The Kepler Signal",
        "voice_assignments": {
            "BABA": {
                "voice_preset": "v2/en_speaker_8",
                "notes": "Female, 60s, weary, low",
            }
        },
        "sfx_plan": [],
        "music_plan": [],
        "pacing": {"beat_pause_ms": 100},
        "visual_plan": {
            "characters": {
                "BABA": {
                    "portrait_prompt": (
                        "Cinematic portrait of elderly spacer, silver braids, "
                        "grey engineer jumpsuit, weathered face, blue console light"
                    )
                }
            },
            "scenes": [
                {
                    "scene_id": "scene_1",
                    "shot_description": "BABA at the console",
                    "visual_prompt": "dim comms bay, red alert pulse, blue console glow",
                },
                {
                    "scene_id": "scene_2",
                    "shot_description": "BABA in corridor",
                    "visual_prompt": "narrow corridor, emergency lights",
                },
            ],
        },
    }


def test_build_shot_plan_basic_counts():
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json,
        focus_character="BABA",
        shots_per_scene=3,
        style="mission_control_procedural",
    )
    # 2 scenes * 3 shots + 1 final end = 7
    assert plan["total_prompts"] == 7
    assert plan["scenes_covered"] == 2
    assert plan["shots_per_scene"] == 3
    assert plan["focus_character"] == "BABA"


def test_build_shot_plan_tokens_are_env_shaped():
    """BatchFluxRender reads tokens where type='environment'."""
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json,
        focus_character="BABA",
        shots_per_scene=2,
        style="mission_control_procedural",
    )
    for tok in plan["tokens"]:
        assert tok["type"] == "environment"
        assert tok["description"]
        # Was `tok["shot_id"]` pre-S27. The shot_id alias on these
        # envelope tokens was deleted as a back-compat surface in
        # cleanbreak-tail Phase 2 / QA-3; frame_id was always the
        # canonical key and the alias copied frame_id verbatim.
        assert tok["frame_id"]
        assert tok["focus_character"] == "BABA"


def test_build_shot_plan_compose_includes_portrait_and_scene():
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json,
        focus_character="BABA",
        shots_per_scene=1,
        style="mission_control_procedural",
    )
    first = plan["tokens"][0]
    desc = first["description"]
    # Should contain portrait keywords
    assert "silver braids" in desc
    # Should contain scene_1's visual_prompt text
    assert "dim comms bay" in desc or "red alert pulse" in desc
    # Should contain era tail (mission_control_procedural keyword)
    assert "instrument-panel" in desc.lower() or "console" in desc.lower()


def test_build_shot_plan_empty_portrait_falls_to_generic():
    """Voice-path-cleanbreak Sprint 6.2 (2026-05-12): the legacy
    Tier-2 fallback (synthesize from voice_assignments[NAME].notes)
    is retired. With portrait_prompt empty, build_shot_plan falls
    straight to the generic Tier-3 template ("Cinematic portrait of
    BABA, ...") -- the notes string MUST NOT appear."""
    from nodes.otr_video_plan import build_shot_plan
    director = _sample_director()
    director["visual_plan"]["characters"]["BABA"]["portrait_prompt"] = ""
    script_json = _ledger_wrap(director)
    plan = build_shot_plan(
        script_json,
        focus_character="BABA",
        shots_per_scene=1,
    )
    first = plan["tokens"][0]
    assert "Female, 60s, weary, low" not in first["description"]
    assert "BABA" in first["description"]


def test_build_shot_plan_final_end_frame_toggle():
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())

    with_end = build_shot_plan(
        script_json, "BABA", shots_per_scene=3,
        include_final_end_frame=True,
    )
    without = build_shot_plan(
        script_json, "BABA", shots_per_scene=3,
        include_final_end_frame=False,
    )
    assert with_end["total_prompts"] == without["total_prompts"] + 1
    # First frame is a pure "start" (no ends_shot)
    assert with_end["tokens"][0]["kind"] == "start"
    # Last frame with include_final_end_frame is a pure "end"
    assert with_end["tokens"][-1]["kind"] == "end"
    # Middle frames are "bridge" (start of next shot AND end of previous)
    # Without the final end frame, the last frame is still a bridge/start
    assert without["tokens"][0]["kind"] == "start"
    # Without end toggle, no "end"-only frame exists
    assert not any(t["kind"] == "end" for t in without["tokens"])


def test_build_shot_plan_empty_director():
    """Empty (legacy) Director input -> empty token list, doesn't crash."""
    from nodes.otr_video_plan import build_shot_plan
    plan = build_shot_plan("", "BABA", shots_per_scene=3)
    assert plan["total_prompts"] == 0
    assert plan["tokens"] == []


def test_build_shot_plan_malformed_json_graceful():
    from nodes.otr_video_plan import build_shot_plan
    plan = build_shot_plan("{not valid json", "BABA", shots_per_scene=3)
    assert plan["total_prompts"] == 0


def test_build_shot_plan_invalid_shots_per_scene_raises():
    from nodes.otr_video_plan import build_shot_plan
    with pytest.raises(ValueError):
        build_shot_plan("{}", "BABA", shots_per_scene=0)
    with pytest.raises(ValueError):
        build_shot_plan("{}", "BABA", shots_per_scene=-1)


def test_build_shot_plan_unknown_genre_uses_default_era():
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json, "BABA", shots_per_scene=1,
        style="gothic_romance",  # not in dict
    )
    assert "timeless" in plan["era_tail"].lower()


def test_build_shot_plan_frame_ids_unique():
    """Was test_build_shot_plan_shot_ids_unique pre-S27. Renamed in
    lockstep with the QA-3 envelope-alias deletion -- the test
    contract is unchanged (per-token unique identifiers across the
    episode), only the key name moved to its canonical form."""
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json, "BABA", shots_per_scene=3,
    )
    frame_ids = [t["frame_id"] for t in plan["tokens"]]
    assert len(frame_ids) == len(set(frame_ids)), \
        "frame_ids must be unique across the episode"


def test_build_shot_plan_frame_ids_are_global_4digit():
    """Schema: frame IDs are global 4-digit, no character suffix."""
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json, "KENJI CROSS", shots_per_scene=1,
    )
    for tok in plan["tokens"]:
        # frame_0000, frame_0001, ...
        assert tok["frame_id"].startswith("frame_")
        assert len(tok["frame_id"]) == len("frame_0000")


def test_build_shot_plan_shot_ids_are_global_3digit():
    """Schema: shot IDs are global 'shot_NNN' across the whole episode."""
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json, "BABA", shots_per_scene=3,
    )
    shot_ids = [s["shot_id"] for s in plan["shots"]]
    # 2 scenes x 3 shots = 6 shots
    assert shot_ids == [
        "shot_001", "shot_002", "shot_003",
        "shot_004", "shot_005", "shot_006",
    ]


def test_build_shot_plan_clip_ids_indexed_from_one():
    """Schema: clip IDs are 'shot_NNN_c1', 'shot_NNN_c2', ..."""
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json, "BABA", shots_per_scene=1,
    )
    for shot in plan["shots"]:
        assert len(shot["segments"]) >= 1
        # Default: 1 clip per shot
        assert shot["segments"][0]["clip_id"] == f"{shot['shot_id']}_c1"


def test_build_shot_plan_shared_boundary_frames():
    """Schema: adjacent shots share boundary frames (FLF chain)."""
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_sample_director())
    plan = build_shot_plan(
        script_json, "BABA", shots_per_scene=2,
    )
    shots = plan["shots"]
    # shot_001's end_frame == shot_002's start_frame (shared)
    for i in range(len(shots) - 1):
        assert shots[i]["end_frame_id"] == shots[i + 1]["start_frame_id"], \
            f"shots {shots[i]['shot_id']} and {shots[i+1]['shot_id']} don't share boundary frame"


def test_build_shot_plan_scene_without_visual_prompt_uses_shot_description():
    from nodes.otr_video_plan import build_shot_plan
    director = _sample_director()
    director["visual_plan"]["scenes"][0]["visual_prompt"] = ""
    director["visual_plan"]["scenes"][0]["shot_description"] = "custom_desc_xyz"
    script_json = _ledger_wrap(director)
    plan = build_shot_plan(script_json, "BABA", shots_per_scene=1)
    # First scene's shot should include the shot_description fallback
    scene_1_tokens = [
        t for t in plan["tokens"] if t["scene_id"] == "scene_1"
    ]
    assert any("custom_desc_xyz" in t["description"] for t in scene_1_tokens)


# ------------------------------------------------------------------
# ComfyUI node class surface area
# ------------------------------------------------------------------


def test_input_types_schema():
    """Voice-path-cleanbreak Sprint 2 (2026-05-12): the legacy
    director_json socket was renamed to script_json. The node now
    reads the L3 ledger from FreezeCascade and derives the legacy
    director-shape internally before calling the build_* helpers
    (whose director_json parameter NAME is preserved for back-compat
    with their own test surface)."""
    from nodes.otr_video_plan import OTRVideoPlan
    schema = OTRVideoPlan.INPUT_TYPES()
    assert "required" in schema
    assert "script_json" in schema["required"]
    assert "director_json" not in schema["required"]  # legacy name
    assert "focus_character" in schema["required"]
    assert "shots_per_scene" in schema["required"]
    assert "style" in schema["required"]
    assert schema["required"]["script_json"][0] == "STRING"
    assert schema["required"]["shots_per_scene"][0] == "INT"


def test_script_json_multiline():
    from nodes.otr_video_plan import OTRVideoPlan
    schema = OTRVideoPlan.INPUT_TYPES()
    assert schema["required"]["script_json"][1].get("multiline") is True


def test_return_types():
    from nodes.otr_video_plan import OTRVideoPlan
    # 5-tuple: pass1_chars, pass2_scenes, pass3_composites, count, summary
    assert OTRVideoPlan.RETURN_TYPES == (
        "STRING", "STRING", "STRING", "INT", "STRING"
    )
    assert OTRVideoPlan.RETURN_NAMES[0] == "pass1_char_prompts_json"
    assert OTRVideoPlan.RETURN_NAMES[1] == "pass2_scene_prompts_json"
    assert OTRVideoPlan.RETURN_NAMES[2] == "pass3_compose_prompts_json"
    assert OTRVideoPlan.RETURN_NAMES[3] == "pass3_prompt_count"
    assert OTRVideoPlan.RETURN_NAMES[4] == "debug_summary"


def test_function_and_category():
    from nodes.otr_video_plan import OTRVideoPlan
    assert OTRVideoPlan.FUNCTION == "plan"
    assert OTRVideoPlan.CATEGORY.startswith("OldTimeRadio")


def test_plan_method_end_to_end():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    script_json = _ledger_wrap(_sample_director())
    pass1_json, pass2_json, pass3_json, pass3_count, summary = node.plan(
        script_json=script_json,
        focus_character="BABA",
        shots_per_scene=3,
        style="mission_control_procedural",
    )
    pass3 = json.loads(pass3_json)
    assert pass3["total_prompts"] == pass3_count
    assert pass3["focus_character"] == "BABA"
    assert "tokens" in pass3
    assert pass3["tokens"][0]["type"] == "environment"
    # Summary covers all 3 passes
    assert "pass 1" in summary.lower()
    assert "pass 2" in summary.lower()
    assert "pass 3" in summary.lower()

    # PASS 1 = one character portrait
    pass1 = json.loads(pass1_json)
    assert pass1["total_prompts"] == 1
    assert pass1["tokens"][0]["role"] == "char_portrait"

    # PASS 2 = one token per scene (2 scenes in sample)
    pass2 = json.loads(pass2_json)
    assert pass2["total_prompts"] == 2
    assert pass2["tokens"][0]["role"] == "scene_env"


def test_plan_method_handles_none_genre():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    script_json = _ledger_wrap(_sample_director())
    _p1, _p2, pass3_json, _count, _summary = node.plan(
        script_json=script_json,
        focus_character="BABA",
        shots_per_scene=1,
        style="(none)",
    )
    payload = json.loads(pass3_json)
    assert payload["style"] == ""


def test_plan_method_empty_scenes_still_returns_envelope():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    script_json = _ledger_wrap({
        "voice_assignments": {"BABA": {"voice_preset": "x", "notes": "old"}},
        "visual_plan": {"characters": {"BABA": {"portrait_prompt": "p"}}, "scenes": []},
    })
    _p1, _p2, pass3_json, count, _summary = node.plan(
        script_json=script_json,
        focus_character="BABA",
        shots_per_scene=3,
        style="mission_control_procedural",
    )
    assert count == 0
    payload = json.loads(pass3_json)
    assert payload["tokens"] == []
    assert payload["scenes_covered"] == 0


# ------------------------------------------------------------------
# Multi-character mode
# ------------------------------------------------------------------


def _multi_char_director() -> dict:
    return {
        "episode_title": "The Saturn Broadcast",
        "voice_assignments": {
            "LEMMY": {"voice_preset": "v2/en_speaker_8",
                      "notes": "Male, 50s, gravelly"},
            "KENJI CROSS": {"voice_preset": "v2/en_speaker_3",
                            "notes": "Male, 30s, clipped"},
        },
        "sfx_plan": [],
        "music_plan": [],
        "pacing": {"beat_pause_ms": 100},
        "visual_plan": {
            "characters": {
                "LEMMY": {"portrait_prompt": "weathered male spacer, 50s, muttonchops, flight jacket"},
                "KENJI CROSS": {"portrait_prompt": "disciplined officer, 30s, black hair, charcoal uniform"},
            },
            "scenes": [
                {"scene_id": "scene_1", "visual_prompt": "dim comms bay"},
                {"scene_id": "scene_2", "visual_prompt": "narrow corridor"},
            ],
        },
    }


def test_pass1_multi_char_emits_one_token_per_character():
    from nodes.otr_video_plan import build_pass1_char_prompts
    script_json = _ledger_wrap(_multi_char_director())
    # Empty focus_character => multi-char mode
    pass1 = build_pass1_char_prompts(script_json, "", style_tail="style")
    assert pass1["character_count"] == 2
    assert set(pass1["characters"]) == {"LEMMY", "KENJI CROSS"}
    # Each token has the right portrait
    desc_by_name = {t["character"]: t["description"] for t in pass1["tokens"]}
    assert "muttonchops" in desc_by_name["LEMMY"]
    assert "charcoal uniform" in desc_by_name["KENJI CROSS"]


def test_pass1_all_sentinel_equals_multi_char():
    from nodes.otr_video_plan import build_pass1_char_prompts
    script_json = _ledger_wrap(_multi_char_director())
    # "(all)" sentinel => multi-char mode
    pass1 = build_pass1_char_prompts(script_json, "(all)", style_tail="style")
    assert pass1["character_count"] == 2


def test_pass1_single_char_mode_respects_focus():
    """When focus_character is a specific name, only that one portrait."""
    from nodes.otr_video_plan import build_pass1_char_prompts
    script_json = _ledger_wrap(_multi_char_director())
    pass1 = build_pass1_char_prompts(script_json, "LEMMY", style_tail="style")
    assert pass1["character_count"] == 1
    assert pass1["characters"] == ["LEMMY"]


def test_pass3_multi_char_includes_all_portraits():
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_multi_char_director())
    # Empty focus_character => include all chars
    plan = build_shot_plan(
        script_json, focus_character="",
        shots_per_scene=1, style="mission_control_procedural",
    )
    first_desc = plan["tokens"][0]["description"]
    # Both character portraits should appear
    assert "muttonchops" in first_desc
    assert "charcoal uniform" in first_desc


def test_pass3_single_char_mode_only_focus_character():
    from nodes.otr_video_plan import build_shot_plan
    script_json = _ledger_wrap(_multi_char_director())
    plan = build_shot_plan(
        script_json, focus_character="LEMMY",
        shots_per_scene=1, style="mission_control_procedural",
    )
    first_desc = plan["tokens"][0]["description"]
    # Only LEMMY's portrait in compose, not KENJI's
    assert "muttonchops" in first_desc
    assert "charcoal uniform" not in first_desc


def test_plan_method_multi_char_default():
    """Default "(all)" widget value => multi-char mode, both PASS 1 chars."""
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    script_json = _ledger_wrap(_multi_char_director())
    pass1_json, _p2, _p3, _count, summary = node.plan(
        script_json=script_json,
        focus_character="(all)",
        shots_per_scene=1,
        style="mission_control_procedural",
    )
    pass1 = json.loads(pass1_json)
    assert pass1["character_count"] == 2
    # Summary should say "all cast"
    assert "all cast" in summary.lower() or "2" in summary


def test_plan_method_accepts_freeze_done_gate_and_ignores_value():
    """freeze_done_gate is a topsort-dependency input only; value must not affect output."""
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    script_json = _ledger_wrap(_sample_director())

    # Without freeze_done_gate
    out_a = node.plan(
        script_json=script_json,
        focus_character="BABA",
        shots_per_scene=1,
        style="mission_control_procedural",
    )
    # With freeze_done_gate wired (arbitrary string value)
    out_b = node.plan(
        script_json=script_json,
        focus_character="BABA",
        shots_per_scene=1,
        style="mission_control_procedural",
        freeze_done_gate="sentinel_dependency_value",
    )
    # Parse the JSON envelopes and compare -- freeze_done_gate should not affect output
    # (we compare the parsed dicts to avoid JSON whitespace differences)
    assert json.loads(out_a[0]) == json.loads(out_b[0])
    assert json.loads(out_a[1]) == json.loads(out_b[1])
    assert json.loads(out_a[2]) == json.loads(out_b[2])
    assert out_a[3] == out_b[3]


def test_input_types_has_freeze_done_gate_optional():
    from nodes.otr_video_plan import OTRVideoPlan
    schema = OTRVideoPlan.INPUT_TYPES()
    assert "optional" in schema
    assert "freeze_done_gate" in schema["optional"]
    assert schema["optional"]["freeze_done_gate"][0] == "STRING"
    # forceInput ensures the widget is replaced with an input socket
    assert schema["optional"]["freeze_done_gate"][1].get("forceInput") is True


def test_plan_method_single_char_when_focus_set():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    script_json = _ledger_wrap(_multi_char_director())
    pass1_json, _p2, pass3_json, _count, _summary = node.plan(
        script_json=script_json,
        focus_character="LEMMY",
        shots_per_scene=1,
        style="mission_control_procedural",
    )
    pass1 = json.loads(pass1_json)
    assert pass1["character_count"] == 1
    assert pass1["characters"] == ["LEMMY"]


def test_plan_output_consumable_by_batch_flux_render_parser():
    """Smoke: all 3 pass outputs match the schema BatchFluxRender's
    _parse_env_prompts expects so wiring is zero-effort."""
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    script_json = _ledger_wrap(_sample_director())
    pass1_json, pass2_json, pass3_json, _count, _summary = node.plan(
        script_json=script_json,
        focus_character="BABA",
        shots_per_scene=2,
        style="mission_control_procedural",
    )
    # Every pass output must match BatchFluxRender contract:
    # dict with "tokens" list where each token has type="environment"
    # and a non-empty description string.
    for name, js in (
        ("pass1", pass1_json), ("pass2", pass2_json), ("pass3", pass3_json)
    ):
        payload = json.loads(js)
        assert isinstance(payload, dict), f"{name} not dict"
        assert isinstance(payload.get("tokens"), list), f"{name} tokens not list"
        for tok in payload["tokens"]:
            assert tok.get("type") == "environment", f"{name} non-env token"
            assert isinstance(tok.get("description"), str)
            assert tok["description"], f"{name} empty description"
