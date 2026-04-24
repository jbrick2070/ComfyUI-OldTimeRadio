"""
test_otr_video_plan.py  --  OTR_VideoPlan adapter unit tests
=============================================================

Validates ``nodes.otr_video_plan`` without torch, diffusers, or
ComfyUI.  Covers the pure helpers + node surface area + the
fallback chain for missing Director fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


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
    assert "cyberpunk" in resolve_era_tail("cyberpunk").lower()
    assert "post-apocalyptic" in resolve_era_tail("post_apocalyptic").lower()
    assert "lovecraftian" in resolve_era_tail("cosmic_horror").lower()
    assert "near-future" in resolve_era_tail("hard_sci_fi").lower()


def test_resolve_era_tail_empty_and_unknown():
    from nodes.otr_video_plan import resolve_era_tail
    default = resolve_era_tail("")
    assert default
    assert resolve_era_tail("gothic_romance") == default
    assert resolve_era_tail("nonexistent_genre") == default


def test_resolve_era_tail_normalizes_case_and_punctuation():
    from nodes.otr_video_plan import resolve_era_tail
    r1 = resolve_era_tail("cyberpunk")
    r2 = resolve_era_tail("CYBERPUNK")
    r3 = resolve_era_tail("Cyberpunk")
    assert r1 == r2 == r3


# ------------------------------------------------------------------
# resolve_character_portrait
# ------------------------------------------------------------------


def test_portrait_canonical_path():
    from nodes.otr_video_plan import resolve_character_portrait
    director = {
        "visual_plan": {
            "characters": {
                "BABA": {
                    "portrait_prompt": "Cinematic portrait of elderly spacer, silver braids"
                }
            }
        }
    }
    result = resolve_character_portrait(director, "BABA", "style_tail")
    assert "silver braids" in result
    assert "style_tail" not in result  # canonical path doesn't append tail


def test_portrait_fallback_from_notes():
    from nodes.otr_video_plan import resolve_character_portrait
    director = {
        "visual_plan": {"characters": {"BABA": {}}},
        "voice_assignments": {
            "BABA": {"voice_preset": "v2/en_speaker_8",
                     "notes": "Female, 60s, weary, low"}
        },
    }
    result = resolve_character_portrait(director, "BABA", "cinematic tail")
    assert "BABA" in result
    assert "Female, 60s, weary, low" in result
    assert "cinematic tail" in result


def test_portrait_fallback_generic():
    """No visual_plan, no voice_assignments -> generic template."""
    from nodes.otr_video_plan import resolve_character_portrait
    director = {}
    result = resolve_character_portrait(director, "BOOEY", "the tail")
    assert "BOOEY" in result
    assert "the tail" in result


def test_portrait_empty_character_name():
    from nodes.otr_video_plan import resolve_character_portrait
    result = resolve_character_portrait({}, "", "style")
    assert result  # non-empty
    assert "style" in result


def test_portrait_empty_portrait_prompt_triggers_fallback():
    """If visual_plan.characters[NAME].portrait_prompt is '', fall through."""
    from nodes.otr_video_plan import resolve_character_portrait
    director = {
        "visual_plan": {"characters": {"BABA": {"portrait_prompt": ""}}},
        "voice_assignments": {"BABA": {"notes": "old pilot"}},
    }
    result = resolve_character_portrait(director, "BABA", "x")
    assert "old pilot" in result


# ------------------------------------------------------------------
# extract_scenes
# ------------------------------------------------------------------


def test_extract_scenes_basic():
    from nodes.otr_video_plan import extract_scenes
    director = {
        "visual_plan": {
            "scenes": [
                {"scene_id": "scene_1", "visual_prompt": "bridge"},
                {"scene_id": "scene_2", "visual_prompt": "corridor"},
            ]
        }
    }
    scenes = extract_scenes(director)
    assert len(scenes) == 2
    assert scenes[0]["scene_id"] == "scene_1"


def test_extract_scenes_empty():
    from nodes.otr_video_plan import extract_scenes
    assert extract_scenes({}) == []
    assert extract_scenes({"visual_plan": {}}) == []
    assert extract_scenes({"visual_plan": {"scenes": []}}) == []


def test_extract_scenes_malformed_skips_non_dicts():
    from nodes.otr_video_plan import extract_scenes
    director = {
        "visual_plan": {
            "scenes": [
                {"scene_id": "good", "visual_prompt": "ok"},
                "bogus_string_entry",
                42,
                {"scene_id": "also_good"},
            ]
        }
    }
    scenes = extract_scenes(director)
    assert len(scenes) == 2


def test_extract_scenes_malformed_root_returns_empty():
    from nodes.otr_video_plan import extract_scenes
    director = {"visual_plan": {"scenes": "not a list"}}
    assert extract_scenes(director) == []


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
# build_shot_plan — the core integration
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
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json,
        focus_character="BABA",
        shots_per_scene=3,
        genre_flavor="hard_sci_fi",
    )
    # 2 scenes * 3 shots + 1 final end = 7
    assert plan["total_prompts"] == 7
    assert plan["scenes_covered"] == 2
    assert plan["shots_per_scene"] == 3
    assert plan["focus_character"] == "BABA"


def test_build_shot_plan_tokens_are_env_shaped():
    """BatchFluxRender reads tokens where type='environment'."""
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json,
        focus_character="BABA",
        shots_per_scene=2,
        genre_flavor="hard_sci_fi",
    )
    for tok in plan["tokens"]:
        assert tok["type"] == "environment"
        assert tok["description"]
        assert tok["shot_id"]
        assert tok["focus_character"] == "BABA"


def test_build_shot_plan_compose_includes_portrait_and_scene():
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json,
        focus_character="BABA",
        shots_per_scene=1,
        genre_flavor="hard_sci_fi",
    )
    first = plan["tokens"][0]
    desc = first["description"]
    # Should contain portrait keywords
    assert "silver braids" in desc
    # Should contain scene_1's visual_prompt text
    assert "dim comms bay" in desc or "red alert pulse" in desc
    # Should contain era tail
    assert "near-future" in desc.lower() or "sci-fi" in desc.lower()


def test_build_shot_plan_fallback_to_notes():
    """Director with no portrait_prompt -> uses voice_assignments.notes."""
    from nodes.otr_video_plan import build_shot_plan
    director = _sample_director()
    director["visual_plan"]["characters"]["BABA"]["portrait_prompt"] = ""
    director_json = json.dumps(director)
    plan = build_shot_plan(
        director_json,
        focus_character="BABA",
        shots_per_scene=1,
    )
    first = plan["tokens"][0]
    assert "Female, 60s, weary, low" in first["description"]


def test_build_shot_plan_final_end_frame_toggle():
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())

    with_end = build_shot_plan(
        director_json, "BABA", shots_per_scene=3,
        include_final_end_frame=True,
    )
    without = build_shot_plan(
        director_json, "BABA", shots_per_scene=3,
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
    """Empty Director -> empty token list, doesn't crash."""
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
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json, "BABA", shots_per_scene=1,
        genre_flavor="gothic_romance",  # not in dict
    )
    assert "timeless" in plan["era_tail"].lower()


def test_build_shot_plan_shot_ids_unique():
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json, "BABA", shots_per_scene=3,
    )
    shot_ids = [t["shot_id"] for t in plan["tokens"]]
    assert len(shot_ids) == len(set(shot_ids)), \
        "shot_ids must be unique across the episode"


def test_build_shot_plan_frame_ids_are_global_4digit():
    """Schema: frame IDs are global 4-digit, no character suffix."""
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json, "KENJI CROSS", shots_per_scene=1,
    )
    for tok in plan["tokens"]:
        # frame_0000, frame_0001, ...
        assert tok["frame_id"].startswith("frame_")
        assert len(tok["frame_id"]) == len("frame_0000")


def test_build_shot_plan_shot_ids_are_global_3digit():
    """Schema: shot IDs are global 'shot_NNN' across the whole episode."""
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json, "BABA", shots_per_scene=3,
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
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json, "BABA", shots_per_scene=1,
    )
    for shot in plan["shots"]:
        assert len(shot["segments"]) >= 1
        # Default: 1 clip per shot
        assert shot["segments"][0]["clip_id"] == f"{shot['shot_id']}_c1"


def test_build_shot_plan_shared_boundary_frames():
    """Schema: adjacent shots share boundary frames (FLF chain)."""
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json, "BABA", shots_per_scene=2,
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
    director_json = json.dumps(director)
    plan = build_shot_plan(director_json, "BABA", shots_per_scene=1)
    # First scene's shot should include the shot_description fallback
    scene_1_tokens = [
        t for t in plan["tokens"] if t["scene_id"] == "scene_1"
    ]
    assert any("custom_desc_xyz" in t["description"] for t in scene_1_tokens)


# ------------------------------------------------------------------
# ComfyUI node class surface area
# ------------------------------------------------------------------


def test_input_types_schema():
    from nodes.otr_video_plan import OTRVideoPlan
    schema = OTRVideoPlan.INPUT_TYPES()
    assert "required" in schema
    assert "director_json" in schema["required"]
    assert "focus_character" in schema["required"]
    assert "shots_per_scene" in schema["required"]
    assert "genre_flavor" in schema["required"]
    assert schema["required"]["director_json"][0] == "STRING"
    assert schema["required"]["shots_per_scene"][0] == "INT"


def test_director_json_multiline():
    from nodes.otr_video_plan import OTRVideoPlan
    schema = OTRVideoPlan.INPUT_TYPES()
    assert schema["required"]["director_json"][1].get("multiline") is True


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
    director_json = json.dumps(_sample_director())
    pass1_json, pass2_json, pass3_json, pass3_count, summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=3,
        genre_flavor="hard_sci_fi",
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
    director_json = json.dumps(_sample_director())
    _p1, _p2, pass3_json, _count, _summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=1,
        genre_flavor="(none)",
    )
    payload = json.loads(pass3_json)
    assert payload["genre_flavor"] == ""


def test_plan_method_empty_scenes_still_returns_envelope():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    director_json = json.dumps({
        "voice_assignments": {"BABA": {"voice_preset": "x", "notes": "old"}},
        "visual_plan": {"characters": {"BABA": {"portrait_prompt": "p"}}, "scenes": []},
    })
    _p1, _p2, pass3_json, count, _summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=3,
        genre_flavor="hard_sci_fi",
    )
    assert count == 0
    payload = json.loads(pass3_json)
    assert payload["tokens"] == []
    assert payload["scenes_covered"] == 0


def test_plan_output_consumable_by_batch_flux_render_parser():
    """Smoke: all 3 pass outputs match the schema BatchFluxRender's
    _parse_env_prompts expects so wiring is zero-effort."""
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    director_json = json.dumps(_sample_director())
    pass1_json, pass2_json, pass3_json, _count, _summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=2,
        genre_flavor="hard_sci_fi",
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
