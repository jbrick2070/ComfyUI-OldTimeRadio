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
    assert with_end["tokens"][-1]["kind"] == "end"
    assert all(t["kind"] == "start" for t in without["tokens"])


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


def test_build_shot_plan_shot_ids_include_character_slug():
    from nodes.otr_video_plan import build_shot_plan
    director_json = json.dumps(_sample_director())
    plan = build_shot_plan(
        director_json, "KENJI CROSS", shots_per_scene=1,
    )
    for tok in plan["tokens"]:
        assert "kenji_cross" in tok["shot_id"]


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
    assert OTRVideoPlan.RETURN_TYPES == ("STRING", "INT", "STRING")
    assert OTRVideoPlan.RETURN_NAMES[0] == "prompt_list_json"
    assert OTRVideoPlan.RETURN_NAMES[1] == "prompt_count"


def test_function_and_category():
    from nodes.otr_video_plan import OTRVideoPlan
    assert OTRVideoPlan.FUNCTION == "plan"
    assert OTRVideoPlan.CATEGORY.startswith("OldTimeRadio")


def test_plan_method_end_to_end():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    director_json = json.dumps(_sample_director())
    prompt_list_json, prompt_count, summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=3,
        genre_flavor="hard_sci_fi",
    )
    payload = json.loads(prompt_list_json)
    assert payload["total_prompts"] == prompt_count
    assert payload["focus_character"] == "BABA"
    # Envelope BatchFluxRender reads
    assert "tokens" in payload
    assert payload["tokens"][0]["type"] == "environment"
    # Summary should be a string with multiple lines
    assert "focus character" in summary.lower()
    assert "total prompts" in summary.lower()


def test_plan_method_handles_none_genre():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    director_json = json.dumps(_sample_director())
    prompt_list_json, count, _summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=1,
        genre_flavor="(none)",
    )
    payload = json.loads(prompt_list_json)
    assert payload["genre_flavor"] == ""


def test_plan_method_empty_scenes_still_returns_envelope():
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    director_json = json.dumps({
        "voice_assignments": {"BABA": {"voice_preset": "x", "notes": "old"}},
        "visual_plan": {"characters": {"BABA": {"portrait_prompt": "p"}}, "scenes": []},
    })
    prompt_list_json, count, summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=3,
        genre_flavor="hard_sci_fi",
    )
    assert count == 0
    payload = json.loads(prompt_list_json)
    assert payload["tokens"] == []
    assert payload["scenes_covered"] == 0


def test_plan_output_consumable_by_batch_flux_render_parser():
    """Smoke: our output must match the schema BatchFluxRender's
    _parse_env_prompts expects so wiring is zero-effort."""
    from nodes.otr_video_plan import OTRVideoPlan
    node = OTRVideoPlan()
    director_json = json.dumps(_sample_director())
    prompt_list_json, _count, _summary = node.plan(
        director_json=director_json,
        focus_character="BABA",
        shots_per_scene=2,
        genre_flavor="hard_sci_fi",
    )
    payload = json.loads(prompt_list_json)
    # BatchFluxRender contract: dict with "tokens", each having
    # type="environment" and a "description" string.
    assert isinstance(payload, dict)
    assert isinstance(payload.get("tokens"), list)
    for tok in payload["tokens"]:
        assert tok.get("type") == "environment"
        assert isinstance(tok.get("description"), str)
        assert tok["description"]  # non-empty
