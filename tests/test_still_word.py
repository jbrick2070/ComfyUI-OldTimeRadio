"""still_word (Sprint B, 2026-07-03) -- the model-agnostic word/title still.

A still_flat SIBLING VIDEO engine whose base still is minted from a WORD/TITLE-
driven prompt (character/announcer beats -> the spoken line as a readable word
card; music beats -> an abstract picture of the episode title, no words). The
IMAGE model that mints it is chosen INDEPENDENTLY, so still_word is model-
agnostic. NO FALLBACKS: a missing base still fails LOUD (never the dark floor);
a blank spoken line / blank title fails LOUD in the composer.

CPU-only: no ffmpeg render is exercised here (the fail-LOUD path raises before
any encode). Covers registration in all sites, the pure composer (word + title
modes, edge cases, determinism, model-agnosticism), the policy reader, the
render_clip no-floor guard, and the derive_image_prompts word/title branch.
"""
from __future__ import annotations

import inspect
import json

import pytest

from nodes import otr_meta_brief_image_prompt as ip
from nodes import otr_video_director as vd
from nodes._otr_video_engines import cheap_families, registry as vreg
from nodes._otr_video_engines import render_driver


# --------------------------------------------------------------------------- #
# Registration -- all sites (else still_word never receives a still)
# --------------------------------------------------------------------------- #
def test_still_word_registered_engine():
    assert "still_word" in vreg.all_engine_names()
    eng = vreg.get_engine("still_word")
    assert eng.family == "static_image_gen"
    assert eng.default_roles == ()                 # selectable, never a default
    assert eng.accepts_still is True
    assert eng._still_motion is False              # flat hold (like still_flat)
    assert eng._require_still is True              # NO dark floor


def test_still_word_capabilities_row():
    assert "still_word" in vreg.CAPABILITIES
    row = vreg.CAPABILITIES["still_word"]
    assert row["cpu_ok"] is True
    # the registry-consistency invariant (CAPABILITIES == registered engines).
    assert set(vreg.CAPABILITIES) == set(vreg.all_engine_names())


def test_still_word_engine_family_map():
    assert render_driver.ENGINE_FAMILY["still_word"] == "static_image_gen"


def test_still_word_in_video_combo_and_parses():
    parsed = {vd._engine_id_from_pick(c) for c in vd._video_model_combo()}
    assert "still_word" in parsed                  # selectable in the dropdown


def test_still_flat_sibling_unchanged_no_require_still():
    # the sibling keeps its always-renders floor (byte-identical behavior).
    assert cheap_families.StillFlatFamily._require_still is False
    assert cheap_families.StillWordFamily._require_still is True


# --------------------------------------------------------------------------- #
# compose_still_word_prompt -- pure, deterministic, model-agnostic, fail-LOUD
# --------------------------------------------------------------------------- #
_META = {"episode_title": "The Signal From Deck Nine",
         "story_brief_terms": {"setting": ["a fog-bound harbor town"],
                               "lighting": ["moody"]}}


def test_word_mode_renders_the_spoken_line():
    out = ip.compose_still_word_prompt(_META, "character_video",
                                       "We have to go back.")
    assert "We have to go back." in out
    assert "title card" in out.lower()
    # WORD mode WANTS text -> the no-text clause must NOT be appended.
    assert "no on-screen text" not in out


def test_word_mode_scrubs_stage_direction_and_speaker_label():
    out = ip.compose_still_word_prompt(
        _META, "announcer_visual", "NARRATOR: [whispering] The signal is gone")
    assert "The signal is gone" in out
    assert "[whispering]" not in out
    assert "NARRATOR:" not in out


def test_word_mode_keeps_quotes_ellipsis_emdash():
    line = 'He said "run" -- and then... silence'
    out = ip.compose_still_word_prompt(_META, "character_video", line)
    assert '"run"' in out
    assert "--" in out
    assert "..." in out


def test_word_mode_blank_line_fails_loud():
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt(_META, "character_video", "")
    # a line that is ONLY a stage direction scrubs to empty -> also LOUD.
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt(_META, "character_video", "[static hiss]")


def test_music_mode_abstract_title_no_words():
    out = ip.compose_still_word_prompt(_META, "music_visual", "")
    assert "The Signal From Deck Nine" in out
    assert "abstract" in out.lower()
    # TITLE mode is wordless -> the no-text clause IS appended.
    assert out.endswith("no on-screen text")


def test_music_mode_blank_title_fails_loud():
    with pytest.raises(ValueError, match="NO FALLBACK"):
        ip.compose_still_word_prompt({"story_brief_terms": {}}, "music_visual", "")


def test_compose_is_deterministic():
    a = ip.compose_still_word_prompt(_META, "character_video", "Once more.")
    b = ip.compose_still_word_prompt(_META, "character_video", "Once more.")
    assert a == b


def test_compose_is_model_agnostic():
    # the composer takes NO engine argument -> the prompt is identical regardless
    # of which image engine mints it (operator priority: model-agnostic).
    params = list(inspect.signature(ip.compose_still_word_prompt).parameters)
    assert params == ["meta", "role", "beat_line"]
    assert "engine" not in params


# --------------------------------------------------------------------------- #
# _still_word_roles_from_policy -- reads video_models via role_slots
# --------------------------------------------------------------------------- #
def test_roles_from_policy_resolves_video_models():
    policy = json.dumps({"video_models": {
        "character_video_model": {"engine_id": "still_word"},
        "announcer_video_model": {"engine_id": "ltx_video"},
        "music_video_model": {"engine_id": "still_word"},
    }})
    roles = ip._still_word_roles_from_policy(policy)
    assert roles == {"character_video", "music_visual"}


def test_roles_from_policy_empty_when_absent():
    assert ip._still_word_roles_from_policy("{}") == set()
    assert ip._still_word_roles_from_policy(
        json.dumps({"video_models": {"character_video_model":
                                     {"engine_id": "ltx_video"}}})) == set()


# --------------------------------------------------------------------------- #
# render_clip -- NO dark floor (fail LOUD when the base still is missing)
# --------------------------------------------------------------------------- #
def test_render_clip_fails_loud_without_a_still():
    req = {"asset_refs": {}, "canvas": {"w": 832, "h": 480, "fps": 25},
           "timing": {"target_frame_count": 10}, "shot_id": "s_word"}
    with pytest.raises(RuntimeError, match="NO FALLBACKS"):
        cheap_families.StillWordFamily().render_clip(req)


# --------------------------------------------------------------------------- #
# derive_image_prompts -- the word/title branch mints still_word-sourced stills
# --------------------------------------------------------------------------- #
def _cast():
    return [{"char_id": "c01",
             "character_description": "a weathered dock inspector in an oilskin coat"}]


def _lines():
    return [
        {"line_id": "b001", "speaker_role": "character", "char_id": "c01",
         "text": "We have to go back.", "start_s": 1.0, "dur_s": 2.0},
        {"line_id": "b002", "speaker_role": "announcer", "char_id": "announcer",
         "text": "And now, our story.", "start_s": 3.0, "dur_s": 1.0},
    ]


def test_derive_image_prompts_word_and_title_stills():
    payload, _warn = ip.derive_image_prompts(
        _cast(), _META, llm_fn=None, lines=_lines(),
        still_word_roles={"character_video", "announcer_visual", "music_visual"})
    objs = payload["objects"]
    word_objs = [o for o in objs if o.get("source") == "still_word"]
    assert word_objs, "expected still_word-sourced scene stills"
    # every still_word beat is ONE scene-still object (picked up by _still_index).
    for o in word_objs:
        assert o["object_id"] == "still_%s" % o["beat_id"]
        assert o["prompt"] and o["prompt_hash"]
    # the character beat's spoken line drives its still.
    char = next((o for o in word_objs if o.get("char_id") == "c01"), None)
    assert char is not None
    assert "We have to go back." in char["prompt"]
    # a music beat (title mode) carries the episode title, no words.
    music = next((o for o in word_objs if o.get("role") == "music_visual"), None)
    if music is not None:                            # music open beat present
        assert "The Signal From Deck Nine" in music["prompt"]
        assert music["prompt"].endswith("no on-screen text")


def test_derive_image_prompts_no_still_word_roles_is_legacy():
    # with no still_word roles the branch is inert (no still_word source objects).
    payload, _warn = ip.derive_image_prompts(
        _cast(), _META, llm_fn=None, lines=_lines())
    assert not [o for o in payload["objects"] if o.get("source") == "still_word"]
