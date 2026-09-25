"""Cloud LTX 2.5 A2V still routing (CPU).

``cloud_ltx25_audio_in`` is the Comfy partner audio-in lane: its first frame
is the beat's wide scene still. It must NOT join the H3 exclusion from
``_engine_scene_init_required`` (that exclusion exists so a lip-sync
tokenizer keeps a FACE).
"""
from __future__ import annotations

import pytest

from nodes._otr_video_engines import render_driver as rd
from nodes._otr_video_engines.render_errors import DeferredImageGapError

ENGINE = "cloud_ltx25_audio_in"
SCENE = "C:/x/otr/episodes/ep/b003_scene_character.png"
PORTRAIT = "C:/x/otr/episodes/ep/c1_portrait.png"
BOOKEND = "C:/x/otr/episodes/ep/b000_scene_open.png"


def _ledger(images):
    return {
        "meta": {},
        "cast": [{"char_id": "c1", "name": "Ann"}],
        "lines": [],
        "images": {"images": images},
        "video": {"shots": []},
    }


def _shot(role, **over):
    s = {
        "shot_id": "shot_b003",
        "engine_id": ENGINE,
        "role": role,
        "char_id": "c1",
        "target_frame_count": 120,
        "render_request_hash": "deadbeef",
        "creative": {
            "text_prompt": "the camera dollies in; the grille vibrates"
        },
        "source_line_ids": ["b003"],
    }
    s.update(over)
    return s


def test_cloud_a2v_stays_on_generic_scene_init_overwrite():
    """Opposite of H3: A2V first-frame IS the scene still."""
    required = (
        "init_image" in rd._required_inputs_for_engine(
            ENGINE, "audio_conditioned_video")
        and ENGINE != "minimax_h3_audio_in"
    )
    assert required is True


def test_character_beat_takes_the_wide_scene_still_not_the_portrait():
    led = _ledger([
        {"kind": "portrait", "object_id": "c1", "path": PORTRAIT},
        {"kind": "scene_character", "beat_id": "b003", "path": SCENE},
    ])
    req = rd.build_request_from_shot(
        _shot("character_video"), led)
    assert req["asset_refs"]["init_image"] == SCENE
    assert req["asset_refs"]["init_image"] != PORTRAIT
    assert req["observability"].get("init_source") == "scene_still"


def test_bookend_takes_the_scene_open_still():
    led = _ledger([
        {"kind": "scene_open", "beat_id": "b000_music_open", "path": BOOKEND},
    ])
    req = rd.build_request_from_shot(
        _shot("music_visual", shot_id="shot_b000_music_open",
              source_line_ids=["b000_music_open"], char_id=""),
        led)
    assert req["asset_refs"]["init_image"] == BOOKEND


def test_missing_scene_still_fails_loud_not_silent_portrait():
    led = _ledger([
        {"kind": "portrait", "object_id": "c1", "path": PORTRAIT},
    ])
    with pytest.raises(DeferredImageGapError, match="NO scene still"):
        rd.build_request_from_shot(_shot("character_video"), led)


def test_mouth_human_does_not_switch_init_to_portrait():
    """W7 face classification is voice ownership; the still stays the scene."""
    led = _ledger([
        {"kind": "portrait", "object_id": "c1", "path": PORTRAIT},
        {"kind": "scene_character", "beat_id": "b003", "path": SCENE},
    ])
    shot = _shot("character_video")
    assert rd._is_character_face_beat(shot) is True
    req = rd.build_request_from_shot(shot, led)
    assert req["observability"]["init_source"] == "scene_still"
    assert req["asset_refs"]["init_image"] == SCENE
