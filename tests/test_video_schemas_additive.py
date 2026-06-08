"""Additive CPU coverage for the typed video-platform schemas (AS-4).

New, self-contained tests for nodes/_otr_video_engines/schemas.py: the
extra='forbid' unknown-key rejection, the per-family required-input validator on
VideoRequest, and the always-silent CanonicalClip defaults (V-1). Pydantic is a
base dep; no GPU, no model load. Complements test_video_platform_aseam.py.
UTF-8, no BOM, ASCII-only, SFW.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from nodes._otr_video_engines import schemas as sc


def _canvas():
    return sc.Canvas(w=1280, h=720, fps=25)


def _req(**over):
    base = dict(
        request_id="r1", shot_id="shot_0001", role="background_abstract",
        family_hint="abstract", profile_id="p1",
        timing=sc.Timing(), canvas=_canvas(),
    )
    base.update(over)
    return sc.VideoRequest(**base)


def test_families_and_tokens_are_single_sourced():
    assert len(sc.FAMILIES) == 7
    assert "character_3d" not in sc.FAMILIES          # 3D ships with B
    assert sc.REQUIRED_INPUT_TOKENS == (
        "text_prompt", "init_image", "audio_ref", "base_clip_ref")
    assert set(sc.FAMILY_REQUIRED_INPUTS) == set(sc.FAMILIES)


def test_abstract_request_needs_no_inputs():
    req = _req()
    assert req.family_hint == "abstract"
    assert req.canvas.aspect_policy == "pad"          # default, never stretch


def test_unknown_family_hint_rejected():
    with pytest.raises(ValidationError):
        _req(family_hint="hologram")


def test_extra_keys_forbidden_on_request_and_canvas():
    with pytest.raises(ValidationError):
        _req(surprise_field="nope")
    with pytest.raises(ValidationError):
        sc.Canvas(w=1, h=1, fps=1, bogus=2)


def test_canvas_requires_w_h_fps():
    with pytest.raises(ValidationError):
        sc.Canvas(w=1280, h=720)                       # missing fps


def test_text_to_video_requires_text_prompt():
    with pytest.raises(ValidationError):
        _req(family_hint="text_to_video")
    ok = _req(family_hint="text_to_video", text_prompt="a quiet comms shack")
    assert ok.text_prompt


def test_audio_driven_face_requires_audio_and_init_image():
    with pytest.raises(ValidationError):
        _req(family_hint="audio_driven_face")
    ok = _req(
        family_hint="audio_driven_face",
        audio_ref=sc.AudioRef(path="master.wav"),
        asset_refs={"init_image": "portrait.png"},
    )
    assert ok.audio_ref.path == "master.wav"


def test_static_image_gen_requires_text_or_init_image():
    with pytest.raises(ValidationError):
        _req(family_hint="static_image_gen")
    assert _req(
        family_hint="static_image_gen", text_prompt="station card"
    ).family_hint == "static_image_gen"
    assert _req(
        family_hint="static_image_gen", asset_refs={"init_image": "x.png"}
    ).family_hint == "static_image_gen"


def test_canonical_clip_is_always_silent_bt709_by_default():
    clip = sc.CanonicalClip(clip_id="c1", path="out.mp4")
    assert clip.has_audio is False                     # V-1
    assert clip.pixel_format == "yuv420p"
    assert clip.color_primaries == "bt709"
    assert clip.transfer == "bt709"
    assert clip.matrix == "bt709"
    assert clip.alpha == "none"


def test_canonical_clip_forbids_extra_keys():
    with pytest.raises(ValidationError):
        sc.CanonicalClip(clip_id="c1", path="out.mp4", loud=True)


def test_ledger_section_defaults():
    sec = sc.VideoLedgerSection()
    assert sec.video_revision == 1
    assert sec.runtime_fallback_decisions == []
    assert sec.fps == 25


def test_execution_group_defaults_to_consumer():
    g = sc.ExecutionGroup(group_id="g1")
    assert g.kind == "consumer"
    assert g.depends_on == [] and g.produces_base_for == []
