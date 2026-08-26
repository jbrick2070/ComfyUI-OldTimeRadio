# -*- coding: utf-8 -*-
"""The retired-engine migration policy, EXERCISED at all five boundaries.

A user-saved workflow or an external API client may still name any of the
engine ids retired since the 2026-08-06 SFX rip (the set has grown with each
later retirement -- see the count guard below). The contract
(docs/2026-08-06-BUILD-SPEC-rip-sfx.md section 10): a stale selection fails
with a NAMED :class:`RetiredEngineError` -- ONE type, ONE ``reason_code``
spelling, ONE message shape -- and never silently resolves to another engine.

The five ingress boundaries, each parameterized over EVERY retired id:

  1. the director's ``_resolve_and_validate`` (saved node / custom json),
  2. ``parse_engine_override`` (the OTR_FORCE_ENGINE_MAP force map),
  3. ``_render_one`` (the path the shipped ``/otr/video_render_single``
     HTTP endpoint reaches),
  4. multi-clip session creation (``render_beat_coverage`` on a frozen
     ledger's coverage plan),
  5. the registry's ``assert_usable``.

Every boundary calls ``check_retired_engine`` AFTER public/legacy-name
resolution, so an alias cannot slip past.
"""
from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401  -- populate the registry
from nodes._otr_shared.public_engines import (
    RETIRED_ENGINE_IDS,
    RetiredEngineError,
    check_retired_engine,
    resolve_engine_id,
)
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import render_driver as rd

RETIRED = sorted(RETIRED_ENGINE_IDS)


# ---------------------------------------------------------------------------
# The contract itself: type, fields, message -- ONE spelling, pinned
# ---------------------------------------------------------------------------

def test_the_retired_set_is_exactly_the_sixteen_known_ids():
    # Five from the 2026-08-06 SFX-bed rip, five from the 2026-08-23 dormant-3D
    # retirement (lean-mean order 4), five from the 2026-08-23 Ghost narrowing
    # (operator: "delete any animatediff that are not haunted"), and one from
    # the 2026-08-26 large-Wan rip (operator: "rip the large wan we don't
    # need") -- the 14B `wan_i2v` lane, whose 19.82 GiB of weights never fit
    # the 14.5 GiB target and only ran by offloading continuously.
    # EXACT on purpose: the set is append-only, and an id appearing here that no
    # ruling added -- or one vanishing -- must fail the suite, not drift. When a
    # retirement lands, the COUNT in this test's NAME moves with the set; a
    # guard whose name says fifteen while it asserts sixteen has stopped being
    # a guard and started being a comment.
    assert RETIRED_ENGINE_IDS == frozenset({
        "animatediff15_video",
        "animatediff15_v2_video",
        "animatediff15_v3_video",
        "animatediff15_h3_video",
        "animatediff15_h5_video",
        "cloud_vidu_q2_pro_fast_720p_sfx",
        "google_vid_sfx_omni",
        "google_vid_sfx_veo_fast",
        "google_vid_sfx_veo_lite",
        "google_vid_sfx_veo_pro",
        "triposg_talk",
        "hunyuan3d_talk",
        "trellis_talk",
        "triposr",
        "still_parallax",
        "wan_i2v",
    })
    assert isinstance(RETIRED_ENGINE_IDS, frozenset)
    assert len(RETIRED_ENGINE_IDS) == 16, (
        "the count in this test's name is part of the guard -- rename it in "
        "the same edit that grows the set")


@pytest.mark.parametrize("engine_id", RETIRED)
def test_the_error_contract_is_pinned(engine_id):
    with pytest.raises(RetiredEngineError) as ei:
        check_retired_engine(engine_id)
    exc = ei.value
    assert isinstance(exc, ValueError)          # boundary wrappers may catch ValueError
    assert exc.engine_id == engine_id
    assert exc.reason_code == "retired_engine"  # lower-case; the ONE spelling
    assert str(exc) == (
        "video engine '%s' is retired and is no longer selectable" % engine_id)


def test_non_retired_ids_pass_through_untouched():
    for ok in ("", "humo", "ltx_video", "google_veo_video",
               "cloud_vidu_q2_pro_fast_720p", "unknown_engine"):
        check_retired_engine(ok)                # must not raise


def test_no_alias_resolves_to_a_retired_id():
    # resolve-then-check is the boundary order; prove resolution can never
    # LAUNDER a retired id in from a public or legacy spelling.
    for rid in RETIRED:
        assert resolve_engine_id(rid) == rid    # no alias maps it elsewhere


# ---------------------------------------------------------------------------
# Boundary 1: the director (saved node pick + custom_models_json)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_id", RETIRED)
def test_director_saved_pick_raises_named(engine_id):
    from nodes.otr_video_director import OTRVideoDirector
    with pytest.raises(RetiredEngineError, match="retired"):
        OTRVideoDirector._resolve_and_validate(
            "music_video_model", engine_id, {}, [], [])


@pytest.mark.parametrize("engine_id", RETIRED)
def test_director_custom_json_raises_named(engine_id):
    from nodes.otr_video_director import ADD_CUSTOM, OTRVideoDirector
    with pytest.raises(RetiredEngineError, match="retired"):
        OTRVideoDirector._resolve_and_validate(
            "character_video_model", ADD_CUSTOM,
            {"character_video_model": engine_id}, [], [])


# ---------------------------------------------------------------------------
# Boundary 2: the force map
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_id", RETIRED)
def test_force_map_raises_named(engine_id):
    with pytest.raises(RetiredEngineError, match=engine_id):
        rd.parse_engine_override("*=" + engine_id)


# ---------------------------------------------------------------------------
# Boundary 3: _render_one (the direct-render HTTP endpoint's path)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_id", RETIRED)
def test_render_one_raises_named_before_the_generic_lookup(engine_id):
    request = {"text_prompt": "probe",
               "asset_refs": {"init_image": "probe.png",
                              "audio_ref": "probe.wav"}}
    with pytest.raises(RetiredEngineError, match=engine_id):
        rd._render_one(engine_id, request, force_oom=False)


# ---------------------------------------------------------------------------
# Boundary 4: multi-clip session creation (frozen-ledger ingress)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_id", RETIRED)
def test_multi_clip_session_creation_raises_named(engine_id):
    shot = {
        "shot_id": "shot_stale", "beat_id": "b001",
        "engine_id": engine_id, "role": "music_visual",
        "target_frame_count": 50,
        "coverage_plan": {
            "target_visible_frames": 50, "join_mode": "chain",
            "segments": [
                {"index": 0, "render_frames": 26, "drop_head": 0, "trim_tail": 0},
                {"index": 1, "render_frames": 25, "drop_head": 1, "trim_tail": 0},
            ],
        },
    }
    with pytest.raises(RetiredEngineError, match=engine_id):
        rd.render_beat_coverage(
            shot, {}, request_builder=lambda *a, **k: {})


# ---------------------------------------------------------------------------
# Boundary 5: the registry's assert_usable
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("engine_id", RETIRED)
def test_assert_usable_raises_named_not_generic(engine_id):
    with pytest.raises(RetiredEngineError, match="retired"):
        vreg.assert_usable(engine_id, "music_visual")
