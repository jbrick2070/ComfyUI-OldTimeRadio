"""Route-A (2026-06-28 HuMo-14B promotion) CPU tests.

REWRITTEN 2026-07-01 (rip-sfx-broll): the two retired b-roll/background roles
+ their per-role slots are GONE. Route-A now means: character_video owns
its dedicated character_video_model slot (humo_14B_169 promotion);
unknown roles RAISE (NO FALLBACKS).

Covers:
* the ONE shared role->slot map (nodes/_otr_shared/role_slots.py) -- per-role
  slot resolution + loud raises on unknown roles;
* the HuMo-14B VRAM frame cap (class override) + the exact-fit frame helper;
* the end-to-end routing guarantee: OTR_VideoDirector emits per-role slots,
  every role's resolved engine fits its role (role_compat), the policy carries
  per-role aspects, and the image dispatcher still keeps the character still.
"""
from __future__ import annotations

import json

import numpy as np
import pytest

from nodes._otr_shared import role_compat as rc
from nodes._otr_shared import role_slots as rs
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import wrapper_bridge as wb
from nodes._otr_video_engines import eng_humo
from nodes import otr_video_director as vd
from nodes import otr_image_gen_dispatcher as disp


# --------------------------------------------------------------------------- #
# role_slots -- the ONE shared map
# --------------------------------------------------------------------------- #
def test_slot_for_role_per_role_split():
    assert rs.slot_for_role("character_video") == "character_video_model"
    assert rs.slot_for_role("announcer_visual") == "announcer_video_model"
    assert rs.slot_for_role("music_visual") == "music_video_model"
    # rip-sfx-broll: unknown / retired roles RAISE (the silent legacy-slot map
    # is gone -- NO FALLBACKS)
    for dead in ("retired_broll_role", "retired_background_role", "not_a_role"):
        with pytest.raises(ValueError):
            rs.slot_for_role(dead)


def test_engine_id_for_role_reads_per_role_slot_first():
    vm = {
        "character_video_model": {"engine_id": "humo_14B_169"},
        "retired_video_slot": {"engine_id": "humo_1.7B"},
    }
    assert rs.engine_id_for_role(vm, "character_video") == "humo_14B_169"


def test_engine_id_for_role_no_fallback():
    # NO FALLBACK (2026-07-03 consolidation): an empty character slot resolves to
    # "" -- there is no legacy catch-all slot to fall back to.
    assert rs.engine_id_for_role(
        {"character_video_model": {"engine_id": ""}}, "character_video") == ""
    assert rs.engine_id_for_role({}, "character_video") == ""
    # a retired/unknown role still raises before any lookup
    with pytest.raises(ValueError):
        rs.engine_id_for_role({}, "retired_role")


def test_engine_id_for_role_accepts_bare_string_values():
    vm = {"character_video_model": "wan_ti2v"}
    assert rs.engine_id_for_role(vm, "character_video") == "wan_ti2v"


def test_video_slot_roles_has_three_per_role_slots():
    sr = rs.VIDEO_SLOT_ROLES
    assert sr["character_video_model"] == ("character_video",)
    assert sr["announcer_video_model"] == ("announcer_visual",)
    assert sr["music_video_model"] == ("music_visual",)
    # exactly three first-class video slots -- a closed set that catches ANY
    # stray/legacy slot leaking back in (stronger than per-name absence checks)
    assert set(sr) == {
        "character_video_model", "announcer_video_model", "music_video_model",
    }


# --------------------------------------------------------------------------- #
# wrapper_bridge.fit_frames_to_target -- exact-fit (trim / mirror-extend)
# --------------------------------------------------------------------------- #
def _frames(n):
    return np.arange(n * 2 * 2 * 3, dtype=np.uint8).reshape(n, 2, 2, 3)


def test_fit_frames_trims_when_over():
    out = wb.fit_frames_to_target(_frames(49), 30)
    assert out.shape[0] == 30
    assert np.array_equal(out, _frames(49)[:30])  # trim keeps the leading frames


def test_fit_frames_REFUSES_when_short():
    """Was "extends when short" -- it mirror-extended 49 frames to 120.

    The mirror is gone (operator directive 2026-08-02: original video for every
    second of audio), so a render that cannot cover its beat is terminal. The
    49-to-120 case this used to assert is precisely a capped tier being asked
    for more than it can render, which coverage planning now answers by
    splitting the beat instead of doubling frames back on themselves.
    """
    with pytest.raises(wb.MirrorExtensionForbidden):
        wb.fit_frames_to_target(_frames(49), 120)


def test_fit_frames_identity_on_match_and_empty():
    assert wb.fit_frames_to_target(_frames(40), 40).shape[0] == 40
    assert wb.fit_frames_to_target(_frames(0), 40).shape[0] == 0
    assert wb.fit_frames_to_target(_frames(40), 0).shape[0] == 40  # target<=0 -> no-op


# --------------------------------------------------------------------------- #
# HuMo-14B frame cap -- class override ONLY on the 14B tier
# --------------------------------------------------------------------------- #
def test_the_frame_cap_follows_the_MODEL_not_the_orientation():
    """BOTH 14B routes share one cap; both 1.7B routes are uncapped.

    This test used to assert that ONLY ``humo_14B_169`` was capped -- which
    encoded the defect rather than an invariant. ``HuMoEngine`` loads the SAME
    14B checkpoint at the SAME 399,360 pixels, and was uncapped to 177 while its
    landscape twin was pinned to 49. Nothing but orientation separated them, and
    orientation cannot separate them: HuMo/Wan use square patching, a global
    attention window, and RoPE reshaped to ``f*h*w``, so 480x832 and 832x480
    produce an identical 1,560-token grid per latent time.

    The cap is a property of the CHECKPOINT, so that is what it keys on.
    """
    cap = eng_humo._HUMO_14B_SAFE_RENDER_FRAMES
    # The 14B pair -- same model, same pixels, same cap.
    assert eng_humo.HuMoEngine.safe_render_frames == cap
    assert eng_humo.HuMo14BLandscapeEngine.safe_render_frames == cap

    # The 1.7B pair loads a different, far lighter checkpoint and must NOT
    # inherit a bound measured on a model roughly eight times its size -- it is
    # the downgrade target reached precisely because the 14B did not fit.
    assert eng_humo.HuMo17BEngine.safe_render_frames is None
    assert eng_humo.HuMo17BLandscapeEngine.safe_render_frames is None

    # And each declares a ceiling matching what render_clip will actually
    # produce, so no contract advertises frames its engine cannot render.
    assert eng_humo.HuMoEngine.frame_contract.max_frames == cap
    assert eng_humo.HuMo14BLandscapeEngine.frame_contract.max_frames == cap
    assert eng_humo.HuMo17BEngine.frame_contract.max_frames == \
        eng_humo._HUMO_MAX_FRAMES
    assert eng_humo.HuMo17BLandscapeEngine.frame_contract.max_frames == \
        eng_humo._HUMO_MAX_FRAMES


# --------------------------------------------------------------------------- #
# End-to-end routing: OTR_VideoDirector -> policy -> role_compat + dispatcher
# --------------------------------------------------------------------------- #
def _direct_policy():
    pol_json, = vd.OTRVideoDirector().direct(
        announcer_video_model="ltx_audio_in (16:9)",
        music_video_model="ltx_audio_in (16:9)",
        announcer_image_model="flux_gen1",
        music_image_model="flux_gen1",
        character_image_model="flux_gen1",
        fps=25, canvas_w=832, canvas_h=480,
        seed_mode="request_hash", request_seed=42,
        character_video_model="humo_14B_169 (16:9)",
    )
    return json.loads(pol_json)


def test_director_emits_per_role_video_models():
    vm = _direct_policy()["video_models"]
    assert vm["character_video_model"]["engine_id"] == "humo_14B_169"
    # exactly the three per-role video-model slots (closed set)
    assert set(vm) == {
        "announcer_video_model", "music_video_model", "character_video_model",
    }


def test_every_role_engine_fits_its_role():
    vm = _direct_policy()["video_models"]
    for role in ("announcer_visual", "music_visual", "character_video"):
        eid = rs.engine_id_for_role(vm, role)
        eng = vreg.get_engine(eid)
        desc = {"engine_id": eid, "roles": tuple(getattr(eng, "roles", ())),
                "required_inputs": tuple(getattr(eng, "required_inputs", ()))}
        assert rc.engine_fits_role(desc, role), (role, eid)


def test_policy_aspects_has_per_role_entries():
    aspects = _direct_policy()["aspects"]
    assert set(aspects) == {"announcer_visual", "music_visual",
                            "character_video"}
    # humo_14B_169 is the wide tier -> the character still is minted 16:9.
    assert aspects["character_video"] == "wide"


