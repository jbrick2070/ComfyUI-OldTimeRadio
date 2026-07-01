"""Route-A (2026-06-28 HuMo-14B promotion) CPU tests.

Covers the per-sub-role video routing feature that promotes humo_14B_169 onto the
character_video role while scene_broll / background_abstract ride lighter engines:

* the ONE shared role->slot map (nodes/_otr_shared/role_slots.py) -- per-role slot
  resolution + the legacy other_beats migration fallback;
* the HuMo-14B VRAM frame cap (class override) + the exact-fit frame helper;
* the end-to-end routing guarantee: OTR_VideoDirector emits per-role slots, every
  role's resolved engine fits its role (role_compat), the policy carries per-role
  aspects, and the image dispatcher still keeps the character still HuMo needs.
"""
from __future__ import annotations

import json

import numpy as np

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
    assert rs.slot_for_role("scene_broll") == "scene_broll_video_model"
    assert rs.slot_for_role("background_abstract") == "background_abstract_video_model"
    assert rs.slot_for_role("announcer_visual") == "announcer_video_model"
    assert rs.slot_for_role("music_visual") == "music_video_model"
    # unknown role tolerantly maps to the legacy slot (never raises)
    assert rs.slot_for_role("not_a_role") == rs.LEGACY_OTHER_BEATS_SLOT


def test_engine_id_for_role_reads_per_role_slot_first():
    vm = {
        "character_video_model": {"engine_id": "humo_14B_169"},
        "scene_broll_video_model": {"engine_id": "wan_ti2v"},
        "background_abstract_video_model": {"engine_id": "ltx_video"},
        "other_beats_video_model": {"engine_id": "humo_1.7B"},
    }
    assert rs.engine_id_for_role(vm, "character_video") == "humo_14B_169"
    assert rs.engine_id_for_role(vm, "scene_broll") == "wan_ti2v"
    assert rs.engine_id_for_role(vm, "background_abstract") == "ltx_video"


def test_engine_id_for_role_legacy_fallback():
    # An old policy with ONLY the legacy slot -> every other-beats role falls back.
    vm = {"other_beats_video_model": {"engine_id": "humo_1.7B"}}
    for role in ("character_video", "scene_broll", "background_abstract"):
        assert rs.engine_id_for_role(vm, role) == "humo_1.7B"
    # an empty per-role slot ALSO falls back to the legacy pick
    vm2 = {"character_video_model": {"engine_id": ""},
           "other_beats_video_model": {"engine_id": "humo_1.7B"}}
    assert rs.engine_id_for_role(vm2, "character_video") == "humo_1.7B"


def test_engine_id_for_role_accepts_bare_string_values():
    vm = {"scene_broll_video_model": "wan_ti2v"}
    assert rs.engine_id_for_role(vm, "scene_broll") == "wan_ti2v"


def test_video_slot_roles_has_per_role_and_legacy_slots():
    sr = rs.VIDEO_SLOT_ROLES
    assert sr["character_video_model"] == ("character_video",)
    assert sr["scene_broll_video_model"] == ("scene_broll",)
    assert sr["background_abstract_video_model"] == ("background_abstract",)
    assert set(sr[rs.LEGACY_OTHER_BEATS_SLOT]) == {
        "character_video", "scene_broll", "background_abstract"}


# --------------------------------------------------------------------------- #
# wrapper_bridge.fit_frames_to_target -- exact-fit (trim / mirror-extend)
# --------------------------------------------------------------------------- #
def _frames(n):
    return np.arange(n * 2 * 2 * 3, dtype=np.uint8).reshape(n, 2, 2, 3)


def test_fit_frames_trims_when_over():
    out = wb.fit_frames_to_target(_frames(49), 30)
    assert out.shape[0] == 30
    assert np.array_equal(out, _frames(49)[:30])  # trim keeps the leading frames


def test_fit_frames_extends_when_short():
    out = wb.fit_frames_to_target(_frames(49), 120)
    assert out.shape[0] == 120                     # mirror-extended to the target


def test_fit_frames_identity_on_match_and_empty():
    assert wb.fit_frames_to_target(_frames(40), 40).shape[0] == 40
    assert wb.fit_frames_to_target(_frames(0), 40).shape[0] == 0
    assert wb.fit_frames_to_target(_frames(40), 0).shape[0] == 40  # target<=0 -> no-op


# --------------------------------------------------------------------------- #
# HuMo-14B frame cap -- class override ONLY on the 14B tier
# --------------------------------------------------------------------------- #
def test_only_humo_14b_169_is_frame_capped():
    assert eng_humo.HuMo14BLandscapeEngine.safe_render_frames == \
        eng_humo._HUMO_14B_SAFE_RENDER_FRAMES
    # base + the 1.7B tiers stay uncapped (None -> legacy 177 ceiling)
    assert eng_humo.HuMoEngine.safe_render_frames is None
    assert eng_humo.HuMo17BEngine.safe_render_frames is None
    assert eng_humo.HuMo17BLandscapeEngine.safe_render_frames is None


# --------------------------------------------------------------------------- #
# End-to-end routing: OTR_VideoDirector -> policy -> role_compat + dispatcher
# --------------------------------------------------------------------------- #
def _direct_policy():
    pol_json, = vd.OTRVideoDirector().direct(
        announcer_video_model="ltx_audio_in (16:9)",
        music_video_model="ltx_audio_in (16:9)",
        other_beats_video_model="humo_1.7B",
        announcer_image_model="flux_gen1",
        music_image_model="flux_gen1",
        other_beats_image_model="flux_gen1",
        other_beats_clip_mode="pool_n_loop",
        other_beats_n=4, fps=25, canvas_w=832, canvas_h=480,
        seed_mode="request_hash", request_seed=42, allow_auto_fallback=True,
        character_video_model="humo_14B_169 (16:9)",
        scene_broll_video_model="wan_ti2v",
        background_abstract_video_model="ltx_video",
    )
    return json.loads(pol_json)


def test_director_emits_per_role_video_models():
    vm = _direct_policy()["video_models"]
    assert vm["character_video_model"]["engine_id"] == "humo_14B_169"
    assert vm["scene_broll_video_model"]["engine_id"] == "wan_ti2v"
    assert vm["background_abstract_video_model"]["engine_id"] == "ltx_video"


def test_every_role_engine_fits_its_role():
    vm = _direct_policy()["video_models"]
    for role in ("announcer_visual", "music_visual", "character_video",
                 "scene_broll", "background_abstract"):
        eid = rs.engine_id_for_role(vm, role)
        eng = vreg.get_engine(eid)
        desc = {"engine_id": eid, "roles": tuple(getattr(eng, "roles", ())),
                "required_inputs": tuple(getattr(eng, "required_inputs", ()))}
        assert rc.engine_fits_role(desc, role), (role, eid)


def test_policy_aspects_has_per_role_entries():
    aspects = _direct_policy()["aspects"]
    assert set(aspects) == {"announcer_visual", "music_visual", "character_video",
                            "scene_broll", "background_abstract"}
    # humo_14B_169 is the wide tier -> the character still is minted 16:9.
    assert aspects["character_video"] == "wide"


def test_dispatcher_keeps_character_still_for_humo_14b():
    # The image dispatcher must NOT skip the character still: humo_14B_169 needs
    # an init_image, so _still_needed_for_role is True on character_video.
    pol = _direct_policy()
    assert disp._still_needed_for_role(pol, "character_video") is True
    # background_abstract routes to ltx_video, which inherits accepts_still=True
    # (MotionEngineBase) -> it CAN use the still (i2v), so one is generated.
    assert disp._still_needed_for_role(pol, "background_abstract") is True
    # a role routed to an accepts_still=False engine (viz_green) skips the still.
    pol_vis = {"video_models": {"music_video_model": {"engine_id": "viz_green"}}}
    assert disp._still_needed_for_role(pol_vis, "music_visual") is False
    # unknown role stays fail-safe (keep the still)
    assert disp._still_needed_for_role(pol, "not_a_role") is True
