"""C5 (2026-06-30; 3-role rewrite 2026-07-01 rip-sfx-broll) -- OFFLINE proof of
the all-slots canonical-JSON soak.

The live GPU soak is gated, but its load-bearing wiring is proven here on the CPU:

  * the REAL workflow (workflows/otr_scifi_16gb_full.json) is loaded and ALL
    THREE video slots are set INDEPENDENTLY via the capability-profile
    role_overrides -> apply_profile patches the OTR_VideoDirector video-model
    widgets BY NODE TYPE (config/profiles/widget_mapping.json; no node ids),
    and NONE is left at the legacy "(use Other Beats default)" sentinel;
  * every chosen engine is role_compat-ELIGIBLE for its slot (capability, C2/C4);
  * the per-beat CONTENT ORACLE catches the D2 dark floor + a frozen motion clip,
    and exempts static engines -- proven on real ffmpeg-rendered fixtures.

CPU only; no server, no model load (apply_profile uses build_offline_schemas).
UTF-8, no BOM, SFW.
"""
from __future__ import annotations

import json
import os
import pathlib
import shutil
import subprocess

import pytest

import nodes._otr_video_engines  # noqa: F401  (self-registers every engine)
from nodes import _otr_workflow_apply as wa
from nodes._otr_shared import role_compat as rc
from nodes._otr_shared import slot_matrix as sm
from nodes._otr_shared import content_oracle as co
from nodes._otr_shared.capability_profiles import load_profile
from nodes._otr_video_engines import registry as vreg

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CANONICAL = REPO_ROOT / "workflows" / "otr_scifi_16gb_full.json"
_HAS_FFMPEG = shutil.which("ffmpeg") is not None

#: distinct, capability-eligible engine per slot.
ROLE_ENGINES = {
    "announcer_visual": "humo_1.7B",
    "music_visual": "viz_green",
    "character_video": "humo_1.7B_169",
}
_OTHER_BEATS_SENTINEL = "(use Other Beats default)"
_VIDEO_WIDGET = {
    "announcer_visual": "announcer_video_model",
    "music_visual": "music_video_model",
    "character_video": "character_video_model",
}


def _load_canonical() -> dict:
    with open(CANONICAL, "r", encoding="utf-8") as f:
        return json.load(f)


# --------------------------------------------------------------------------- #
# all-slots applied to the REAL workflow JSON
# --------------------------------------------------------------------------- #
def test_all_slots_set_on_canonical_json():
    base = load_profile("16gb_full")
    profile = sm.build_all_role_profile(base, ROLE_ENGINES)
    # exactly the three video + three image role keys -- a closed set (no legacy
    # catch-all / b-roll / background keys leak in)
    assert set(profile["role_overrides"]) == {
        "announcer_visual", "music_visual", "character_visual",
        "announcer_image", "music_image", "character_image",
    }
    wf = wa.apply_profile(_load_canonical(), profile)
    schemas = wa.build_offline_schemas()
    vd = [n for n in wf["nodes"] if n.get("type") == "OTR_VideoDirector"]
    assert len(vd) == 1
    slots = wa.serialized_slot_names("OTR_VideoDirector", schemas)
    vals = vd[0]["widgets_values"]
    for role, engine in ROLE_ENGINES.items():
        widget = _VIDEO_WIDGET[role]
        got = vals[slots.index(widget)]
        assert got == engine, (role, widget, got)
        assert got != _OTHER_BEATS_SENTINEL, role          # no legacy fallback


def test_image_keys_are_exactly_the_three():
    # 2026-07-04 rename: the third image role key is character_image (guards the
    # silent-drop path in build_all_role_profile).
    assert sm.IMAGE_KEYS == ("announcer_image", "music_image", "character_image")


def test_canonical_json_character_granularity_widget():
    # node 88 (OTR_ImageDirector) granularity widget renamed to character_granularity:
    # all three name fields agree, the value stays positional, and the code side matches.
    import inspect
    wf = _load_canonical()
    n88 = next(n for n in wf["nodes"] if n.get("id") == 88)
    assert n88.get("type") == "OTR_ImageDirector"
    inp = next(i for i in n88["inputs"] if i.get("name") == "character_granularity")
    assert inp["localized_name"] == "character_granularity"
    assert inp["widget"]["name"] == "character_granularity"
    assert n88["widgets_values"][2] == "per_object"          # positional value unchanged
    from nodes.otr_image_director import OTRImageDirector
    assert "character_granularity" in OTRImageDirector.INPUT_TYPES()["required"]
    sig = inspect.signature(OTRImageDirector().direct)
    assert "character_granularity" in sig.parameters


def test_character_image_override_flows_and_stale_key_is_dropped():
    # silent-drop guard (build_all_role_profile loop over IMAGE_KEYS): a character_image
    # override must LAND; an unknown/stale image key is ignored and the slot falls to
    # the baseline -- why the rename must move producers + IMAGE_KEYS together.
    base = load_profile("16gb_full")
    ok = sm.build_all_role_profile(base, {}, image_engines={"character_image": "z_image_turbo"})
    assert ok["role_overrides"]["character_image"] == "z_image_turbo"
    stale = sm.build_all_role_profile(base, {}, image_engines={"legacy_image_key": "z_image_turbo"})
    assert stale["role_overrides"]["character_image"] == sm.DEFAULT_IMAGE_BASELINE
    assert "legacy_image_key" not in stale["role_overrides"]


def test_every_chosen_engine_is_capability_eligible():
    for role, engine in ROLE_ENGINES.items():
        desc = vreg.descriptor_for_engine(engine)
        assert rc.engine_fits_role(desc, role), (engine, role)


def test_builder_fills_missing_roles_with_baseline():
    base = load_profile("16gb_full")
    profile = sm.build_all_role_profile(base, {})   # nothing specified
    ro = profile["role_overrides"]
    for role in sm.ALL_ROLES:
        assert ro[sm.ROLE_TO_PROFILE_KEY[role]] == sm.DEFAULT_VIDEO_BASELINE
    # the baseline still fits every slot by capability (so an unset matrix is valid)
    desc = vreg.descriptor_for_engine(sm.DEFAULT_VIDEO_BASELINE)
    for role in sm.ALL_ROLES:
        assert rc.engine_fits_role(desc, role), role


def test_eligibility_matrix_is_capability_grounded():
    matrix = sm.build_eligibility_matrix()
    assert set(matrix) == set(sm.ALL_ROLES)
    for role, names in matrix.items():
        for name in names:
            assert rc.engine_fits_role(vreg.descriptor_for_engine(name), role)
        # at least the universal still floor is eligible everywhere
        assert sm.DEFAULT_VIDEO_BASELINE in names, role


# --------------------------------------------------------------------------- #
# content oracle on REAL ffmpeg fixtures
# --------------------------------------------------------------------------- #
def _render(path, src, *, dur=1.0):
    # lavfi option separator: ':' when the source already opened options with '='
    # (e.g. "color=c=black"), else '=' (e.g. "testsrc2").
    sep = ":" if "=" in src else "="
    spec = f"{src}{sep}d={dur}:r=25:s=160x120"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", spec,
         "-pix_fmt", "yuv420p", str(path)],
        check=True)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_oracle_flags_dark_floor(tmp_path):
    dark = tmp_path / "dark.mp4"
    _render(dark, "color=c=black")
    with pytest.raises(co.ContentOracleError) as exc:
        co.check_clip(str(dark), "still_flat")           # luma applies to ALL engines
    assert "DARK" in str(exc.value)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_oracle_passes_bright_static_for_static_engine(tmp_path):
    bright = tmp_path / "bright.mp4"
    _render(bright, "color=c=gray")
    v = co.check_clip(str(bright), "still_flat")          # static engine: motion-exempt
    assert v["ok"] and v["motion_required"] is False and v["mean_yavg"] > 16


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_oracle_flags_frozen_clip_for_motion_engine(tmp_path):
    static = tmp_path / "static.mp4"
    _render(static, "color=c=gray")
    with pytest.raises(co.ContentOracleError) as exc:
        co.check_clip(str(static), "ltx_video")           # motion engine + frozen
    assert "FROZEN" in str(exc.value)


@pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not on PATH")
def test_oracle_passes_moving_clip_for_motion_engine(tmp_path):
    moving = tmp_path / "moving.mp4"
    _render(moving, "testsrc2", dur=1.5)
    v = co.check_clip(str(moving), "ltx_video")
    assert v["ok"] and v["motion_required"] is True and v["frozen"] is False


def test_motion_classification_from_registry():
    # genuine moving lanes require motion; stills + viz_green are exempt
    assert co.motion_required_for_engine("ltx_video") is True
    assert co.motion_required_for_engine("humo_1.7B") is True
    assert co.motion_required_for_engine("wan_i2v") is True
    assert co.motion_required_for_engine("still_flat") is False
    assert co.motion_required_for_engine("still_motion") is False
    assert co.motion_required_for_engine("viz_green") is False
