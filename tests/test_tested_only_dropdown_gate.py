"""Dropdown == registry (C4 -- "registry IS the menu", 2026-06-29).

The TESTED-ONLY dropdown gate (VALIDATED_ENGINES + validated_engine_names) was
DELETED 2026-06-29 (C4): there is no validated-subset filter. The
OTR_VideoDirector / OTR_ImageDirector per-role COMBOs now list EVERY REGISTERED
engine + the "+ Add Custom Model" sentinel -- a registered engine is selectable
and renders (it may hard-fail LOUD; validation is the operator's MANUAL process,
never a code gate). The bare engine id is still the SAVED value (the video combo
shows aspect-derived labels for display only).
"""
from __future__ import annotations

from nodes import otr_image_director as idr
from nodes import otr_video_director as vd
from nodes._otr_image_engines import registry as ireg
from nodes._otr_video_engines import registry as vreg


def test_video_combo_is_the_full_registry_plus_sentinel():
    combo = vd._video_model_combo()
    assert combo[-1] == vd.ADD_CUSTOM
    # entries are aspect-LABELLED (e.g. "humo (portrait)"); parse each back to the
    # bare engine id before comparing to the FULL registry.
    listed = {vd._engine_id_from_pick(c) for c in combo} - {vd.ADD_CUSTOM}
    assert listed == set(vreg.all_engine_names())


def test_image_combo_is_the_full_registry_plus_sentinel():
    combo = idr._image_model_combo()
    assert combo[-1] == idr.ADD_CUSTOM
    listed = set(combo) - {idr.ADD_CUSTOM}
    assert listed == set(ireg.all_engine_names())


def test_no_validated_subset_filter_remains():
    # C4: the validated-subset filter is GONE from BOTH registries.
    assert not hasattr(vreg, "VALIDATED_ENGINES")
    assert not hasattr(vreg, "validated_engine_names")
    assert not hasattr(ireg, "VALIDATED_ENGINES")
    assert not hasattr(ireg, "validated_engine_names")


def test_formerly_hidden_engines_are_now_selectable():
    # Engines the old gate HID (untested-but-registered) are now in the dropdown --
    # registry IS the menu. (still_parallax UNREGISTERED 2026-06-30, item 2
    # rip-out -- no longer expected here.)
    video = {vd._engine_id_from_pick(c) for c in vd._video_model_combo()}
    assert {"wan_i2v", "still_motion"} <= video
    assert "still_parallax" not in video
    image = set(idr._image_model_combo())
    assert "z_image_turbo" in image


def test_registry_unchanged_v6_intact():
    """V-6: the registry is the single source -- the dropdown no longer narrows it."""
    assert set(vd._engine_id_from_pick(c) for c in vd._video_model_combo()
               if c != vd.ADD_CUSTOM) == set(vreg.all_engine_names())
    assert (set(idr._image_model_combo()) - {idr.ADD_CUSTOM}) == set(
        ireg.all_engine_names())
