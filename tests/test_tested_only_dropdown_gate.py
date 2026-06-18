"""Tested-only dropdown gate (2026-06-17 operator directive).

The OTR_VideoDirector / OTR_ImageDirector per-role COMBOs must list ONLY engines
that have been GPU-validated end-to-end -- untested engines (3D talkers,
wan_i2v/wan_ti2v until tested, the abstract / CPU-floor families, untested image
peers) are HIDDEN from the dropdown. This is a DISPLAY gate only: every engine
stays REGISTERED so role_compat / assert_usable / the force-map experiment knob
still see the full set (invariant V-6).
"""
from __future__ import annotations

from nodes import otr_video_director as vd
from nodes import otr_image_director as idr
from nodes._otr_video_engines import registry as vreg
from nodes._otr_image_engines import registry as ireg


# Engines that exist in the registry but are NOT validated -> must be HIDDEN.
_HIDDEN_VIDEO = {
    "abstract", "flux_still", "hunyuan3d_talk", "mesh_stage", "station_card",
    "still_kenburns", "still_parallax", "trellis_talk", "triposg_talk",
    "triposr", "visualizer", "wan_i2v", "wan_ti2v",
}
_HIDDEN_IMAGE = {
    "chroma_hd", "flux2_klein", "hidream_i1", "lumina_image", "qwen_image",
    "sd35_large",
}


def test_video_validated_set_is_the_seven():
    assert vreg.VALIDATED_ENGINES == frozenset({
        "ltx_video", "ltx_av_music", "ltx_av_talk",
        "humo", "humo_1.7B", "humo_1.7B_169", "humo_14B_169",
    })


def test_image_validated_set():
    # flux_gen1 (default) + z_image_turbo (GPU-validated low-VRAM option 2026-06-18).
    assert ireg.VALIDATED_ENGINES == frozenset({"flux_gen1", "z_image_turbo"})


def test_video_combo_lists_only_validated_plus_sentinel():
    combo = vd._video_model_combo()
    assert combo[-1] == vd.ADD_CUSTOM
    listed = set(combo) - {vd.ADD_CUSTOM}
    assert listed == set(vreg.VALIDATED_ENGINES)
    # the untested engines are HIDDEN from the dropdown
    assert listed.isdisjoint(_HIDDEN_VIDEO)


def test_image_combo_lists_only_validated_plus_sentinel():
    combo = idr._image_model_combo()
    assert combo[-1] == idr.ADD_CUSTOM
    listed = set(combo) - {idr.ADD_CUSTOM}
    assert listed == set(ireg.VALIDATED_ENGINES)
    assert listed.isdisjoint(_HIDDEN_IMAGE)


def test_registry_still_full_v6_intact():
    """V-6: hidden engines remain REGISTERED (only the DISPLAY list narrows)."""
    all_video = set(vreg.all_engine_names())
    all_image = set(ireg.all_engine_names())
    # every hidden engine is still in the registry...
    assert _HIDDEN_VIDEO <= all_video
    assert _HIDDEN_IMAGE <= all_image
    # ...and so are the validated ones.
    assert set(vreg.VALIDATED_ENGINES) <= all_video
    assert set(ireg.VALIDATED_ENGINES) <= all_image


def test_validated_names_are_intersection_with_registry():
    assert set(vreg.validated_engine_names()) == (
        set(vreg.all_engine_names()) & vreg.VALIDATED_ENGINES
    )
    assert set(ireg.validated_engine_names()) == (
        set(ireg.all_engine_names()) & ireg.VALIDATED_ENGINES
    )
