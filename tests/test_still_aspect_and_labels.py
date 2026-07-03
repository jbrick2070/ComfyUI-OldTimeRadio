"""Still-aspect correctness + obvious HuMo dropdown labels (2026-06-17; VRAM
tier suffix added 2026-06-30, HuMo-improve plan item 5).

Operator directive: every registered video/3D engine must declare an EXPLICIT
``render_aspect`` in {portrait, wide} so the director mints a matching init still
(``ltx_video`` previously declared none -> silently defaulted portrait -> the wide
render decapitated heads), and the per-role dropdown must show an aspect-DERIVED
label (``humo (portrait)`` vs ``humo_1.7B_169 (16:9)``) while the SAVED value stays
the bare engine id (back-compat with old saved graphs). The label now ALSO
appends a VRAM-tier suffix auto-derived from the registry CAPABILITIES table
(``humo_1.7B (portrait) (~6.8GB)``) so the operator sees model + VRAM cost at a
glance without opening the registry.
"""
from __future__ import annotations

from nodes import otr_video_director as vd
from nodes._otr_video_engines import registry as vreg


# The only PORTRAIT still-feed paths (operator rule: "wide UNLESS HuMo-portrait OR
# a requires_mesh_portrait 3D engine"). Everything else renders 16:9 -> wide still.
_PORTRAIT_ENGINES = {
    "humo", "humo_1.7B",                       # HuMo-portrait (the only registered
                                               # portraits; the mesh-portrait 3D
                                               # talkers were unregistered C3)
}


def _engine(name):
    return vreg.get_engine(name)


# --------------------------------------------------------------------------- #
# Goal A -- every registered engine has an explicit render_aspect in {portrait, wide}
# --------------------------------------------------------------------------- #
def test_every_engine_declares_explicit_render_aspect():
    for name in vreg.all_engine_names():
        eng = _engine(name)
        assert hasattr(eng, "render_aspect"), (
            f"engine '{name}' has no explicit render_aspect (would silently "
            f"default -> wrong still aspect)"
        )
        assert eng.render_aspect in ("portrait", "wide"), (
            f"engine '{name}' render_aspect={eng.render_aspect!r} not in "
            f"{{portrait, wide}}"
        )


def test_portrait_vs_wide_partition_matches_operator_rule():
    for name in vreg.all_engine_names():
        eng = _engine(name)
        expected = "portrait" if name in _PORTRAIT_ENGINES else "wide"
        assert eng.render_aspect == expected, (
            f"engine '{name}' render_aspect={eng.render_aspect!r}, expected "
            f"{expected!r} per the wide-unless-(HuMo-portrait|mesh-portrait) rule"
        )


def test_ltx_video_is_wide_the_bug_fix():
    # The reported root cause: ltx_video declared no render_aspect -> portrait
    # default -> 832x1216 portrait stills the 16:9 render decapitated.
    assert _engine("ltx_video").render_aspect == "wide"


def test_mesh_portrait_3d_engines_are_portrait():
    # The 3D talkers were UNREGISTERED 2026-06-29 (C3); their SOURCE classes still
    # declare requires_mesh_portrait + render_aspect=portrait (they feed the mesher
    # a PORTRAIT still even though the final video is 16:9). Checked directly since
    # they are no longer in the registry.
    from nodes._otr_video_engines.eng_character_3d import (
        Hunyuan3DTalkEngine, TripoSGTalkEngine, TrellisTalkEngine,
    )
    for cls in (TripoSGTalkEngine, Hunyuan3DTalkEngine, TrellisTalkEngine):
        eng = cls()
        assert getattr(eng, "requires_mesh_portrait", False) is True
        assert eng.render_aspect == "portrait"


# --------------------------------------------------------------------------- #
# _role_aspects mapping (VideoDirector -> ImageDirector -> MetaBrief still dims)
# --------------------------------------------------------------------------- #
def _resolved(announcer, music, character):
    return {
        "announcer_video_model": {"engine_id": announcer},
        "music_video_model": {"engine_id": music},
        "character_video_model": {"engine_id": character},
    }


def test_role_aspects_ltx_and_169_humos_are_wide():
    aspects = vd.OTRVideoDirector._role_aspects(
        _resolved("ltx_video", "humo_1.7B_169", "humo_14B_169"))
    assert aspects["announcer_visual"] == "wide"
    assert aspects["music_visual"] == "wide"
    assert aspects["character_video"] == "wide"


def test_role_aspects_portrait_humos_are_portrait():
    aspects = vd.OTRVideoDirector._role_aspects(
        _resolved("humo", "humo", "humo_1.7B"))
    assert aspects["announcer_visual"] == "portrait"
    assert aspects["music_visual"] == "portrait"
    assert aspects["character_video"] == "portrait"


def test_role_aspects_unknown_engine_falls_back_portrait():
    aspects = vd.OTRVideoDirector._role_aspects(
        _resolved("not_an_engine", "", "also_missing"))
    assert aspects["announcer_visual"] == "portrait"
    assert aspects["music_visual"] == "portrait"
    assert aspects["character_video"] == "portrait"


# --------------------------------------------------------------------------- #
# Goal B -- label <-> id round-trip (display-only, value-safe)
# --------------------------------------------------------------------------- #
def test_label_suffix_is_aspect_derived():
    assert vd._label_for("humo") == "humo (portrait) (~13.7GB)"
    assert vd._label_for("humo_1.7B") == "humo_1.7B (portrait) (~6.8GB)"
    assert vd._label_for("humo_1.7B_169") == "humo_1.7B_169 (16:9) (~6.8GB)"
    assert vd._label_for("ltx_video") == "ltx_video (16:9) (~13.7GB)"


def test_label_vram_tier_suffix_is_registry_derived():
    # 2026-06-30 item 5: the VRAM-tier suffix is COMPUTED from CAPABILITIES'
    # vram_estimate_mb, never a hand-maintained per-engine string.
    assert vreg.vram_tier_label("humo_1.7B") == " (~6.8GB)"
    assert vreg.vram_tier_label("not_a_real_engine") == ""     # unknown -> none


def test_label_zero_vram_engine_gets_no_vram_suffix():
    # A CPU-tier engine (vram_estimate_mb=0) needs no VRAM caveat -- the bare
    # label (plus any aspect suffix) round-trips unchanged, preserving the
    # back-compat contract for old saved graphs.
    assert vreg.vram_tier_label("still_flat") == ""
    assert vreg.CAPABILITIES["still_flat"]["vram_estimate_mb"] == 0


def test_label_descriptor_suffix_is_family_derived():
    # 2026-07-01 E4: an audio-reactive visualizer (family "abstract" that mints
    # NO scene image) carries a plain-language caveat so a music/announcer/scene
    # pick of it is obviously scene-still-ignoring. DERIVED from family +
    # accepts_still, never a hand-maintained per-engine string.
    for name in ("viz_green", "viz_mxc_cpu", "viz_mxc_mandala"):
        assert vd._descriptor_suffix(name) == " (audio-reactive, no scene image)"
        assert vd._label_for(name).endswith(" (audio-reactive, no scene image)")
    # a non-abstract / still-consuming engine gets NO descriptor
    for name in ("humo", "ltx_video", "still_flat", "wan_i2v"):
        assert vd._descriptor_suffix(name) == ""
    # unknown engine -> '' (never raises)
    assert vd._descriptor_suffix("not_a_real_engine") == ""


def test_descriptor_label_still_round_trips_to_bare_id():
    # The descriptor suffix starts with ' (' so it strips with the aspect/VRAM
    # suffixes -> the saved value stays the bare engine id.
    for name in ("viz_green", "viz_mxc_cpu", "viz_mxc_mandala"):
        assert vd._engine_id_from_pick(vd._label_for(name)) == name


def test_label_round_trips_to_bare_id():
    for name in vreg.all_engine_names():
        assert vd._engine_id_from_pick(vd._label_for(name)) == name


def test_bare_legacy_value_still_resolves():
    # Old saved graphs store the bare id -> must pass through unchanged.
    assert vd._engine_id_from_pick("humo") == "humo"
    assert vd._engine_id_from_pick("ltx_audio_in") == "ltx_audio_in"


def test_renamed_engine_legacy_alias_resolves():
    # Renamed 2026-06-29: a saved graph/ledger that still picks the OLD id resolves
    # to the current engine (flat_still -> still_flat, flux_still -> still_pan; the
    # misleading "flux" name was dropped -- the engine is ffmpeg, never Flux).
    # still_kenburns -> still_motion (the always-renders radio floor was renamed).
    assert vd._engine_id_from_pick("flat_still") == "still_flat"
    assert vd._engine_id_from_pick("flux_still") == "still_pan"
    assert vd._engine_id_from_pick("still_kenburns") == "still_motion"
    # ...and the current ids are real registered engines (the menu).
    assert "still_flat" in vreg.all_engine_names()
    assert "still_pan" in vreg.all_engine_names()
    assert "still_motion" in vreg.all_engine_names()


def test_add_custom_sentinel_is_preserved():
    assert vd._engine_id_from_pick(vd.ADD_CUSTOM) == vd.ADD_CUSTOM


def test_combo_entries_all_parse_to_registered_ids():
    combo = vd._video_model_combo()
    parsed = {vd._engine_id_from_pick(c) for c in combo} - {vd.ADD_CUSTOM}
    assert parsed == set(vreg.all_engine_names())   # registry IS the menu (C4)


def test_direct_parses_labelled_pick_to_bare_engine_id():
    """End-to-end: a labelled pick produces a policy whose video_models carry the
    BARE engine id, and aspects follow the engine's render_aspect."""
    import json
    out = vd.OTRVideoDirector().direct(
        announcer_video_model="ltx_video (16:9)",
        music_video_model="humo_1.7B_169 (16:9)",
        character_video_model="humo_1.7B (portrait)",
        announcer_image_model="flux_gen1",
        music_image_model="flux_gen1",
        other_beats_image_model="flux_gen1",
        fps=25, canvas_w=832, canvas_h=480,
        seed_mode="request_hash", request_seed=42,
    )
    policy = json.loads(out[0])
    assert policy["video_models"]["announcer_video_model"]["engine_id"] == "ltx_video"
    assert policy["video_models"]["music_video_model"]["engine_id"] == "humo_1.7B_169"
    assert policy["video_models"]["character_video_model"]["engine_id"] == "humo_1.7B"
    assert policy["aspects"]["announcer_visual"] == "wide"
    assert policy["aspects"]["music_visual"] == "wide"
    assert policy["aspects"]["character_video"] == "portrait"
