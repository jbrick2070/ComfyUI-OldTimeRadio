"""Video-tiers public-name resolver + boundary tests (2026-07-20).

Covers the C2 contract: the `_otr_shared.public_engines` resolver, the menu relabel
+ round-trip, `exact_menu_option_for`, the applier / forced-engine / cross-validate
boundaries resolving public+legacy ids, the ShotLock-internal-only invariant, and the
extended downloader integrity check. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

import pytest

from nodes import otr_video_director as vd
from nodes._otr_shared import public_engines as pub
from nodes._otr_shared.public_engines import resolve_engine_id
from nodes._otr_video_engines import registry as vreg
from nodes._otr_workflow_apply import (
    _is_engine_director_admissible, apply_profile, build_offline_schemas)

_TIER = {
    "ltx_8gb": "ltx_8gb",
    "wan_8gb": "wan_ti2v",
    "ltx23_16gb_audio_in": "ltx_audio_in",
    "ltx23_16gb_video": "ltx_video",
}


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #
def test_public_engines_bijection():
    assert len(pub._PUBLIC_ENGINES) == len(pub._INTERNAL_TO_PUBLIC)
    assert pub._PUBLIC_ENGINES == _TIER
    # every public label maps back to the internal it was built from
    for public, internal in _TIER.items():
        assert pub._INTERNAL_TO_PUBLIC[internal] == public


@pytest.mark.parametrize("public,internal", list(_TIER.items()))
def test_resolve_public_id_and_suffixed_label(public, internal):
    assert resolve_engine_id(public) == internal
    assert resolve_engine_id(public + " (16:9)") == internal


def test_resolve_legacy_aliases():
    assert resolve_engine_id("flat_still") == "still_flat"
    assert resolve_engine_id("flux_still") == "still_pan"
    assert resolve_engine_id("still_kenburns") == "still_motion"
    assert resolve_engine_id("visualizer") == "viz_green"
    # legacy id even with a display suffix
    assert resolve_engine_id("visualizer (16:9)") == "viz_green"


def test_resolve_bare_internal_and_sentinel_passthrough():
    for name in vreg.all_engine_names():
        assert resolve_engine_id(name) == name          # idempotent
    assert resolve_engine_id(vd.ADD_CUSTOM) == vd.ADD_CUSTOM
    assert resolve_engine_id("") == ""
    assert resolve_engine_id(None) == ""
    assert resolve_engine_id("not_a_real_engine") == "not_a_real_engine"


def test_resolve_all_menu_labels_round_trip_to_registered_internals():
    combo = vd._video_model_combo()
    parsed = {resolve_engine_id(c) for c in combo} - {vd.ADD_CUSTOM}
    assert parsed == set(vreg.all_engine_names())


# --------------------------------------------------------------------------- #
# menu relabel + exact_menu_option_for
# --------------------------------------------------------------------------- #
def test_menu_shows_public_ids_uniquely():
    combo = vd._video_model_combo()
    for public in _TIER:
        assert ("%s (16:9)" % public) in combo
        assert combo.count("%s (16:9)" % public) == 1
    # the renamed internal ids never leak into the menu
    assert "wan_ti2v (16:9)" not in combo
    assert "ltx_video (16:9)" not in combo
    assert "ltx_audio_in (16:9)" not in combo
    assert vd.ADD_CUSTOM in combo
    assert len(combo) == len(set(combo))               # no duplicates


@pytest.mark.parametrize("public,internal", list(_TIER.items()))
def test_exact_menu_option_for_tier(public, internal):
    assert vd.exact_menu_option_for(internal) == "%s (16:9)" % public
    # and it round-trips
    assert resolve_engine_id(vd.exact_menu_option_for(internal)) == internal


def test_exact_menu_option_for_non_tier_and_missing():
    assert vd.exact_menu_option_for("humo") == vd._label_for("humo")
    with pytest.raises(ValueError):
        vd.exact_menu_option_for("engine_that_is_not_registered")


# --------------------------------------------------------------------------- #
# boundary: applier admissibility
# --------------------------------------------------------------------------- #
def test_director_admissible_accepts_public_legacy_and_bare():
    ok = ("wan_8gb (16:9)", "ltx_8gb (16:9)", "ltx23_16gb_video (16:9)",
          "wan_ti2v", "visualizer", "humo (portrait)")
    for value in ok:
        assert _is_engine_director_admissible("announcer_video_model", value)
    assert not _is_engine_director_admissible("music_video_model", "not_an_engine")
    assert _is_engine_director_admissible("music_video_model", vd.ADD_CUSTOM)


# --------------------------------------------------------------------------- #
# boundary: profile apply writes the public option; ShotLock stays internal-only
# --------------------------------------------------------------------------- #
def test_profile_apply_writes_public_menu_option():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    canonical = json.load(open(os.path.join(root, "workflows", "otr_canonical.json"),
                               encoding="utf-8"))
    schemas = build_offline_schemas()
    applied = apply_profile(canonical, "otr_amd16_rocm", schemas=schemas)
    director = [n for n in applied["nodes"]
                if n.get("type") == "OTR_VideoDirector"][0]
    wv = director.get("widgets_values") or []
    assert "wan_8gb (16:9)" in wv          # the public menu option, not the internal
    assert "wan_ti2v" not in wv


def test_director_direct_resolves_public_pick_to_internal_engine_id():
    """A PUBLIC menu pick flows through direct() to the INTERNAL engine id in the
    policy (the ShotLock-internal-only invariant: downstream never sees the public
    label)."""
    out = vd.OTRVideoDirector().direct(
        announcer_video_model="wan_8gb (16:9)",
        music_video_model="ltx23_16gb_video (16:9)",
        character_video_model="ltx_8gb (16:9)",
        announcer_image_model="flux_gen1",
        music_image_model="flux_gen1",
        character_image_model="flux_gen1",
        fps=25, canvas_w=832, canvas_h=480,
        seed_mode="request_hash", request_seed=7,
    )
    policy = json.loads(out[0])
    vm = policy["video_models"]
    assert vm["announcer_video_model"]["engine_id"] == "wan_ti2v"
    assert vm["music_video_model"]["engine_id"] == "ltx_video"
    assert vm["character_video_model"]["engine_id"] == "ltx_8gb"
    # NO public label survives anywhere in the emitted policy string
    for public in _TIER:
        assert public not in out[0] or public == "ltx_8gb"   # ltx_8gb IS internal


# --------------------------------------------------------------------------- #
# boundary: forced-engine override resolves public/legacy -> internal
# --------------------------------------------------------------------------- #
def test_parse_engine_override_resolves_public_and_legacy():
    from nodes._otr_video_engines.render_driver import parse_engine_override
    assert parse_engine_override("*=wan_8gb") == {"*": "wan_ti2v"}
    assert parse_engine_override("music_visual=ltx23_16gb_video") == {
        "music_visual": "ltx_video"}
    assert parse_engine_override("character_video=visualizer") == {
        "character_video": "viz_green"}
    with pytest.raises(ValueError):
        parse_engine_override("*=not_a_real_engine")


# --------------------------------------------------------------------------- #
# boundary: capability cross-validation resolves a public id
# --------------------------------------------------------------------------- #
def test_cross_validate_resolves_public_id():
    from nodes._otr_shared import capability_profiles as cp
    from nodes._otr_audio_engines import registry as areg
    from nodes._otr_image_engines import registry as ireg
    decls = {"video": vreg.CAPABILITIES, "audio": areg.CAPABILITIES,
             "image": ireg.CAPABILITIES}
    mapping = cp.load_widget_mapping()
    prof = dict(cp.load_profile("otr_amd16_rocm"))
    # baseline: the committed profile already cross-validates (carries wan_ti2v).
    cp.cross_validate_profile(prof, mapping, decls)
    # swap every wan_ti2v override to the PUBLIC id; it must STILL validate
    # (cross_validate resolves wan_8gb -> wan_ti2v, the enabled internal engine).
    swapped = 0
    for section in ("role_overrides", "slot_overrides"):
        sec = dict(prof.get(section) or {})
        for k, v in list(sec.items()):
            if v == "wan_ti2v":
                sec[k] = "wan_8gb"
                swapped += 1
        prof[section] = sec
    assert swapped >= 1, "otr_amd16_rocm should carry a wan_ti2v override"
    cp.cross_validate_profile(prof, mapping, decls)   # public id resolves -> OK


# --------------------------------------------------------------------------- #
# downloader integrity (hf_download_driver extension)
# --------------------------------------------------------------------------- #
def test_download_driver_verify_materialized(tmp_path):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if os.path.join(root, "scripts") not in sys.path:
        sys.path.insert(0, os.path.join(root, "scripts"))
    from hf_download_driver import _verify_materialized
    f = tmp_path / "w.bin"
    f.write_bytes(b"hello-weights")
    good_sha = hashlib.sha256(b"hello-weights").hexdigest()
    # correct size + sha -> no error
    assert _verify_materialized(str(f), len(b"hello-weights"), good_sha) == ""
    # wrong size -> error
    assert "size mismatch" in _verify_materialized(str(f), 999, None)
    # wrong sha -> error
    assert "SHA-256 mismatch" in _verify_materialized(str(f), None, "deadbeef")
    # missing file -> error
    assert _verify_materialized(str(tmp_path / "nope.bin"), 1, None) != ""
    # no checks requested -> passes
    assert _verify_materialized(str(f), None, None) == ""


# --------------------------------------------------------------------------- #
# cold-import: the resolver pulls in nothing heavy
# --------------------------------------------------------------------------- #
def test_public_engines_is_cold_import_clean():
    import importlib
    mod = importlib.import_module("nodes._otr_shared.public_engines")
    assert callable(mod.resolve_engine_id)
    # the module itself must not import torch/numpy/transformers at module scope
    src = open(mod.__file__, encoding="utf-8").read()
    for heavy in ("import torch", "import numpy", "import transformers",
                  "from torch", "from numpy"):
        assert heavy not in src
