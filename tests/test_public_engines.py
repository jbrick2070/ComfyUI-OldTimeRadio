"""Video-tiers public-name resolver + boundary tests (2026-07-20).

Covers the C2 contract: the `_otr_shared.public_engines` resolver, the menu relabel
+ round-trip, `exact_menu_option_for`, the applier / forced-engine / cross-validate
boundaries resolving public+legacy ids, the ShotLock-internal-only invariant, and the
extended downloader integrity check. UTF-8, no BOM, ASCII-only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys

import pytest

from nodes import otr_video_director as vd
from nodes._otr_shared import capability_profiles as cp
from nodes._otr_shared import public_engines as pub
from nodes._otr_shared.public_engines import resolve_engine_id
from nodes._otr_video_engines import registry as vreg
from nodes._otr_workflow_apply import (
    _is_engine_director_admissible, apply_profile, build_offline_schemas)

_TIER = {
    # The low/high convention (operator ruling 2026-08-09) lands ONE LANE AT A
    # TIME with the video transplant, not as a family sweep -- so this dict
    # grew by one row per lane and the `<vramtier>gb` rows retired as their own
    # lanes closed; the last went on 2026-08-11, so every row below is a
    # low/high id.
    # Lane 1, 2026-08-11: wan_i2v.
    "wan22_high_i2v": "wan_i2v",
    # Lane 2, 2026-08-11: the ruled hero cast. The id states audio_in and the
    # aspect because a bare `humo14_high_face` hid which way its sibling
    # renders.
    "humo14_high_audio_in_wide": "humo_14B_169",
    # Lane 3, 2026-08-11: the LONG-BEAT tier and its landscape twin. Same
    # checkpoint, same VRAM class -- the aspect IS the difference, so it is in
    # the id.
    "humo17_high_audio_in_portrait": "humo_1.7B",
    "humo17_high_audio_in_wide": "humo_1.7B_169",
    # Lane 4, 2026-08-11: the last HuMo tier.
    "humo14_high_audio_in_portrait": "humo",
    # Lane 8, 2026-08-11: an IDENTITY row out, so no alias. `low` is measured
    # here (6,835 MB net, cold, at 512x288x161), not inherited from the `8gb`
    # token it replaces.
    "ltx098_low_video": "ltx_8gb",
    # Lane 19, 2026-08-12: the first ADD rather than a rename or a move -- a new
    # engine, so there is no old public id to retire and no alias row. `low` is
    # measured at H3's legal 124-model / 129-canvas-frame floor: 6,315 MB cold
    # absolute at 864x480 on the 5080. That does not qualify a physical 8 GB
    # card; the public id records recipe cost, not a card guarantee.
    "h3_low_video": "minimax_h3_video",
    # Lane 20, 2026-08-12: the second H3 public id, and it maps to a SEPARATE
    # internal engine. That separation is the whole reason lane 19 registered
    # only one adapter -- two public ids on one internal id collapses
    # _INTERNAL_TO_PUBLIC and trips the bijection assert at IMPORT time.
    "h3_low_audio_in": "minimax_h3_audio_in",
    # LTX 2.5, 2026-08-19: the silent lane, on the 16 GB mix4x8 DiT.
    "ltx25_high_video": "ltx25_video",
}

#: Public rows whose INTERNAL engine is retired -- resolvable, never offerable.
#:
#: The row stays in `_PUBLIC_ENGINES` (so `_TIER` still mirrors it exactly and
#: the bijection tests below are unweakened), but the engine is unregistered, so
#: it contributes NO menu option. Every menu-shaped assertion in this file skips
#: these and then checks the tombstone property instead: the id still resolves,
#: and it cannot be picked.
#:
#: `wan22_high_i2v` -> `wan_i2v` joined this set with the 2026-08-26 large-Wan
#: rip (operator: "rip the large wan we don't need"). The public row is kept
#: deliberately so a stale saved graph reaches the named RetiredEngineError
#: rather than the generic "no engine named ..." message.
_TOMBSTONE_PUBLIC_IDS = frozenset({"wan22_high_i2v"})

#: The rows that still render a live menu option -- what the menu tests iterate.
_LIVE_TIER = {public: internal for public, internal in _TIER.items()
              if public not in _TOMBSTONE_PUBLIC_IDS}


# --------------------------------------------------------------------------- #
# resolver
# --------------------------------------------------------------------------- #
def test_the_naming_convention_rows_state_the_model_they_load():
    """A public id is a claim about the model. `wan22_high_i2v` says Wan 2.2
    because the weight is wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors and
    the frozen recipe is wan22_14b_i2v_single_pass_v1.

    RULED 2026-08-11: `wan21` was one mistyped version number in the spec that
    every downstream document inherited; the naming itself was never in doubt.
    Spec and transplant plan corrected, `wan22_high_i2v` stands, no code moved.
    The retired string keeps a legacy-alias row so a paste from any stale copy
    still resolves.

    RETIRED 2026-08-26 (operator: "rip the large wan we don't need"). Both name
    rows above SURVIVE the rip and that is the whole reason this test still
    exists: they are what routes a stale saved graph into the named
    RetiredEngineError instead of the generic "no engine named ..." message.
    The registry assertion that used to close this test went with the rip --
    `wan_i2v` has no CAPABILITIES row any more, because the engine is
    unregistered. What replaces the dead assertion is the property that
    actually matters now -- the id these rows point at is retired.
    """
    assert pub._PUBLIC_ENGINES["wan22_high_i2v"] == "wan_i2v"
    assert pub._LEGACY_ENGINE_ALIASES["wan21_high_i2v"] == "wan_i2v"
    assert "wan21_high_i2v" not in pub._PUBLIC_ENGINES, (
        "two public ids on one internal id collapses _INTERNAL_TO_PUBLIC and "
        "trips the module-scope bijection assert at IMPORT time")
    assert "wan_i2v" in pub.RETIRED_ENGINE_IDS
    assert "wan_i2v" not in vreg.CAPABILITIES, (
        "the 14B i2v lane was unregistered by the 2026-08-26 rip; a CAPABILITIES "
        "row reappearing means the engine came back without the retirement "
        "being lifted")



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
def _expected_label(public, internal):
    """The menu label for a public row, aspect suffix and all.

    Derived rather than hardcoded as ``(16:9)`` (lane 3, 2026-08-11): every
    public row was landscape until `humo17_high_audio_in_portrait` arrived, and
    a portrait lane's label reads `(portrait)`. Hardcoding the suffix would
    have made the FIRST portrait row in the public table look like a bug.
    """
    return "%s%s%s" % (public, vd._aspect_suffix(internal),
                       vd._descriptor_suffix(internal))


def test_menu_shows_public_ids_uniquely():
    combo = vd._video_model_combo()
    for public, internal in _LIVE_TIER.items():
        label = _expected_label(public, internal)
        assert label in combo
        assert combo.count(label) == 1
    # A TOMBSTONED row is resolvable but never offerable. Asserting its absence
    # by PREFIX, not by one built label: the label builder needs a registered
    # engine to derive an aspect suffix from, and a retired lane has none -- so
    # the check has to be "no option starts with this public id" rather than
    # "this exact string is missing".
    for public in _TOMBSTONE_PUBLIC_IDS:
        assert not any(o.startswith(public) for o in combo), (
            "%s is retired and must not render as a menu option" % public)
    # the renamed internal ids never leak into the menu: a lane that HAS a
    # public id is offered only under it
    for public, internal in _LIVE_TIER.items():
        if internal == public:
            continue
        leaked = [o for o in combo if o.split(" ")[0] == internal]
        assert not leaked, (
            "%s is offered under its internal id %r as well as %s"
            % (leaked, internal, public))
    assert vd.ADD_CUSTOM in combo
    assert len(combo) == len(set(combo))               # no duplicates


@pytest.mark.parametrize("public,internal", list(_LIVE_TIER.items()))
def test_exact_menu_option_for_tier(public, internal):
    assert vd.exact_menu_option_for(internal) == _expected_label(
        public, internal)
    # and it round-trips
    assert resolve_engine_id(vd.exact_menu_option_for(internal)) == internal


@pytest.mark.parametrize("public", sorted(_TOMBSTONE_PUBLIC_IDS))
def test_exact_menu_option_for_a_tombstoned_row_fails_loud(public):
    """A retired lane has no menu option, and asking for one must RAISE.

    `exact_menu_option_for` is what the applier and build_variants call to write
    the exact string the UI would have saved. A retired lane resolves to zero
    live options, and the function's 0-or-many guard is what stops a generated
    profile or variant from being written against an engine that cannot render.
    """
    internal = _TIER[public]
    with pytest.raises(ValueError, match="found 0"):
        vd.exact_menu_option_for(internal)


def test_exact_menu_option_for_non_tier_and_missing():
    assert vd.exact_menu_option_for("humo") == vd._label_for("humo")
    with pytest.raises(ValueError):
        vd.exact_menu_option_for("engine_that_is_not_registered")


# --------------------------------------------------------------------------- #
# boundary: applier admissibility
# --------------------------------------------------------------------------- #
def test_director_admissible_accepts_public_legacy_and_bare():
    ok = ("ltx098_low_video (16:9)", "ltx_8gb (16:9)",
          "ltx25_high_video (16:9)", "ltx25_video", "visualizer",
          "still_kenburns (16:9)", "humo (portrait)")
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

    def _director_values(profile):
        applied = apply_profile(canonical, profile, schemas=schemas)
        director = [n for n in applied["nodes"]
                    if n.get("type") == "OTR_VideoDirector"][0]
        return director.get("widgets_values") or []

    # The LIVE public option, not the internal id.
    wv = _director_values("otr_16gb_video")
    assert _expected_label("ltx25_high_video", "ltx25_video") in wv
    assert "ltx25_video" not in wv
    # And never a LEGACY id: an applier still writing one would be writing a
    # string that resolves but is no longer in the menu the UI offers. No row
    # names one, so this is a real row with one role spelled the old way.
    legacy = copy.deepcopy(cp.load_profile("otr_8gb_still"))
    legacy["role_overrides"]["announcer_visual"] = "still_kenburns"
    wv = _director_values(legacy)
    assert vd.exact_menu_option_for("still_motion") in wv
    assert not any(str(v).startswith("still_kenburns") for v in wv)


def test_director_direct_resolves_public_pick_to_internal_engine_id():
    """A PUBLIC menu pick flows through direct() to the INTERNAL engine id in the
    policy (the ShotLock-internal-only invariant: downstream never sees the public
    label)."""
    out = vd.OTRVideoDirector().direct(
        announcer_video_model="ltx25_high_video (16:9)",
        music_video_model="ltx098_low_video (16:9)",
        character_video_model="ltx_8gb (16:9)",
        announcer_image_model="flux_gen1",
        music_image_model="flux_gen1",
        character_image_model="flux_gen1",
        fps=25, canvas_w=832, canvas_h=480,
    )
    policy = json.loads(out[0])
    vm = policy["video_models"]
    assert vm["announcer_video_model"]["engine_id"] == "ltx25_video"
    assert vm["music_video_model"]["engine_id"] == "ltx_8gb"
    assert vm["character_video_model"]["engine_id"] == "ltx_8gb"
    # `seed_mode`/`request_seed` are gone from direct() (2026-09-13, write-only
    # widgets removed) -- the emitted VIDEO policy has no "seed" key at all.
    assert "seed" not in policy
    # NO public label survives anywhere in the emitted policy string
    for public in _TIER:
        assert public not in out[0], public


# --------------------------------------------------------------------------- #
# boundary: forced-engine override resolves public/legacy -> internal
# --------------------------------------------------------------------------- #
def test_parse_engine_override_resolves_public_and_legacy():
    from nodes._otr_video_engines.render_driver import parse_engine_override
    assert parse_engine_override("*=ltx25_high_video") == {"*": "ltx25_video"}
    assert parse_engine_override("music_visual=still_kenburns") == {
        "music_visual": "still_motion"}
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
    prof = dict(cp.load_profile("otr_8gb_video"))
    # Baseline: this matrix row names its lane by the INTERNAL id ltx_8gb.
    # AMD rows are images-only, so they are not a valid vehicle for this
    # public-video-id boundary.
    cp.cross_validate_profile(prof, mapping, decls)
    # Swap every ltx_8gb override to the LIVE PUBLIC id; it must STILL
    # validate. Legacy alias coverage lives in the dedicated alias boundary
    # below.
    swapped = 0
    for section in ("role_overrides", "slot_overrides"):
        sec = dict(prof.get(section) or {})
        for k, v in list(sec.items()):
            if v == "ltx_8gb":
                sec[k] = "ltx098_low_video"
                swapped += 1
        prof[section] = sec
    assert swapped >= 1, "otr_8gb_video should carry an ltx_8gb override"
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


def test_a_renamed_lane_MOVES_its_old_public_id_and_never_keeps_two():
    """The rename rule that protects the whole node menu (lane 5, 2026-08-11).

    Two public ids on one internal id collapses `_INTERNAL_TO_PUBLIC`, which
    trips the MODULE-SCOPE bijection assert at IMPORT time. Because the
    director and the shared profile/driver modules import this module
    unguarded, and the pack wraps each node import in its own try/except, the
    blast radius is most of OTR silently vanishing from the ComfyUI menu with
    scattered logged exceptions -- not one clean lane failure.

    So a rename MOVES: the old id lands in `_LEGACY_ENGINE_ALIASES`, where it
    still resolves every saved graph, profile and variant that names it, and
    never renders as a second menu option. The moves that wrote this rule
    retired with their lanes, so every row of the alias table is held to it.
    """
    combo = vd._video_model_combo()
    assert pub._LEGACY_ENGINE_ALIASES, "the alias table is empty"
    for old, internal in pub._LEGACY_ENGINE_ALIASES.items():
        assert old not in pub._PUBLIC_ENGINES, (
            "%s must MOVE to the alias table, not stay a second public row"
            % old)
        assert pub._LEGACY_ENGINE_ALIASES[old] == internal
        assert resolve_engine_id(old) == internal
        assert resolve_engine_id(old + " (16:9)") == internal
        assert not any(o.startswith(old) for o in combo), (
            "%s still renders as a live menu option" % old)
