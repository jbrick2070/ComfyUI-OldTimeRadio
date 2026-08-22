"""The Ghost Signal PEER lanes on the official AnimateDiff modules.

THE TEST THAT MATTERS MOST IS `test_each_peer_actually_loads_its_own_module`.
`eng_fastwan_8gb` records the exact defect this guards: a parent whose recipe
accessors read MODULE-LEVEL constants means "a subclass declaring its own recipe
would have SILENTLY rendered with the parent's and stamped its own receipt on
the result". A peer lane that quietly loads `mm-p_0.5.pth` while labelling
itself v3 would make every comparison between them meaningless AND put a false
licence claim on the output -- the two things these lanes exist to fix.

The peers are ADDITIVE (operator 2026-08-22: "a peer lane", "so if it doesnt
work we have our golden ghost untouched"), so the golden lane's own contract is
re-asserted here too.
"""
from __future__ import annotations

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_shared import public_engines as pub
from nodes._otr_video_engines import eng_ghost_signal as gs
from nodes._otr_video_engines import eng_ghost_signal_official as peers
from nodes._otr_video_engines import registry as vreg

GOLDEN = "animatediff15_video"
V3 = "animatediff15_v3_video"
V2 = "animatediff15_v2_video"
ALL_THREE = (GOLDEN, V3, V2)

EXPECTED_MODULE = {
    GOLDEN: "mm-p_0.5.pth",
    V3: "v3_sd15_mm.ckpt",
    V2: "mm_sd_v15_v2.ckpt",
}


# --------------------------------------------------------------------------- #
# THE SEAM
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL_THREE)
def test_each_peer_actually_loads_its_own_module(name):
    """The fastwan defect, guarded. Every read of the module name must go
    through `self`, or a peer renders with the parent's weights."""
    eng = vreg.get_engine(name)
    assert eng.motion_module_name == EXPECTED_MODULE[name]


@pytest.mark.parametrize("name", ALL_THREE)
def test_the_graph_asks_for_that_module_and_not_a_constant(name, monkeypatch):
    """Not the declaration -- the GRAPH. Builds the real sample graph and reads
    the `model_name` the ADE loader would actually receive."""
    import inspect
    src = inspect.getsource(gs.GhostSignalEngine.render_clip)
    assert '"model_name": self.motion_module_name' in src, (
        "the ADE graph reads a module-level constant, so every peer would load "
        "the golden lane's module while stamping its own receipt")
    # And no stale module-level read survives in the methods.
    assert '"model_name": GHOST_MOTION_MODULE_NAME' not in src


@pytest.mark.parametrize("name", ALL_THREE)
def test_each_lane_stamps_a_receipt_naming_its_own_recipe(name):
    eng = vreg.get_engine(name)
    receipt = eng._recipe_receipt()
    assert receipt == eng.recipe_receipt_id
    # Three distinct receipts, so a manifest can never confuse two lanes.
    others = {vreg.get_engine(n)._recipe_receipt()
              for n in ALL_THREE if n != name}
    assert receipt not in others


def test_all_three_receipts_are_distinct():
    receipts = [vreg.get_engine(n)._recipe_receipt() for n in ALL_THREE]
    assert len(set(receipts)) == 3, receipts


def test_all_three_modules_are_distinct():
    mods = [vreg.get_engine(n).motion_module_name for n in ALL_THREE]
    assert len(set(mods)) == 3, mods


# --------------------------------------------------------------------------- #
# G2 -- canvas truth. The pin the gate asks for.
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", ALL_THREE)
def test_the_declared_canvas_is_512x288_and_matches_the_graph(name):
    """G2.1/G2.2: the declaration must equal what the graph emits. Both read
    the same two constants, so a drift between them is impossible by
    construction -- and this asserts the constants themselves."""
    eng = vreg.get_engine(name)
    assert eng.render_canvas == (512, 288)
    assert (gs.GHOST_CANVAS_W, gs.GHOST_CANVAS_H) == (512, 288)
    # /32-legal on both axes (the live OTR canvas law).
    assert gs.GHOST_CANVAS_W % 32 == 0
    assert gs.GHOST_CANVAS_H % 32 == 0
    assert eng.target_fps == 25


# --------------------------------------------------------------------------- #
# ADDITIVE: the golden lane is untouched
# --------------------------------------------------------------------------- #

def test_the_golden_lane_is_unchanged():
    """The operator's condition for building these at all."""
    eng = vreg.get_engine(GOLDEN)
    assert eng.motion_module_name == "mm-p_0.5.pth"
    assert eng.recipe_receipt_id == "animatediff_sd15_mmp05_static16_512x288_v1"
    assert gs.GHOST_MOTION_MODULE_NAME == "mm-p_0.5.pth", (
        "the module-level default moved -- the golden lane is the frozen "
        "artifact and must not follow a peer")


@pytest.mark.parametrize("name", (V3, V2))
def test_a_peer_inherits_everything_else_from_the_golden_lane(name):
    """A comparison is only meaningful if NOTHING else differs."""
    gold = vreg.get_engine(GOLDEN)
    peer = vreg.get_engine(name)
    for attr in ("family", "roles", "default_roles", "required_inputs",
                 "render_aspect", "render_canvas", "target_fps",
                 "accepts_still", "still_plan", "subject_ownership",
                 "prompt_profile", "prompt_budget_chars", "style_join",
                 "delivery_scale_mode", "motion_source",
                 "negative_prompt_binding"):
        assert getattr(peer, attr) == getattr(gold, attr), attr
    assert peer.frame_contract == gold.frame_contract


# --------------------------------------------------------------------------- #
# Registration surfaces
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", (V3, V2))
def test_the_peer_is_registered_with_its_own_capability_row(name):
    assert vreg.is_registered(name)
    row = vreg.CAPABILITIES[name]
    assert set(row) == set(vreg.CAPABILITIES[GOLDEN])
    assert row["device_backends"] == ["cuda"]
    assert row["needs_fp8_te"] is False and row["needs_fp4_te"] is False
    # The row names the module THIS lane loads.
    assert EXPECTED_MODULE[name] in row["model_requirements"]
    assert "v1-5-pruned-emaonly-fp16.safetensors" in row["model_requirements"]
    # and NOT the golden module.
    assert "mm-p_0.5.pth" not in row["model_requirements"]


@pytest.mark.parametrize("name", (V3, V2))
def test_the_peer_has_an_honest_public_label(name):
    label = pub._PUBLIC_LABEL[name]
    assert "Apache-2.0" in label, (
        "the licence is the most load-bearing fact about these lanes and "
        "belongs in the label")
    # G7.4: no low/high marker without a measurement receipt.
    assert " low " not in label.lower() and " high " not in label.lower()
    assert pub.resolve_engine_id(name) == name


@pytest.mark.parametrize("name", (V3, V2))
def test_the_peer_makes_no_vram_claim(name):
    """No measurement campaign was authorized for any Ghost lane."""
    from nodes._otr_video_engines import motion_common as mc
    assert name not in getattr(mc, "QUALIFIED_COST_ROWS", ())
    label = pub._PUBLIC_LABEL[name]
    for claim in ("GiB", "GB VRAM", "fits"):
        assert claim not in label


@pytest.mark.parametrize("name", (V3, V2))
def test_the_peer_declares_commercial_clean_false(name):
    """Apache-2.0 removes the BLOCKER; it does not by itself make the lane
    commercially clean. RAIL-M carries use restrictions and nobody has reviewed
    that here, so the conservative declaration stands."""
    assert vreg.get_engine(name).commercial_clean is False


def test_the_module_constants_are_the_official_filenames():
    assert peers.MM_V3_NAME == "v3_sd15_mm.ckpt"
    assert peers.MM_V2_NAME == "mm_sd_v15_v2.ckpt"


# --------------------------------------------------------------------------- #
# THE BYTE FLOOR TRAVELS WITH THE MODULE.
#
# Caught by the first live v3 leg, which died in 6 minutes with:
#   "motion_module 'v3_sd15_mm.ckpt' is only 1673262583 bytes
#    (< the 1700000000 floor) -- that is a truncated or wrong file"
# The file was byte-perfect. The FLOOR was inherited from mm-p_0.5, which is
# 144 MB larger. A floor sized for one artifact is a false accusation against
# every other -- the same defect class as the module name itself, missed in the
# same way, and it cost a full leg to find.
# --------------------------------------------------------------------------- #

EXPECTED_FLOOR_UNDER = {
    GOLDEN: 1_817_894_327,   # mm-p_0.5.pth
    V3: 1_673_262_583,       # v3_sd15_mm.ckpt -- the smallest of the three
    V2: 1_817_888_431,       # mm_sd_v15_v2.ckpt
}


@pytest.mark.parametrize("name", ALL_THREE)
def test_each_lane_declares_a_floor_below_its_own_artifact(name):
    """A floor ABOVE the real file size refuses a perfect download by name."""
    eng = vreg.get_engine(name)
    real = EXPECTED_FLOOR_UNDER[name]
    assert eng.motion_min_bytes < real, (
        "%s: floor %d is at or above its module's real size %d -- this lane "
        "would refuse a byte-perfect file as truncated"
        % (name, eng.motion_min_bytes, real))


@pytest.mark.parametrize("name", ALL_THREE)
def test_the_floor_is_still_tight_enough_to_catch_a_truncated_fetch(name):
    """A floor of 1 byte would pass this file and every broken one. The guard
    only earns its place if a materially short file still fails."""
    eng = vreg.get_engine(name)
    real = EXPECTED_FLOOR_UNDER[name]
    assert eng.motion_min_bytes > real * 0.85, (
        "%s: floor %d is more than 15%% below the real size %d -- a badly "
        "truncated fetch would slip through"
        % (name, eng.motion_min_bytes, real))


def test_the_v3_floor_is_genuinely_lower_because_the_file_is_smaller():
    """Not a copy-paste of the golden floor. v3 is 144 MB smaller and its floor
    has to reflect that, or the lane cannot run at all."""
    v3 = vreg.get_engine(V3)
    gold = vreg.get_engine(GOLDEN)
    assert v3.motion_min_bytes < gold.motion_min_bytes
