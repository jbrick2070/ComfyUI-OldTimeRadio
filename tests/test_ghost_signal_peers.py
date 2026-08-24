"""The per-artifact seam on the surviving Ghost lane.

THIS FILE USED TO BE ABOUT THE PEERS -- three lanes on three motion modules,
compared against each other. The operator retired every non-haunted lane on
2026-08-23 ("delete any animatediff that are not haunted"), so the comparison
has no second subject and the distinctness tests went with it.

WHAT DID NOT GO IS THE DEFECT CLASS, and it is why this file still exists.
`eng_fastwan_8gb` records it: a parent whose per-artifact constants are read from
MODULE SCOPE instead of through ``self`` means a subclass "would have SILENTLY
rendered with the parent's [weights] and stamped its own receipt on the result".
The lane that survived is a subclass TWO levels deep --
``GhostSignalV3HauntedEngine`` <- ``GhostSignalV3Engine`` <- ``GhostSignalEngine``
-- which makes it the most exposed lane this repo has ever had to exactly that
bug. Retiring its siblings removed the comparison, not the risk.

AND THE BYTE FLOOR IS HERE FOR A REASON THAT COST A LIVE LEG. The first v3 leg
died in six minutes with *"motion_module 'v3_sd15_mm.ckpt' is only 1673262583
bytes (< the 1700000000 floor) -- that is a truncated or wrong file"*. The file
was byte-perfect; the FLOOR had been inherited from mm-p_0.5, which is 144 MB
larger. A floor sized for one artifact is a false accusation against every other,
and the surviving lane inherits both its module and its floor.

``V3`` below is the UNREGISTERED parent class -- still instantiable, still the
machinery the haunted lane inherits, no longer a public id.
"""
from __future__ import annotations

import inspect

import pytest

import nodes._otr_video_engines  # noqa: F401 -- populate the registry
from nodes._otr_video_engines import eng_ghost_signal as gs
from nodes._otr_video_engines import eng_ghost_signal_official as peers
from nodes._otr_video_engines import registry as vreg

HAUNTED = "animatediff15_v3_haunted_video"
V3 = "animatediff15_v3_video"          # unregistered parent class
LANES = (HAUNTED, V3)

#: Both run the official v3 module -- the haunted lane INHERITS it rather than
#: declaring its own, which is precisely why the seam below still matters.
EXPECTED_MODULE = {HAUNTED: "v3_sd15_mm.ckpt", V3: "v3_sd15_mm.ckpt"}

#: The REAL on-disk size of each lane's module. A floor at or above this refuses
#: a perfect download by name.
EXPECTED_FLOOR_UNDER = {HAUNTED: 1_673_262_583, V3: 1_673_262_583}

_UNREGISTERED = {V3: peers.GhostSignalV3Engine}


def _engine(name):
    """The engine for ``name``, registered or not."""
    if name in _UNREGISTERED:
        return _UNREGISTERED[name]()
    return vreg.get_engine(name)


# --------------------------------------------------------------------------- #
# THE SEAM
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", LANES)
def test_each_lane_actually_loads_its_own_module(name):
    """The fastwan defect, guarded. Every read of the module name must go
    through ``self``, or a subclass renders with the parent's weights."""
    assert _engine(name).motion_module_name == EXPECTED_MODULE[name]


def test_the_graph_asks_for_that_module_and_not_a_constant():
    """Not the declaration -- the GRAPH. The ADE loader must receive
    ``self.motion_module_name``, never the module-level constant."""
    src = inspect.getsource(gs.GhostSignalEngine.render_clip)
    assert '"model_name": self.motion_module_name' in src, (
        "the ADE graph reads a module-level constant, so a subclass would load "
        "the parent's module while stamping its own receipt")
    assert '"model_name": GHOST_MOTION_MODULE_NAME' not in src


@pytest.mark.parametrize("name", LANES)
def test_each_lane_stamps_a_receipt_naming_its_own_recipe(name):
    """A receipt is what a render is read against; it may never be a guess."""
    engine = _engine(name)
    assert engine._recipe_receipt() == engine.recipe_receipt_id
    assert engine._recipe_receipt()


def test_the_haunted_receipt_is_not_its_parents():
    """The one distinctness claim that survives, and the one that matters: the
    subclass must not inherit the RECEIPT along with the module. It shares the
    weights deliberately; sharing the receipt would make a render unattributable.
    """
    assert _engine(HAUNTED)._recipe_receipt() != _engine(V3)._recipe_receipt()


def test_the_module_constant_is_the_official_filename():
    assert peers.MM_V3_NAME == "v3_sd15_mm.ckpt"


# --------------------------------------------------------------------------- #
# THE BYTE FLOOR TRAVELS WITH THE MODULE
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", LANES)
def test_each_lane_declares_a_floor_below_its_own_artifact(name):
    """A floor ABOVE the real file size refuses a perfect download by name."""
    engine = _engine(name)
    real = EXPECTED_FLOOR_UNDER[name]
    assert engine.motion_min_bytes < real, (
        "%s: floor %d is at or above its module's real size %d -- this lane "
        "would refuse a byte-perfect file as truncated"
        % (name, engine.motion_min_bytes, real))


@pytest.mark.parametrize("name", LANES)
def test_the_floor_is_still_tight_enough_to_catch_a_truncated_fetch(name):
    """A floor of 1 byte would pass this file and every broken one. The guard
    only earns its place if a materially short file still fails."""
    engine = _engine(name)
    real = EXPECTED_FLOOR_UNDER[name]
    assert engine.motion_min_bytes > real * 0.85, (
        "%s: floor %d is more than 15%% below the real size %d -- a badly "
        "truncated fetch would slip through"
        % (name, engine.motion_min_bytes, real))


@pytest.mark.parametrize("name", LANES)
def test_the_declared_canvas_is_512x288(name):
    assert _engine(name).render_canvas == (512, 288)
