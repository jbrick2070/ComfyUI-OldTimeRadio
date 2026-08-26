"""GO_FORWARD section 4A S1 + S5 + S2 -- the Wan CAPABILITIES rows.

S1: wan_i2v vram_estimate raised to the conservative 14500 (the 14499 MB smoke
figure was WITHOUT the load-bearing free_after_use). S5: model_requirements
carries the real Wan 2.2 I2V asset id, not the stale wan2.1 label.

S2 (2026-06-14): the concrete wan_ti2v 8GB-tier row now lands -- the 5B core node
class (Wan22ImageToVideoLatent) was captured from a live /object_info and the
engine (eng_wan_ti2v) is registered, so the registry-consistency invariant (every
CAPABILITIES key has a REGISTERED engine) holds.
"""

from __future__ import annotations

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import eng_wan_ti2v  # noqa: F401  (register adapter)


def test_retired_wan_i2v_has_no_capability_row_and_no_engine():
    """wan_i2v RETIRED 2026-08-26 (19.82 GiB of weights vs a 14.5 GiB target).

    The registry-consistency invariant cuts BOTH ways: a row without an engine
    is a bug, and so is a row for an engine that no longer exists. This is the
    same assertion the deleted `test_wan_i2v_row_still_consistent_with_registry`
    made, inverted -- so the invariant keeps a test after the rip instead of
    losing one.
    """
    assert "wan_i2v" not in vreg.CAPABILITIES
    assert "wan_i2v" not in vreg.all_engine_names()


def test_retired_wan_i2v_is_a_named_tombstone_not_an_unknown_engine():
    """A saved graph naming the 14B must get the RETIRED diagnosis, never the
    generic unregistered-engine refusal -- "not registered" reads as a broken
    install, which is exactly what the tombstone list exists to prevent."""
    from nodes._otr_shared import public_engines as _pe
    assert "wan_i2v" in _pe.RETIRED_ENGINE_IDS
    # the two PUBLIC aliases must still route INTO that refusal
    assert _pe._PUBLIC_ENGINES.get("wan22_high_i2v") == "wan_i2v"


def test_wan_ti2v_row_present_and_registered():
    # S2: the row + the engine land together (registry-consistency invariant).
    assert "wan_ti2v" in vreg.CAPABILITIES
    assert "wan_ti2v" in vreg.all_engine_names()


def test_wan_ti2v_row_capabilities():
    row = vreg.CAPABILITIES["wan_ti2v"]
    assert row["required_toolchain"] is None
    assert row["requires_sidecar"] is False
    assert row["device_backends"] == ["cuda"]
    assert row["model_requirements"] == ["wan2.2-ti2v-5b"]
