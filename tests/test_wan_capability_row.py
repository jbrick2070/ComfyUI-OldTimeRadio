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
from nodes._otr_video_engines import eng_wan_i2v  # noqa: F401  (register adapter)
from nodes._otr_video_engines import eng_wan_ti2v  # noqa: F401  (register adapter)


def test_wan_i2v_model_requirement_is_wan22_not_stale_wan21():
    reqs = vreg.CAPABILITIES["wan_i2v"]["model_requirements"]
    assert reqs == ["wan2.2-i2v"]
    assert "wan2.1-i2v" not in reqs


def test_wan_i2v_row_still_consistent_with_registry():
    # The S1/S5 edit must not break the every-row-has-an-engine invariant.
    assert "wan_i2v" in vreg.all_engine_names()


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
