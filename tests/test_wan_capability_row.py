"""GO_FORWARD section 4A S1 + S5 -- the wan_i2v CAPABILITIES row.

S1: vram_estimate raised to the conservative 14500 (the 14499 MB smoke figure
was WITHOUT the load-bearing free_after_use). S5: model_requirements carries
the real Wan 2.2 I2V asset id, not the stale wan2.1 label.

(S2 -- a concrete wan_ti2v row -- is intentionally NOT added here: the registry
consistency invariant requires every CAPABILITIES key to have a REGISTERED
engine, and wan_ti2v is the deferred 8GB tier whose 5B core node class must be
captured from a live /object_info before the engine is built.)
"""

from __future__ import annotations

from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import eng_wan_i2v  # noqa: F401  (register adapter)


def test_wan_i2v_vram_estimate_is_conservative_14500():
    assert vreg.CAPABILITIES["wan_i2v"]["vram_estimate_mb"] == 14500


def test_wan_i2v_model_requirement_is_wan22_not_stale_wan21():
    reqs = vreg.CAPABILITIES["wan_i2v"]["model_requirements"]
    assert reqs == ["wan2.2-i2v"]
    assert "wan2.1-i2v" not in reqs


def test_wan_i2v_row_still_consistent_with_registry():
    # The S1/S5 edit must not break the every-row-has-an-engine invariant.
    assert "wan_i2v" in vreg.all_engine_names()
    assert "wan_ti2v" not in vreg.CAPABILITIES  # deferred -- not yet built
