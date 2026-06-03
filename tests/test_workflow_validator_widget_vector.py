"""S14.3 / BUG-LOCAL-293 -- widget-vector contract math in OTR_WorkflowValidator.

Unit-tests the serialized-slot math (forceInput dropped, seed companion added,
sockets excluded, COMBO counted). The legacy musicgen / audiogen manifest-drift
cases were removed in the audio clean-break (1c): those nodes + the legacy
invocation manifest are gone, and every audio engine is now a self-contained
registry engine (per_line / clip), so there is no batch-delegation widget vector
to drift-check.
"""
from __future__ import annotations

from nodes._otr_workflow_validator import _expected_slot_count


def test_expected_slot_count_rules():
    it = {
        "required": {
            "script_json": ("STRING", {"forceInput": True}),  # dropped
            "episode_seed": ("STRING", {}),                    # 1
            "seed": ("INT", {}),                               # 2 (+companion -> 3)
            "model": ("MODEL", {}),                            # socket, excluded
            "mode": (["a", "b"], {}),                          # COMBO -> 4
        }
    }
    assert _expected_slot_count(it) == 4


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v"]))
