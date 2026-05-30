"""S10.1 -- Drift guards for the G7 SFX dur_s contract surface.

The bounds [SFX_DUR_MIN_S, SFX_DUR_MAX_S] live in
``nodes/_otr_ledger_freeze.py`` and are the single source of truth.
AudioGen MUST import those constants for its widget min/max AND its
internal defensive clamp -- no magic numbers. This test file pins that
contract.

If any of these tests fail, a consumer has either dropped the import
or hard-coded a number. Both modes silently re-create the contract-
lie surface S10.1 closed.
"""
from __future__ import annotations

from nodes._otr_ledger_freeze import SFX_DUR_MIN_S, SFX_DUR_MAX_S
import nodes.batch_audiogen_generator as ag


def _widget_spec(cls, name):
    """Pull the widget spec dict for ``name`` from a node class.

    INPUT_TYPES() returns ``{"required": {...}, "optional": {...}}``.
    Each entry is ``(type, spec_dict)``. We want the spec_dict.
    """
    types = cls.INPUT_TYPES()
    for bucket in ("required", "optional"):
        slot = (types.get(bucket) or {}).get(name)
        if slot is not None:
            assert isinstance(slot, tuple) and len(slot) >= 2, slot
            return slot[1]
    raise AssertionError(
        f"{cls.__name__} has no widget {name!r} in required+optional"
    )


# ---------------------------------------------------------------------------
# Widget-min / widget-max imports
# ---------------------------------------------------------------------------


def test_audiogen_default_duration_widget_uses_g7_constants():
    spec = _widget_spec(ag.BatchAudioGenGenerator, "default_duration")
    assert spec["min"] == SFX_DUR_MIN_S, (
        f"AudioGen default_duration min={spec['min']} != "
        f"G7 SFX_DUR_MIN_S={SFX_DUR_MIN_S}. Magic number leak."
    )
    assert spec["max"] == SFX_DUR_MAX_S, (
        f"AudioGen default_duration max={spec['max']} != "
        f"G7 SFX_DUR_MAX_S={SFX_DUR_MAX_S}. Magic number leak."
    )


# ---------------------------------------------------------------------------
# Source-level "no magic numbers in clamps" guards
# ---------------------------------------------------------------------------


def test_audiogen_module_clamps_through_constants_not_literals():
    """AudioGen's per-cue clamp must reference SFX_DUR_MIN_S /
    SFX_DUR_MAX_S, not raw numeric literals."""
    import inspect
    src = inspect.getsource(ag)
    assert "max(SFX_DUR_MIN_S" in src and "min(SFX_DUR_MAX_S" in src, (
        "AudioGen clamp does not reference G7 constants. The defensive "
        "max/min around line.dur_s must call max(SFX_DUR_MIN_S, "
        "min(SFX_DUR_MAX_S, ...)) so the consumer moves with G7."
    )


# ---------------------------------------------------------------------------
# Cross-module identity guard
# ---------------------------------------------------------------------------


def test_consumer_clamp_constants_are_identical():
    """Sanity check: the constants imported by the consumers MUST be
    the same object as those exported by _otr_ledger_freeze. Catches
    a future refactor that defines local SFX_DUR_MIN_S/MAX_S in the
    consumer file (which would silently shadow the contract)."""
    from nodes._otr_ledger_freeze import (
        SFX_DUR_MIN_S as freeze_min,
        SFX_DUR_MAX_S as freeze_max,
    )
    assert ag.SFX_DUR_MIN_S is freeze_min
    assert ag.SFX_DUR_MAX_S is freeze_max
