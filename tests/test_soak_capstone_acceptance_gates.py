"""GO_FORWARD section 4A acceptance-gate unit tests (M1 + M4).

Pure-Python coverage of the two capstone gate helpers added to harden the
GATE-A coverage sweep against the false-GREEN failure modes the 2026-06-13
roundtable surfaced:

    * M1 -- assert_no_runtime_fallback: the sweep ran every leg with
      expect_engine="" (informational), so a shot that silently fell back to
      the radio floor scored PASS (CS-1). The gate fails any shot whose
      final_engine != its requested attempts[0].
    * M4 -- resolve_driver_peak_mb: the V-3 VRAM gate defaulted a missing
      measurement to -1 and only failed on > ceiling, so a missing / 0 /
      negative peak read GREEN with no measurement. The helper returns None for
      any unusable measurement; the caller fails closed on None.

These import the live harness module from scripts/ -- otr_api is stdlib +
requests only, so the import is cold-clean. No server, no GPU, no real render.
"""

from __future__ import annotations

import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS = os.path.join(_REPO, "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

import _otr_soak_capstone as soak  # noqa: E402


# --------------------------------------------------------------------------- #
# M1 -- assert_no_runtime_fallback
# --------------------------------------------------------------------------- #
def _row(shot_id, requested, final, *, extra_attempts=()):
    """A trace row shaped like render_driver.run_episode emits (attempts[0] is
    the requested engine, final_engine is what actually rendered)."""
    return {"shot_id": shot_id, "attempts": [requested, *extra_attempts],
            "final_engine": final}


def test_no_fallback_passes_when_every_shot_stays_on_requested():
    trace = [
        _row("b000", "wan_i2v", "wan_i2v"),
        _row("b001", "humo_1.7B", "humo_1.7B"),
        _row("b002", "visualizer", "visualizer"),
    ]
    # No raise == pass.
    soak.assert_no_runtime_fallback(trace)


def test_no_fallback_passes_on_empty_trace():
    # Vacuously clean: nothing fell off. (Emptiness is the sweep's concern, M2.)
    soak.assert_no_runtime_fallback([])
    soak.assert_no_runtime_fallback(None)


def test_no_fallback_fails_on_silent_floor_fallback():
    # The exact CS-1 shape: a wan_i2v shot that fell back to the radio floor.
    trace = [
        _row("b000", "wan_i2v", "still_motion", extra_attempts=("still_motion",)),
        _row("b001", "humo_1.7B", "humo_1.7B"),
    ]
    with pytest.raises(soak.SoakFail) as exc:
        soak.assert_no_runtime_fallback(trace)
    assert "RUNTIME FALLBACK" in str(exc.value)
    assert "b000" in str(exc.value)
    # The clean shot is not named.
    assert "b001" not in str(exc.value)


def test_no_fallback_fails_closed_on_missing_final_engine():
    trace = [{"shot_id": "b000", "attempts": ["wan_i2v"]}]  # no final_engine
    with pytest.raises(soak.SoakFail):
        soak.assert_no_runtime_fallback(trace)


def test_no_fallback_fails_closed_on_missing_attempts():
    trace = [{"shot_id": "b000", "final_engine": "wan_i2v"}]  # no attempts
    with pytest.raises(soak.SoakFail):
        soak.assert_no_runtime_fallback(trace)


def test_no_fallback_fails_closed_on_non_dict_row():
    with pytest.raises(soak.SoakFail):
        soak.assert_no_runtime_fallback(["not-a-dict"])


# --------------------------------------------------------------------------- #
# M4 -- resolve_driver_peak_mb (fail-closed measurement)
# --------------------------------------------------------------------------- #
def test_driver_peak_returns_value_for_real_measurement():
    assert soak.resolve_driver_peak_mb({"vram_peak_mb": 11300}) == 11300


def test_driver_peak_none_when_absent():
    assert soak.resolve_driver_peak_mb({}) is None


def test_driver_peak_none_when_zero():
    assert soak.resolve_driver_peak_mb({"vram_peak_mb": 0}) is None


def test_driver_peak_none_when_negative():
    # The old `or -1` default -- must now read as "no measurement".
    assert soak.resolve_driver_peak_mb({"vram_peak_mb": -1}) is None


def test_driver_peak_none_when_unparseable():
    assert soak.resolve_driver_peak_mb({"vram_peak_mb": None}) is None
    assert soak.resolve_driver_peak_mb({"vram_peak_mb": "garbage"}) is None


def test_driver_peak_accepts_numeric_string():
    assert soak.resolve_driver_peak_mb({"vram_peak_mb": "12500"}) == 12500
