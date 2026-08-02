"""The estimator-fit's math, CPU-proven before any GPU hour is spent on it.

The fit's one job: turn (length, demand) samples into a cost row the ONLY
enforcing guard on the planned path can trust. A mean fit converts run-to-run
variance into an unguarded CUDA OOM, so the shipped row is the UPPER envelope
-- every measured point must sit at or under it.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "otr_video_arm_bakeoff",
    pathlib.Path(__file__).resolve().parents[1]
    / "scripts" / "run_video_arm_bakeoff.py")
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
fit_cost_row = _MOD.fit_cost_row
BenchError = _MOD.BenchError


def _covered(fit, points):
    env = fit["envelope"]
    return all(d <= env["overhead_mib"] + env["per_frame_mib"] * l + 1e-6
               for l, d in points)


def test_a_clean_linear_curve_fits_exactly():
    points = [(17, 7000 + 17 * 10), (49, 7000 + 49 * 10),
              (81, 7000 + 81 * 10), (113, 7000 + 113 * 10)]
    fit = fit_cost_row(points)
    assert fit["least_squares"]["overhead_mib"] == pytest.approx(7000, abs=0.1)
    assert fit["least_squares"]["per_frame_mib"] == pytest.approx(10, abs=0.01)
    assert fit["max_positive_residual_mib"] == pytest.approx(0, abs=0.1)
    assert _covered(fit, points)


def test_the_envelope_covers_every_noisy_point():
    """The property that matters: no measured demand may exceed the row."""
    points = [(17, 6500), (17, 6620), (49, 6480), (49, 6710),
              (81, 6550), (81, 6600), (113, 6900), (145, 6470), (177, 6580)]
    fit = fit_cost_row(points)
    assert _covered(fit, points), "a point above the envelope is a future OOM"
    assert fit["max_positive_residual_mib"] > 0


def test_a_FLAT_curve_clamps_slope_at_zero_and_still_covers():
    """The fastwan bench measured FLAT (6563/6531/6563 at 17/49/81). Noise can
    tilt the least-squares slope negative; the model's shape does not admit a
    negative per-frame cost, so it clamps to 0 and the envelope re-anchors."""
    points = [(17, 6563), (49, 6531), (81, 6563), (113, 6540),
              (145, 6520), (177, 6555)]
    fit = fit_cost_row(points)
    assert fit["envelope"]["per_frame_mib"] >= 0.0
    assert _covered(fit, points)


def test_degenerate_samples_REFUSE_rather_than_extrapolate():
    with pytest.raises(BenchError):
        fit_cost_row([(17, 6500), (17, 6600), (17, 6550)])   # one length
    with pytest.raises(BenchError):
        fit_cost_row([(17, 6500), (49, 6600)])               # too few points


def test_the_receipt_carries_the_evidence_not_just_the_verdict():
    points = [(17, 6500), (49, 6600), (81, 6700), (113, 6800)]
    fit = fit_cost_row(points)
    assert len(fit["points"]) == 4
    assert len(fit["residuals_mib"]) == 4
    assert fit["unit"].startswith("MiB of demand above")
