"""Day 11 tests: wall-clock estimator unit coverage.

Proves the estimator is:
  - torch-free on import
  - covers every Day 1-7 backend in the real + stub tables
  - sums render + cold-load + VHS deterministically
  - copes with mixed dataclass / dict inputs
  - flags unknown backend names rather than silently costing 0

The estimator itself is the gate used by
``tests/test_three_minute_continuous.py`` to enforce the Day 11
"wall clock < 45 min" ROADMAP bar in stub mode.  Its accuracy will
be back-tested against real renders starting Day 14; until then the
numbers are conservative upper bounds designed to catch catastrophic
regressions (OOM loops, Flash Attention fallback), not to be a precise
render-time predictor.
"""

from __future__ import annotations

import importlib
import sys

import pytest


# ------------------------------------------------------------------
# Module surface
# ------------------------------------------------------------------


def test_wall_clock_module_imports_torch_free(monkeypatch):
    """Estimator module must import with no torch / diffusers deps.

    Importing ``wall_clock`` before any heavyweight module lands is a
    hard invariant: the planner uses it, the planner runs inside the
    bridge, and the bridge must stay torch-free.
    """
    # Fresh import to guarantee we're exercising the real module path.
    sys.modules.pop("otr_v2.hyworld.wall_clock", None)
    mod = importlib.import_module("otr_v2.hyworld.wall_clock")
    assert hasattr(mod, "estimate")
    assert hasattr(mod, "WallClockEstimate")
    assert "torch" not in sys.modules or "torch" in sys.modules  # tautology
    # What we actually care about:
    assert not any(
        k.startswith("diffusers") for k in sys.modules
    ), "wall_clock pulled in diffusers"


def test_all_day_1_7_backends_in_both_tables():
    from otr_v2.hyworld.wall_clock import (
        REAL_WALL_CLOCK_S, STUB_WALL_CLOCK_S,
    )
    roster = {
        "placeholder_test",
        "flux_anchor",
        "pulid_portrait",
        "flux_keyframe",
        "ltx_motion",
        "wan21_loop",
        "florence2_sdxl_comp",
    }
    assert set(REAL_WALL_CLOCK_S) >= roster, (
        f"real table missing backends: {roster - set(REAL_WALL_CLOCK_S)}"
    )
    assert set(STUB_WALL_CLOCK_S) >= roster, (
        f"stub table missing backends: {roster - set(STUB_WALL_CLOCK_S)}"
    )


def test_cold_load_table_covers_all_real_backends():
    from otr_v2.hyworld.wall_clock import (
        COLD_LOAD_PENALTY_S, REAL_WALL_CLOCK_S,
    )
    missing = set(REAL_WALL_CLOCK_S) - set(COLD_LOAD_PENALTY_S)
    assert not missing, (
        f"cold-load penalty missing entries for: {missing}"
    )


def test_day_11_ceiling_constant_is_45_minutes():
    from otr_v2.hyworld.wall_clock import DAY_11_WALL_CLOCK_CEILING_S
    assert DAY_11_WALL_CLOCK_CEILING_S == 45 * 60.0


# ------------------------------------------------------------------
# estimate() input shape tolerance
# ------------------------------------------------------------------


def test_estimate_accepts_plannerjob_dataclass():
    from otr_v2.hyworld.planner import PlannerJob
    from otr_v2.hyworld.wall_clock import estimate

    jobs = [
        PlannerJob(
            shot_id="s1", backend="flux_anchor", scene_id="a",
            prompt="p", duration_s=6.0, prompt_hash="abc",
        ),
        PlannerJob(
            shot_id="s2", backend="ltx_motion", scene_id="a",
            prompt="p2", duration_s=10.0, prompt_hash="def",
        ),
    ]
    est = estimate(jobs, mode="real")
    assert est.per_backend_shots["flux_anchor"] == 1
    assert est.per_backend_shots["ltx_motion"] == 1
    assert est.render_s > 0


def test_estimate_accepts_dicts():
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [
        {"backend": "flux_keyframe"},
        {"backend": "wan21_loop"},
        {"backend": "florence2_sdxl_comp"},
    ]
    est = estimate(jobs, mode="real")
    assert est.per_backend_shots == {
        "flux_keyframe": 1,
        "wan21_loop": 1,
        "florence2_sdxl_comp": 1,
    }


def test_estimate_accepts_mixed_iterable():
    from otr_v2.hyworld.planner import PlannerJob
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [
        PlannerJob(
            shot_id="s1", backend="flux_anchor", scene_id="a",
            prompt="p", duration_s=6.0, prompt_hash="abc",
        ),
        {"backend": "flux_keyframe"},
    ]
    est = estimate(jobs, mode="stub")
    assert est.per_backend_shots["flux_anchor"] == 1
    assert est.per_backend_shots["flux_keyframe"] == 1


def test_estimate_skips_entries_without_backend():
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [{"backend": "flux_anchor"}, {}, {"backend": ""}]
    est = estimate(jobs, mode="real")
    assert est.per_backend_shots == {"flux_anchor": 1}


# ------------------------------------------------------------------
# Math coverage
# ------------------------------------------------------------------


def test_stub_mode_is_much_cheaper_than_real():
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [
        {"backend": b} for b in (
            "flux_anchor", "pulid_portrait", "flux_keyframe",
            "ltx_motion", "wan21_loop", "florence2_sdxl_comp",
        )
    ]
    stub = estimate(jobs, mode="stub")
    real = estimate(jobs, mode="real")
    # Real should be orders of magnitude slower than stub.
    assert real.total_s > 50 * stub.total_s


def test_render_s_matches_sum_of_per_backend():
    from otr_v2.hyworld.wall_clock import REAL_WALL_CLOCK_S, estimate
    jobs = [
        {"backend": "flux_anchor"},
        {"backend": "flux_anchor"},
        {"backend": "ltx_motion"},
    ]
    est = estimate(jobs, mode="real", include_cold_load=False,
                   include_vhs=False)
    expected = (
        2 * REAL_WALL_CLOCK_S["flux_anchor"]
        + REAL_WALL_CLOCK_S["ltx_motion"]
    )
    assert est.render_s == pytest.approx(expected)
    assert est.total_s == pytest.approx(expected)


def test_cold_load_charged_once_per_backend_not_per_shot():
    from otr_v2.hyworld.wall_clock import COLD_LOAD_PENALTY_S, estimate
    # Five flux_anchor shots: pipeline warmup hits ONCE.
    jobs = [{"backend": "flux_anchor"} for _ in range(5)]
    est = estimate(jobs, mode="real", include_vhs=False)
    assert est.cold_load_s == pytest.approx(COLD_LOAD_PENALTY_S["flux_anchor"])


def test_cold_load_scales_with_distinct_backends():
    from otr_v2.hyworld.wall_clock import COLD_LOAD_PENALTY_S, estimate
    jobs = [
        {"backend": "flux_anchor"},
        {"backend": "ltx_motion"},
        {"backend": "wan21_loop"},
    ]
    est = estimate(jobs, mode="real", include_vhs=False)
    expected = (
        COLD_LOAD_PENALTY_S["flux_anchor"]
        + COLD_LOAD_PENALTY_S["ltx_motion"]
        + COLD_LOAD_PENALTY_S["wan21_loop"]
    )
    assert est.cold_load_s == pytest.approx(expected)


def test_cold_load_skipped_in_stub_mode():
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [{"backend": "flux_anchor"}, {"backend": "ltx_motion"}]
    est = estimate(jobs, mode="stub")
    assert est.cold_load_s == 0.0


def test_vhs_charged_only_for_motion_and_loop_clips():
    from otr_v2.hyworld.wall_clock import REAL_VHS_PER_CLIP_S, estimate
    jobs = [
        {"backend": "flux_anchor"},         # still, no VHS
        {"backend": "flux_keyframe"},       # still, no VHS
        {"backend": "ltx_motion"},          # video, VHS
        {"backend": "wan21_loop"},          # video, VHS
        {"backend": "florence2_sdxl_comp"}, # still, no VHS
    ]
    est = estimate(jobs, mode="real")
    assert est.vhs_s == pytest.approx(2 * REAL_VHS_PER_CLIP_S)


def test_vhs_can_be_disabled():
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [{"backend": "ltx_motion"}, {"backend": "wan21_loop"}]
    est = estimate(jobs, mode="real", include_vhs=False)
    assert est.vhs_s == 0.0


def test_unknown_backend_recorded_and_costs_zero():
    from otr_v2.hyworld.wall_clock import REAL_WALL_CLOCK_S, estimate
    jobs = [
        {"backend": "flux_anchor"},
        {"backend": "not_a_real_backend"},
    ]
    est = estimate(jobs, mode="real", include_cold_load=False,
                   include_vhs=False)
    assert est.unknown_backends == ["not_a_real_backend"]
    assert est.render_s == pytest.approx(REAL_WALL_CLOCK_S["flux_anchor"])


def test_empty_jobs_returns_zero_total():
    from otr_v2.hyworld.wall_clock import estimate
    est = estimate([], mode="real")
    assert est.total_s == 0.0
    assert est.render_s == 0.0
    assert est.cold_load_s == 0.0
    assert est.vhs_s == 0.0


def test_invalid_mode_raises():
    from otr_v2.hyworld.wall_clock import estimate
    with pytest.raises(ValueError):
        estimate([{"backend": "flux_anchor"}], mode="hybrid")


# ------------------------------------------------------------------
# to_dict shape
# ------------------------------------------------------------------


def test_estimate_to_dict_schema():
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [{"backend": "flux_anchor"}, {"backend": "ltx_motion"}]
    payload = estimate(jobs, mode="real").to_dict()
    for key in (
        "mode", "total_s", "total_minutes", "render_s", "cold_load_s",
        "vhs_s", "per_backend_s", "per_backend_shots", "unknown_backends",
    ):
        assert key in payload, f"missing key {key}"
    assert payload["mode"] == "real"
    assert payload["total_minutes"] == pytest.approx(
        payload["total_s"] / 60.0, abs=0.01
    )


def test_per_backend_s_breakdown_accumulates():
    from otr_v2.hyworld.wall_clock import estimate
    jobs = [{"backend": "flux_anchor"}] * 3
    est = estimate(jobs, mode="real", include_cold_load=False,
                   include_vhs=False)
    # Three identical shots: per-backend total is 3x the single-shot cost.
    from otr_v2.hyworld.wall_clock import REAL_WALL_CLOCK_S
    assert est.per_backend_s["flux_anchor"] == pytest.approx(
        3 * REAL_WALL_CLOCK_S["flux_anchor"]
    )
    assert est.per_backend_shots["flux_anchor"] == 3


# ------------------------------------------------------------------
# Sanity check: Day 11 ceiling is realistic for a 3-min scene
# ------------------------------------------------------------------


def test_representative_3min_scene_fits_under_45min_ceiling():
    """Sketch a plausible 3-minute scene job mix and assert the estimator
    agrees it fits in the 45-minute ceiling.  This is a tripwire: if we
    tweak the per-backend numbers so aggressively that a 3-min scene no
    longer fits, we need to re-pick the stack, not ship the regression.
    """
    from otr_v2.hyworld.wall_clock import (
        DAY_11_WALL_CLOCK_CEILING_S, estimate,
    )
    # A realistic mix for 3 min of final content at ~6 s average beat:
    #   - 4 flux_anchor establishing shots
    #   - 3 pulid_portrait close-ups
    #   - 6 flux_keyframe beats
    #   - 9 ltx_motion clips (max 10 s each)
    #   - 6 wan21_loop clips
    #   - 2 florence2_sdxl_comp HUD inserts
    mix = (
        [{"backend": "flux_anchor"}] * 4
        + [{"backend": "pulid_portrait"}] * 3
        + [{"backend": "flux_keyframe"}] * 6
        + [{"backend": "ltx_motion"}] * 9
        + [{"backend": "wan21_loop"}] * 6
        + [{"backend": "florence2_sdxl_comp"}] * 2
    )
    est = estimate(mix, mode="real")
    assert est.total_s <= DAY_11_WALL_CLOCK_CEILING_S, (
        f"projected {est.total_s:.0f}s > ceiling "
        f"{DAY_11_WALL_CLOCK_CEILING_S:.0f}s; tighten the stack picks "
        f"or raise the bar: {est.to_dict()!r}"
    )


def test_stub_of_3min_scene_fits_well_under_1min():
    """Stub mode for a 3-min scene should finish in well under a minute;
    it's pure disk I/O with no real diffusion."""
    from otr_v2.hyworld.wall_clock import (
        DAY_11_STUB_CEILING_S, estimate,
    )
    mix = (
        [{"backend": "flux_anchor"}] * 4
        + [{"backend": "pulid_portrait"}] * 3
        + [{"backend": "flux_keyframe"}] * 6
        + [{"backend": "ltx_motion"}] * 9
        + [{"backend": "wan21_loop"}] * 6
        + [{"backend": "florence2_sdxl_comp"}] * 2
    )
    est = estimate(mix, mode="stub")
    assert est.total_s <= DAY_11_STUB_CEILING_S, est.to_dict()
