"""Day 11 gate: 3-minute continuous scene at full res.

ROADMAP bar:
    "3-min continuous scene at full res.
     No stagnation, no duplicate stills.  Wall clock < 45 min."

This is a torch-free stub-mode test.  It drives the planner with a
3-minute outline, asserts the planned job list itself obeys the
"no stagnation / no duplicate stills" invariant, then projects wall
clock via ``otr_v2.visual.wall_clock.estimate`` and enforces the 45
minute ceiling.  Real-mode render-time validation is deferred to
Day 14 once every backend's weights are on disk.

The outline is a continuous 3-min scene inside the "Cockpit" location
with eight beats spanning every Day 1-7 backend kind, so the planner
has room to produce variety and the non-repetition window gets stressed.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from otr_v2.visual.backends import resolve
from otr_v2.visual.planner import (
    DEFAULT_NONREPEAT_WINDOW,
    PlannerResult,
    plan_episode,
)
from otr_v2.visual.postproc import vhs as vhs_mod
from otr_v2.visual.wall_clock import (
    DAY_11_STUB_CEILING_S,
    DAY_11_WALL_CLOCK_CEILING_S,
    estimate,
)


# ------------------------------------------------------------------
# Outline: 3-minute continuous Cockpit scene
# ------------------------------------------------------------------


def _three_minute_cockpit_outline() -> dict[str, Any]:
    """A continuous 180-s scene set inside the Cockpit location.

    Eight beats covering every Day 1-7 backend kind.  Scene rotation
    will kick in (since sum(beats) ~ 60-70 s < 180 s target) which
    exercises the planner's "rotate scenes from top" safety net.
    """
    return {
        "runtime_s": 180.0,
        "scenes": [
            {
                "scene_id": "scene_01",
                "title": "3MIN COCKPIT",
                "location": "Cockpit",
                "characters": ["BABA", "BOOEY"],
                "beats": [
                    {
                        "beat_id": "b01",
                        "kind": "establishing",
                        "prompt": (
                            "wide establishing of cockpit at dawn, "
                            "retrofuture panels glowing amber"
                        ),
                        "duration_sec": 8.0,
                        "camera": "wide establishing, slight dutch tilt",
                    },
                    {
                        "beat_id": "b02",
                        "kind": "close_up",
                        "character": "BABA",
                        "refs": ["baba_01.png", "baba_02.png"],
                        "prompt": (
                            "BABA close-up, hand on tuning dial, "
                            "weary focus"
                        ),
                        "duration_sec": 6.0,
                    },
                    {
                        "beat_id": "b03",
                        "kind": "keyframe",
                        "prompt": (
                            "mid-shot of radio console, needle crossing "
                            "107.3 MHz, soft neon"
                        ),
                        "duration_sec": 7.0,
                    },
                    {
                        "beat_id": "b04",
                        "kind": "motion",
                        "prompt": (
                            "push-in on console, LED bars ticking upward"
                        ),
                        "duration_sec": 9.0,
                    },
                    {
                        "beat_id": "b05",
                        "kind": "loop",
                        "prompt": (
                            "cockpit ambient, dust motes drifting "
                            "in amber light"
                        ),
                        "duration_sec": 10.0,
                    },
                    {
                        "beat_id": "b06",
                        "kind": "insert",
                        "prompt": "HUD overlay with frequency readout",
                        "mask_prompt": "cockpit viewport frame",
                        "insert_prompt": (
                            "retro amber HUD bars and frequency readout "
                            "107.3 MHz"
                        ),
                        "duration_sec": 4.0,
                    },
                    {
                        "beat_id": "b07",
                        "kind": "two_shot",
                        "character": "BOOEY",
                        "refs": ["booey_01.png"],
                        "prompt": (
                            "BOOEY in the co-pilot seat, half-turned, "
                            "catching BABA's glance"
                        ),
                        "duration_sec": 6.0,
                    },
                    {
                        "beat_id": "b08",
                        "kind": "ambient",
                        "prompt": (
                            "slow orbital pan across flickering stars "
                            "beyond the viewport"
                        ),
                        "duration_sec": 10.0,
                    },
                ],
            }
        ],
    }


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


_STUB_ENVVARS = (
    "OTR_FLUX_STUB",
    "OTR_PULID_STUB",
    "OTR_FLUX_KEYFRAME_STUB",
    "OTR_LTX_STUB",
    "OTR_WAN_STUB",
    "OTR_FLORENCE_STUB",
    "OTR_VHS_STUB",
)


@pytest.fixture
def stub_env(monkeypatch):
    for key in _STUB_ENVVARS:
        monkeypatch.setenv(key, "1")
    yield


@pytest.fixture
def scene_root(tmp_path):
    (tmp_path / "io" / "visual_in").mkdir(parents=True, exist_ok=True)
    (tmp_path / "io" / "visual_out").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _job_dir(root: Path, job_id: str) -> Path:
    d = root / "io" / "visual_in" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_dir(root: Path, job_id: str) -> Path:
    d = root / "io" / "visual_out" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _shot_payload(job) -> dict[str, Any]:
    payload = {
        "shot_id": job.shot_id,
        "scene_id": job.scene_id,
        "duration_sec": float(job.duration_s),
        "prompt": job.prompt,
        "character": job.character or "",
        "refs": list(job.refs),
        "env_prompt": job.prompt,
        "mask_prompt": job.mask_prompt or "",
        "insert_prompt": job.insert_prompt or "",
        "handoff_from": job.handoff_from or "",
    }
    return payload


def _write_shotlist(job_dir: Path, shots: list[dict[str, Any]]) -> None:
    (job_dir / "shotlist.json").write_text(
        json.dumps({"shots": shots}, indent=2), encoding="utf-8"
    )


# ------------------------------------------------------------------
# Planner-only gates
# ------------------------------------------------------------------


def test_planner_covers_full_3_minute_runtime():
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)
    assert isinstance(result, PlannerResult)
    assert result.target_runtime_s == pytest.approx(180.0)
    assert result.total_duration_s >= 180.0, (
        f"planner covered only {result.total_duration_s:.1f}s of 180s; "
        f"scenes_covered={result.scenes_covered}"
    )


def test_planner_emits_enough_jobs_to_avoid_stagnation():
    """Day 11 'no stagnation' gate, planner-level.

    A 3-min scene at ~6 s per beat should produce at least 20 jobs.
    Fewer than that means the planner is coalescing shots and the
    scene will visually stagnate regardless of backend fidelity.
    """
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)
    assert len(result.jobs) >= 20, (
        f"planner emitted only {len(result.jobs)} jobs for a 3-min scene; "
        f"expected >= 20 to avoid stagnation"
    )


def test_planner_has_at_least_four_distinct_backends():
    """Day 11 'no duplicate stills' diversity floor.

    A continuous scene must cycle through at least four Day 1-7
    backends.  One backend for 3 min is exactly the stagnation trap
    Day 11 is built to catch.
    """
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)
    distinct = {j.backend for j in result.jobs}
    assert len(distinct) >= 4, (
        f"scene used only {len(distinct)} backend(s): {distinct}"
    )


def test_planner_respects_nonrepetition_window_on_3min_scene():
    """No ``(backend, prompt_hash)`` tuple may appear twice inside the
    sliding non-repetition window.  This is the core "no duplicate
    stills" invariant."""
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)
    window = result.repetition_window
    assert window >= 1
    seen: list[tuple[str, str]] = []
    for job in result.jobs:
        key = (job.backend, job.prompt_hash)
        recent = seen[-window:]
        assert key not in recent, (
            f"duplicate (backend, prompt_hash) within window {window}: "
            f"{key} seen again at {job.shot_id}"
        )
        seen.append(key)


def test_planner_clamps_motion_and_loop_to_C4_on_3min_scene():
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)
    for job in result.jobs:
        if job.backend in ("ltx_motion", "wan21_loop"):
            assert job.duration_s <= 10.0 + 1e-6, (
                f"{job.shot_id} {job.backend} duration {job.duration_s:.2f}s "
                f"breaches C4 10 s cap"
            )


# ------------------------------------------------------------------
# Wall-clock ceiling gate
# ------------------------------------------------------------------


def test_3min_scene_projected_real_wall_clock_under_45_min():
    """Day 11 ROADMAP bar: real-mode wall-clock projection must fit
    under 45 minutes for a 3-min continuous scene.

    This is a projection, not a measurement -- it uses per-backend
    point estimates from ``otr_v2.visual.wall_clock``.  The Day 14
    overnight dry run will close this loop with actual wall-clock data.
    """
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)
    est = estimate(result.jobs, mode="real")
    assert est.total_s <= DAY_11_WALL_CLOCK_CEILING_S, (
        f"projected real wall clock {est.total_s / 60.0:.1f} min > "
        f"{DAY_11_WALL_CLOCK_CEILING_S / 60.0:.0f} min ceiling; "
        f"breakdown: {est.to_dict()!r}"
    )


def test_3min_scene_stub_wall_clock_under_1min_ceiling():
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)
    est = estimate(result.jobs, mode="stub")
    assert est.total_s <= DAY_11_STUB_CEILING_S, (
        f"projected stub wall clock {est.total_s:.1f}s > "
        f"{DAY_11_STUB_CEILING_S:.0f}s ceiling"
    )


# ------------------------------------------------------------------
# End-to-end stub-mode execution gate
# ------------------------------------------------------------------


def test_3min_scene_stub_execution_finishes_in_under_60s(stub_env, scene_root):
    """End-to-end stub pass: plan the 3-min scene, dispatch every backend
    under stub mode, apply VHS post-processor, assert total wall clock
    stayed under 60 s.  This is the concrete Day 11 CI gate -- if stub
    mode itself takes more than a minute for a 3-min scene, the harness
    has a regression (most likely a subprocess / disk I/O leak)."""
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)

    job_id = "three_min_canary"
    job_dir = _job_dir(scene_root, job_id)
    out_dir = _out_dir(scene_root, job_id)

    # Group by backend so each backend sees only its own shots.
    grouped: dict[str, list] = {}
    for j in result.jobs:
        grouped.setdefault(j.backend, []).append(j)

    start = time.monotonic()
    for backend_name, group in grouped.items():
        shots = [_shot_payload(j) for j in group]
        _write_shotlist(job_dir, shots)
        resolve(backend_name).run(job_dir)
    vhs_mod.apply_vhs_to_job_dir(out_dir, force_stub=True)
    elapsed = time.monotonic() - start

    assert elapsed <= 60.0, (
        f"stub-mode 3-min scene took {elapsed:.1f}s (>60s ceiling); "
        f"jobs={len(result.jobs)}"
    )


def test_3min_scene_no_zero_byte_outputs(stub_env, scene_root):
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)

    job_id = "three_min_nobytes"
    job_dir = _job_dir(scene_root, job_id)
    out_dir = _out_dir(scene_root, job_id)

    grouped: dict[str, list] = {}
    for j in result.jobs:
        grouped.setdefault(j.backend, []).append(j)

    for backend_name, group in grouped.items():
        shots = [_shot_payload(j) for j in group]
        _write_shotlist(job_dir, shots)
        resolve(backend_name).run(job_dir)

    zero = [
        str(p.relative_to(out_dir))
        for p in out_dir.rglob("*")
        if p.is_file() and p.stat().st_size == 0
    ]
    assert not zero, f"zero-byte outputs: {zero}"


def test_3min_scene_emits_expected_still_and_video_mix(stub_env, scene_root):
    """Stagnation gate, execution-level: the scene must produce at least
    one of each core artifact type (render.png, keyframe.png, motion.mp4,
    loop.mp4)."""
    outline = _three_minute_cockpit_outline()
    result = plan_episode(outline)

    job_id = "three_min_mix"
    job_dir = _job_dir(scene_root, job_id)
    out_dir = _out_dir(scene_root, job_id)

    grouped: dict[str, list] = {}
    for j in result.jobs:
        grouped.setdefault(j.backend, []).append(j)

    for backend_name, group in grouped.items():
        shots = [_shot_payload(j) for j in group]
        _write_shotlist(job_dir, shots)
        resolve(backend_name).run(job_dir)

    assert list(out_dir.rglob("render.png")), "no render.png emitted"
    assert list(out_dir.rglob("keyframe.png")), "no keyframe.png emitted"
    assert list(out_dir.rglob("motion.mp4")), "no motion.mp4 emitted"
    assert list(out_dir.rglob("loop.mp4")), "no loop.mp4 emitted"
