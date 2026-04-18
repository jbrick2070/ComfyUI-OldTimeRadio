"""Day 13 gate: 20-minute episode dry run.

ROADMAP bar:
    "20-min episode dry run (stub mode, all 7 backends).
     No OOM, no zero-byte outputs, LHM poller captures expected
     sample count, wall-clock projection fits overnight budget."

This is a torch-free stub-mode test.  It drives the planner with a
20-minute outline spanning six scenes, dispatches every backend in
stub mode, runs the LHM poller (Day 13) against an injected fake
telemetry tree, and asserts the four Day 13 acceptance gates:

  1. Planner covers the full 1200-s runtime.
  2. Every shot produces a non-zero-byte artifact in
     ``io/hyworld_out/<job_id>/``.
  3. Every STATUS.json written by a backend ends in ``READY``
     (no ``ERROR``, no ``OOM``, no ``RUNNING`` left behind).
  4. The LHM poller captures >= 18 samples across the projected run
     and its summary shows no ceiling breach when the injected fake
     telemetry tree reports nominal hardware values.
  5. Wall-clock projection in real mode fits under an 8 h overnight
     budget once cold-load + VHS cost are folded in.

No torch / diffusers / numpy imports.  No audio imports.  C7 audio
byte-identical gate is unaffected.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pytest

from otr_v2.hyworld.backends import resolve
from otr_v2.hyworld.planner import PlannerResult, plan_episode
from otr_v2.hyworld.postproc import vhs as vhs_mod
from otr_v2.hyworld.wall_clock import estimate
from otr_v2.hyworld import lhm_monitor as lm


# ------------------------------------------------------------------
# Outline: 20-minute, six-scene episode
# ------------------------------------------------------------------


EPISODE_RUNTIME_S: float = 20 * 60.0  # 1200 s

# Day 13 overnight budget: 8 h of wall clock.  The estimator's
# point estimates are conservative upper bounds, so fitting under
# this means the overnight Task Scheduler run has head-room for
# normal variance.
DAY_13_REAL_CEILING_S: float = 8 * 3600.0  # 28 800 s

# Day 13 stub CI floor -- stub execution of a 20-min episode should
# complete in well under two minutes.  If it doesn't, something is
# leaking subprocesses or blocking on I/O.
DAY_13_STUB_CEILING_S: float = 120.0


def _twenty_minute_episode_outline() -> dict[str, Any]:
    """Six-scene, 20-minute outline exercising every Day 1-7 backend.

    Scene rotation will kick in (sum(beats) per scene ~= 50 s,
    runtime 1200 s).  That stress-tests the planner's rotate-from-top
    logic while giving the Day 13 dry run honest backend coverage.
    """
    return {
        "runtime_s": EPISODE_RUNTIME_S,
        "scenes": [
            {
                "scene_id": "scene_01_cockpit",
                "title": "Cockpit at dawn",
                "location": "Cockpit",
                "characters": ["BABA", "BOOEY"],
                "beats": [
                    {
                        "beat_id": "s01_b01",
                        "kind": "establishing",
                        "prompt": (
                            "wide establishing of cockpit at dawn, "
                            "retrofuture panels glowing amber"
                        ),
                    },
                    {
                        "beat_id": "s01_b02",
                        "kind": "close_up",
                        "character": "BABA",
                        "refs": ["baba_01.png", "baba_02.png"],
                        "prompt": (
                            "BABA close-up, hand on tuning dial, "
                            "weary focus"
                        ),
                    },
                    {
                        "beat_id": "s01_b03",
                        "kind": "keyframe",
                        "prompt": (
                            "mid-shot of radio console, needle crossing "
                            "107.3 MHz"
                        ),
                    },
                    {
                        "beat_id": "s01_b04",
                        "kind": "motion",
                        "prompt": "push-in on console, LEDs ticking upward",
                    },
                    {
                        "beat_id": "s01_b05",
                        "kind": "insert",
                        "prompt": "HUD overlay frequency readout",
                        "mask_prompt": "cockpit viewport frame",
                        "insert_prompt": "retro amber HUD bars, 107.3 MHz",
                    },
                ],
            },
            {
                "scene_id": "scene_02_corridor",
                "title": "Transit corridor",
                "location": "Corridor",
                "characters": ["BABA", "BOOEY"],
                "beats": [
                    {
                        "beat_id": "s02_b01",
                        "kind": "establishing",
                        "prompt": (
                            "narrow transit corridor, striped emergency "
                            "lights pulsing red"
                        ),
                    },
                    {
                        "beat_id": "s02_b02",
                        "kind": "two_shot",
                        "character": "BOOEY",
                        "refs": ["booey_01.png"],
                        "prompt": (
                            "BABA and BOOEY in profile, tense quick step, "
                            "footfalls echoing"
                        ),
                    },
                    {
                        "beat_id": "s02_b03",
                        "kind": "motion",
                        "prompt": (
                            "tracking shot down corridor, red light "
                            "cycling on metal walls"
                        ),
                    },
                    {
                        "beat_id": "s02_b04",
                        "kind": "ambient",
                        "prompt": "distant klaxon, coolant vapor seeping",
                    },
                    {
                        "beat_id": "s02_b05",
                        "kind": "keyframe",
                        "prompt": "sealed bulkhead door, amber warning stripe",
                    },
                ],
            },
            {
                "scene_id": "scene_03_engine_room",
                "title": "Engine room",
                "location": "Engine Room",
                "characters": ["BOOEY"],
                "beats": [
                    {
                        "beat_id": "s03_b01",
                        "kind": "establishing",
                        "prompt": (
                            "cavernous engine room, plasma conduits "
                            "humming blue"
                        ),
                    },
                    {
                        "beat_id": "s03_b02",
                        "kind": "close_up",
                        "character": "BOOEY",
                        "refs": ["booey_02.png"],
                        "prompt": (
                            "BOOEY close-up, sweat-lit face, reading "
                            "a flickering gauge"
                        ),
                    },
                    {
                        "beat_id": "s03_b03",
                        "kind": "keyframe",
                        "prompt": "close of the coolant gauge, needle redlining",
                    },
                    {
                        "beat_id": "s03_b04",
                        "kind": "motion",
                        "prompt": "slow dolly around the main reactor column",
                    },
                    {
                        "beat_id": "s03_b05",
                        "kind": "insert",
                        "prompt": "warning label overlay on console panel",
                        "mask_prompt": "console panel label plate",
                        "insert_prompt": "engraved metal warning label, amber glow",
                    },
                ],
            },
            {
                "scene_id": "scene_04_viewport",
                "title": "Observation viewport",
                "location": "Viewport",
                "characters": ["BABA"],
                "beats": [
                    {
                        "beat_id": "s04_b01",
                        "kind": "establishing",
                        "prompt": (
                            "curved viewport looking onto a slow "
                            "rotating gas giant"
                        ),
                    },
                    {
                        "beat_id": "s04_b02",
                        "kind": "keyframe",
                        "prompt": "star field through viewport, quiet drift",
                    },
                    {
                        "beat_id": "s04_b03",
                        "kind": "loop",
                        "prompt": (
                            "slow rotation of distant gas giant, "
                            "ambient stillness"
                        ),
                    },
                    {
                        "beat_id": "s04_b04",
                        "kind": "motion",
                        "prompt": "dolly right past observation bench",
                    },
                    {
                        "beat_id": "s04_b05",
                        "kind": "portrait",
                        "character": "BABA",
                        "refs": ["baba_01.png", "baba_03.png"],
                        "prompt": (
                            "BABA silhouetted against viewport, coffee "
                            "cooling in hand"
                        ),
                    },
                ],
            },
            {
                "scene_id": "scene_05_galley",
                "title": "Crew galley",
                "location": "Galley",
                "characters": ["BABA", "BOOEY"],
                "beats": [
                    {
                        "beat_id": "s05_b01",
                        "kind": "establishing",
                        "prompt": (
                            "cramped crew galley, lockers and a foldable "
                            "bench, tungsten lamp"
                        ),
                    },
                    {
                        "beat_id": "s05_b02",
                        "kind": "close_up",
                        "character": "BABA",
                        "refs": ["baba_02.png"],
                        "prompt": (
                            "BABA close-up, chewing ration bar, thousand-"
                            "yard stare"
                        ),
                    },
                    {
                        "beat_id": "s05_b03",
                        "kind": "keyframe",
                        "prompt": "steaming ration pack, paper label curling",
                    },
                    {
                        "beat_id": "s05_b04",
                        "kind": "loop",
                        "prompt": "steam curling from pack, ambient stillness",
                    },
                    {
                        "beat_id": "s05_b05",
                        "kind": "insert",
                        "prompt": "label detail overlay",
                        "mask_prompt": "paper label plate on ration pack",
                        "insert_prompt": (
                            "detailed ration pack label, issue stamp "
                            "and barcode"
                        ),
                    },
                ],
            },
            {
                "scene_id": "scene_06_airlock",
                "title": "Airlock cycle",
                "location": "Airlock",
                "characters": ["BOOEY"],
                "beats": [
                    {
                        "beat_id": "s06_b01",
                        "kind": "establishing",
                        "prompt": (
                            "airlock chamber, yellow and black hazard "
                            "stripes, gear mounted"
                        ),
                    },
                    {
                        "beat_id": "s06_b02",
                        "kind": "two_shot",
                        "character": "BOOEY",
                        "refs": ["booey_01.png", "booey_02.png"],
                        "prompt": (
                            "BOOEY checking suit seals, BABA at the "
                            "console behind"
                        ),
                    },
                    {
                        "beat_id": "s06_b03",
                        "kind": "motion",
                        "prompt": "outer hatch cycling open, slow reveal",
                    },
                    {
                        "beat_id": "s06_b04",
                        "kind": "ambient",
                        "prompt": (
                            "pressure equalising hiss, warning lamp "
                            "sweep"
                        ),
                    },
                    {
                        "beat_id": "s06_b05",
                        "kind": "keyframe",
                        "prompt": "mag-boot close-up, locking onto floor grate",
                    },
                ],
            },
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
    (tmp_path / "io" / "hyworld_in").mkdir(parents=True, exist_ok=True)
    (tmp_path / "io" / "hyworld_out").mkdir(parents=True, exist_ok=True)
    return tmp_path


def _job_dir(root: Path, job_id: str) -> Path:
    d = root / "io" / "hyworld_in" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _out_dir(root: Path, job_id: str) -> Path:
    d = root / "io" / "hyworld_out" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _shot_payload(job) -> dict[str, Any]:
    return {
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


def _write_shotlist(job_dir: Path, shots: list[dict[str, Any]]) -> None:
    (job_dir / "shotlist.json").write_text(
        json.dumps({"shots": shots}, indent=2), encoding="utf-8"
    )


def _dispatch_all(result: PlannerResult, job_dir: Path) -> None:
    """Group jobs by backend and run each backend once on its subset."""
    grouped: dict[str, list] = {}
    for j in result.jobs:
        grouped.setdefault(j.backend, []).append(j)
    for backend_name, group in grouped.items():
        shots = [_shot_payload(j) for j in group]
        _write_shotlist(job_dir, shots)
        resolve(backend_name).run(job_dir)


# ------------------------------------------------------------------
# Planner-level gates
# ------------------------------------------------------------------


def test_planner_covers_full_20_minute_runtime():
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    assert isinstance(result, PlannerResult)
    assert result.target_runtime_s == pytest.approx(EPISODE_RUNTIME_S)
    assert result.total_duration_s >= EPISODE_RUNTIME_S, (
        f"planner covered only {result.total_duration_s:.1f}s of "
        f"{EPISODE_RUNTIME_S:.0f}s; scenes_covered={result.scenes_covered}"
    )


def test_planner_uses_all_six_scenes():
    """20-min episode must visit every scene in the outline.

    Day 13 exists to catch runaway scene-rotation bugs where one
    scene consumes the entire target runtime.
    """
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    scenes_used = {j.scene_id for j in result.jobs}
    assert len(scenes_used) == 6, (
        f"expected all six scenes, got {scenes_used}"
    )


def test_planner_exercises_every_day1_to_day7_backend():
    """20-min episode must hit every backend at least once.

    Day 13 is the integration gate that catches a whole Stage
    dropping out of the rotation silently.
    """
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    backends_used = {j.backend for j in result.jobs}
    expected = {
        "flux_anchor",
        "pulid_portrait",
        "flux_keyframe",
        "ltx_motion",
        "wan21_loop",
        "florence2_sdxl_comp",
    }
    missing = expected - backends_used
    assert not missing, f"backends missing from 20-min rotation: {missing}"


def test_planner_emits_enough_jobs_for_20_min_runtime():
    """A 20-min episode at the default 6 s beat cadence should produce
    at least 150 jobs.  Fewer means the planner is coalescing shots."""
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    assert len(result.jobs) >= 150, (
        f"planner emitted only {len(result.jobs)} jobs for 20-min run; "
        f"expected >= 150"
    )


def test_planner_respects_nonrepetition_window_on_20min_episode():
    """Core non-repetition invariant still holds across 20 minutes."""
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    window = result.repetition_window
    seen: list[tuple[str, str]] = []
    for job in result.jobs:
        key = (job.backend, job.prompt_hash)
        recent = seen[-window:]
        assert key not in recent, (
            f"duplicate (backend, prompt_hash) within window {window}: "
            f"{key} at {job.shot_id}"
        )
        seen.append(key)


def test_planner_clamps_motion_and_loop_to_C4_on_20min_episode():
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    for job in result.jobs:
        if job.backend in ("ltx_motion", "wan21_loop"):
            assert job.duration_s <= 10.0 + 1e-6, (
                f"{job.shot_id} {job.backend} duration {job.duration_s:.2f}s "
                f"breaches C4 10 s cap"
            )


# ------------------------------------------------------------------
# Wall-clock projection gates
# ------------------------------------------------------------------


def test_20min_episode_projected_real_wall_clock_fits_overnight():
    """Day 13: projected real wall clock fits under an 8 h overnight
    budget, including cold loads and VHS postproc."""
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    est = estimate(result.jobs, mode="real")
    assert est.total_s <= DAY_13_REAL_CEILING_S, (
        f"projected real wall clock {est.total_s / 3600.0:.2f} h > "
        f"{DAY_13_REAL_CEILING_S / 3600.0:.0f} h overnight ceiling; "
        f"breakdown: {est.to_dict()!r}"
    )


def test_20min_episode_stub_projection_under_ci_floor():
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)
    est = estimate(result.jobs, mode="stub")
    assert est.total_s <= DAY_13_STUB_CEILING_S, (
        f"projected stub wall clock {est.total_s:.1f}s > "
        f"{DAY_13_STUB_CEILING_S:.0f}s CI floor"
    )


# ------------------------------------------------------------------
# End-to-end stub execution + STATUS.json gates
# ------------------------------------------------------------------


def test_20min_episode_stub_execution_finishes_in_under_ci_floor(
    stub_env, scene_root
):
    """Concrete Day 13 CI gate: dispatching every backend on every
    shot of the 20-min episode must finish in well under two minutes
    of wall clock in stub mode."""
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)

    job_id = "twenty_min_dryrun_ci"
    job_dir = _job_dir(scene_root, job_id)
    out_dir = _out_dir(scene_root, job_id)

    start = time.monotonic()
    _dispatch_all(result, job_dir)
    vhs_mod.apply_vhs_to_job_dir(out_dir, force_stub=True)
    elapsed = time.monotonic() - start

    assert elapsed <= DAY_13_STUB_CEILING_S, (
        f"stub-mode 20-min episode took {elapsed:.1f}s "
        f"(> {DAY_13_STUB_CEILING_S:.0f}s ceiling); jobs={len(result.jobs)}"
    )


def test_20min_episode_no_zero_byte_outputs(stub_env, scene_root):
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)

    job_id = "twenty_min_dryrun_bytes"
    job_dir = _job_dir(scene_root, job_id)
    out_dir = _out_dir(scene_root, job_id)

    _dispatch_all(result, job_dir)

    zero = [
        str(p.relative_to(out_dir))
        for p in out_dir.rglob("*")
        if p.is_file() and p.stat().st_size == 0
    ]
    assert not zero, f"zero-byte outputs across 20-min dry run: {zero}"


def test_20min_episode_status_files_all_ready(stub_env, scene_root):
    """No STATUS.json in the 20-min dry run may end in OOM or ERROR.

    This is the concrete Day 13 "no OOM" gate: if any backend writes
    a failure status, the overnight Task Scheduler run must flag the
    full run as failed.
    """
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)

    job_id = "twenty_min_dryrun_status"
    job_dir = _job_dir(scene_root, job_id)
    out_dir = _out_dir(scene_root, job_id)

    _dispatch_all(result, job_dir)

    status_paths = list(out_dir.rglob("STATUS.json"))
    assert status_paths, "no STATUS.json written by any backend"

    bad_status: list[tuple[str, str]] = []
    for sp in status_paths:
        payload = json.loads(sp.read_text(encoding="utf-8"))
        status = str(payload.get("status") or "")
        if status not in {"READY"}:
            bad_status.append((str(sp.relative_to(out_dir)), status))
    assert not bad_status, (
        f"non-READY STATUS.json files found: {bad_status}"
    )


def test_20min_episode_emits_expected_still_and_video_mix(
    stub_env, scene_root
):
    """Stagnation gate, execution-level: the 20-min dry run must
    produce at least one of every core artifact type."""
    outline = _twenty_minute_episode_outline()
    result = plan_episode(outline)

    job_id = "twenty_min_dryrun_mix"
    job_dir = _job_dir(scene_root, job_id)
    out_dir = _out_dir(scene_root, job_id)

    _dispatch_all(result, job_dir)

    assert list(out_dir.rglob("render.png")), "no render.png emitted"
    assert list(out_dir.rglob("keyframe.png")), "no keyframe.png emitted"
    assert list(out_dir.rglob("motion.mp4")), "no motion.mp4 emitted"
    assert list(out_dir.rglob("loop.mp4")), "no loop.mp4 emitted"


# ------------------------------------------------------------------
# LHM poller integration gate
# ------------------------------------------------------------------


def _nominal_fake_lhm_tree(
    gpu_temp_c: float = 68.0,
    vram_used_gb: float = 9.5,
    ram_used_gb: float = 18.0,
    cpu_temp_c: float = 55.0,
    vram_total_gb: float = 16.0,
    ram_total_gb: float = 32.0,
) -> dict[str, Any]:
    """Construct a minimal LHM JSON tree with one GPU and one CPU
    branch.  Mirrors the real LHM schema closely enough for the
    extractor walk, avoiding any network dependency in CI."""
    return {
        "Text": "Sensor",
        "Children": [
            {
                "Text": "NVIDIA GeForce RTX 5080 Laptop GPU",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {
                                "Text": "GPU Core",
                                "Value": f"{gpu_temp_c:.1f} C",
                            },
                        ],
                    },
                    {
                        "Text": "Memory",
                        "Children": [
                            {
                                "Text": "GPU Memory Used",
                                "Value": f"{vram_used_gb:.2f} GB",
                            },
                            {
                                "Text": "GPU Memory Total",
                                "Value": f"{vram_total_gb:.2f} GB",
                            },
                        ],
                    },
                ],
            },
            {
                "Text": "Intel Core Ultra 9 275HX",
                "Children": [
                    {
                        "Text": "Temperatures",
                        "Children": [
                            {
                                "Text": "CPU Package",
                                "Value": f"{cpu_temp_c:.1f} C",
                            },
                        ],
                    },
                ],
            },
            {
                "Text": "Generic Memory",
                "Children": [
                    {
                        "Text": "Memory",
                        "Children": [
                            {
                                "Text": "Memory Used",
                                "Value": f"{ram_used_gb:.2f} GB",
                            },
                            {
                                "Text": "Memory Available",
                                "Value": f"{ram_total_gb:.2f} GB",
                            },
                        ],
                    },
                ],
            },
        ],
    }


def _state_advancing_sleep(state: dict[str, float]):
    """sleep_fn that advances a shared monotonic clock by the sleep
    duration every call.  Combined with a monotonic_fn reading the
    same state dict, the poll loop runs on a deterministic fake clock.
    """
    def _sleep(s: float) -> None:
        state["t"] += float(s)
    return _sleep


def test_lhm_poller_captures_expected_samples_across_20min_run(tmp_path):
    """Day 13 poller gate: across the projected 20-minute run, the
    poller with a 60 s interval must capture at least 18 samples
    (one per minute, 20 total, minus up to two to absorb edge
    rounding)."""
    state = {"t": 0.0}
    tree = _nominal_fake_lhm_tree()

    out_ndjson = tmp_path / "lhm_20min.ndjson"

    samples = lm.poll_loop(
        out_ndjson,
        interval_s=60.0,
        duration_s=20 * 60.0,
        max_samples=64,
        fetcher=lambda _u, _t: json.dumps(tree).encode("utf-8"),
        sleep_fn=_state_advancing_sleep(state),
        monotonic_fn=lambda: state["t"],
        unix_fn=lambda: state["t"] + 1_700_000_000.0,
    )

    assert 18 <= len(samples) <= 22, (
        f"expected ~20 samples across a 20-min run at 60 s interval, "
        f"got {len(samples)}"
    )

    # NDJSON log must persist one line per captured sample.
    lines = [
        ln for ln in out_ndjson.read_text(encoding="utf-8").splitlines() if ln
    ]
    assert len(lines) == len(samples)


def test_lhm_summary_flags_no_ceiling_breach_for_nominal_hardware(tmp_path):
    """With a nominal fake hardware tree (68 C / 9.5 GB VRAM / 18 GB RAM),
    none of the three Day 13 ceilings should trip."""
    state = {"t": 0.0}
    tree = _nominal_fake_lhm_tree(
        gpu_temp_c=68.0, vram_used_gb=9.5, ram_used_gb=18.0
    )
    out_ndjson = tmp_path / "lhm_nominal.ndjson"

    samples = lm.poll_loop(
        out_ndjson,
        interval_s=60.0,
        duration_s=20 * 60.0,
        max_samples=64,
        fetcher=lambda _u, _t: json.dumps(tree).encode("utf-8"),
        sleep_fn=_state_advancing_sleep(state),
        monotonic_fn=lambda: state["t"],
        unix_fn=lambda: state["t"],
    )
    summary = lm.summarize(samples)

    assert summary.n_samples >= 18
    assert summary.n_unreachable == 0
    assert summary.vram_ceiling_breached is False
    assert summary.ram_ceiling_breached is False
    assert summary.gpu_temp_ceiling_breached is False


def test_lhm_summary_trips_vram_breach_on_thrash_tree(tmp_path):
    """Inverse gate: a VRAM value above VRAM_CEILING_GB must trip the
    breach flag so the overnight run reports a failed dry run
    rather than silently logging."""
    state = {"t": 0.0}
    tree = _nominal_fake_lhm_tree(
        gpu_temp_c=70.0,
        vram_used_gb=lm.VRAM_CEILING_GB + 0.5,
        ram_used_gb=18.0,
    )
    out_ndjson = tmp_path / "lhm_thrash.ndjson"

    samples = lm.poll_loop(
        out_ndjson,
        interval_s=60.0,
        duration_s=5 * 60.0,  # short; we only need one breach sample
        max_samples=8,
        fetcher=lambda _u, _t: json.dumps(tree).encode("utf-8"),
        sleep_fn=_state_advancing_sleep(state),
        monotonic_fn=lambda: state["t"],
        unix_fn=lambda: state["t"],
    )
    summary = lm.summarize(samples)

    assert summary.vram_ceiling_breached is True
    assert any("VRAM ceiling" in n for n in summary.notes)
