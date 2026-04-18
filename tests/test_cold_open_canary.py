"""
Day 10 -- Cold-open canary: full Stage 1->7 pass in stub mode.

Exercises the full pipeline end-to-end without GPU weights:
    outline -> planner -> per-backend job dirs -> each Day 1-7 backend
    runs in stub mode -> VHS post-processor aggregates motion/loop
    clips -> final gate checks STATUS=READY for every backend, every
    shot produced its expected file(s), no zero-byte outputs, and
    vhs_postproc_summary.json is present.

This is the Day 10 canary from ROADMAP.md: "SCENE 01 -- Cockpit, Baba
boots up the radio."  It validates the planner contract against the
real backend registry, not a mock.  Real weights on disk are NOT a
prerequisite -- every backend's stub mode is the unit of work.

Torch is not imported here.  The subprocess pattern is not invoked --
we call backends directly so this test stays deterministic and fast.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from otr_v2.hyworld import planner as planner_mod
from otr_v2.hyworld.backends import list_backends, resolve
from otr_v2.hyworld.postproc import vhs as vhs_mod


# ------------------------------------------------------------------
# Scene 01 outline: "Cockpit, Baba boots up the radio."
# ------------------------------------------------------------------


def _scene_01_outline() -> dict:
    """Scene 01 outline that exercises every Day 1-7 backend.

    Beat roster:
      b01 establishing  -> flux_anchor   (wide cockpit at dawn)
      b02 close_up      -> pulid_portrait (Baba in pilot seat; refs present)
      b03 keyframe      -> flux_keyframe  (interior scene anchor)
      b04 motion        -> ltx_motion     (camera push-in)
      b05 loop          -> wan21_loop     (ambient console lights)
      b06 insert        -> florence2_sdxl_comp (HUD overlay)
    """
    return {
        "title": "SIGNAL LOST -- SCENE 01 -- COLD OPEN",
        "runtime_s": 48.0,
        "scenes": [
            {
                "scene_id": "scene_01",
                "location": "Cockpit",
                "characters": ["BABA"],
                "beats": [
                    {
                        "beat_id": "b01",
                        "kind": "establishing",
                        "prompt": "wide cockpit at dawn, analog switches "
                                  "and a warm amber radio glow",
                        "env_prompt": "derelict starship cockpit interior, "
                                      "dawn light through cracked viewport",
                        "camera": "wide establishing, slight dutch tilt",
                        "duration_s": 6.0,
                    },
                    {
                        "beat_id": "b02",
                        "kind": "close_up",
                        "prompt": "Baba leans toward the radio dial, "
                                  "eyes tracking the needle",
                        "character": "BABA",
                        "refs": [
                            "refs/baba_01.png",
                            "refs/baba_02.png",
                        ],
                        "env_prompt": "cockpit interior, warm amber glow",
                        "camera": "medium close-up, lens ~50mm",
                        "duration_s": 8.0,
                    },
                    {
                        "beat_id": "b03",
                        "kind": "keyframe",
                        "prompt": "the console comes alive: oscilloscope "
                                  "sine traces, brass toggles reflecting "
                                  "lamp light",
                        "env_prompt": "cockpit console detail, retrofuturist",
                        "camera": "over-shoulder onto console",
                        "duration_s": 6.0,
                    },
                    {
                        "beat_id": "b04",
                        "kind": "motion",
                        "prompt": "slow push-in on the radio dial as a "
                                  "carrier tone rises",
                        "motion_prompt": "subtle push-in, no cuts, 24fps",
                        "env_prompt": "same cockpit, same lighting",
                        "camera": "dolly forward",
                        "duration_s": 8.0,
                    },
                    {
                        "beat_id": "b05",
                        "kind": "loop",
                        "prompt": "ambient console lights flicker gently",
                        "loop_prompt": "seamless loop, subtle cycling motion, "
                                       "24fps",
                        "env_prompt": "same cockpit, same lighting",
                        "camera": "static",
                        "duration_s": 10.0,
                    },
                    {
                        "beat_id": "b06",
                        "kind": "insert",
                        "prompt": "HUD overlay with signal strength bars "
                                  "and a frequency readout",
                        "mask_prompt": "cockpit viewport frame",
                        "insert_prompt": "retro amber HUD bars and "
                                         "frequency readout 107.3 MHz",
                        "env_prompt": "cockpit viewport",
                        "camera": "on-axis to viewport",
                        "duration_s": 4.0,
                    },
                ],
            }
        ],
    }


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


# Each Day 1-7 backend: expected stub-envvar, expected per-shot files.
_BACKEND_MATRIX: dict[str, dict] = {
    "flux_anchor": {
        "stub_env": "OTR_FLUX_STUB",
        "outputs": ["render.png", "meta.json"],
    },
    "pulid_portrait": {
        "stub_env": "OTR_PULID_STUB",
        "outputs": ["render.png", "meta.json"],
    },
    "flux_keyframe": {
        "stub_env": "OTR_FLUX_KEYFRAME_STUB",
        "outputs": ["keyframe.png", "depth.png", "meta.json"],
    },
    "ltx_motion": {
        "stub_env": "OTR_LTX_STUB",
        "outputs": ["motion.mp4", "meta.json"],
    },
    "wan21_loop": {
        "stub_env": "OTR_WAN_STUB",
        "outputs": ["loop.mp4", "meta.json"],
    },
    "florence2_sdxl_comp": {
        "stub_env": "OTR_FLORENCE_STUB",
        "outputs": ["composite.png", "mask.png", "meta.json"],
    },
}


def _write_shotlist(job_dir: Path, shots: list[dict]) -> None:
    """Write the backend-flavoured shotlist.json the bridge contract expects."""
    payload = {"shots": shots}
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "shotlist.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def _out_dir_for(otr_root: Path, job_id: str) -> Path:
    """Mirror Backend.out_dir_for for ad-hoc canary runs."""
    return otr_root / "io" / "hyworld_out" / job_id


def _job_dir_for(otr_root: Path, job_id: str) -> Path:
    return otr_root / "io" / "hyworld_in" / job_id


def _all_stub_env() -> dict:
    """Every stub envvar set at once -- backends honour their own."""
    return {m["stub_env"]: "1" for m in _BACKEND_MATRIX.values()}


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


@pytest.fixture
def stub_env(monkeypatch):
    """Force every Day 1-7 backend into stub mode for the whole test."""
    for k, v in _all_stub_env().items():
        monkeypatch.setenv(k, v)
    yield


@pytest.fixture
def canary_root(tmp_path):
    """otr_root/ with io/hyworld_in and io/hyworld_out ready to use."""
    (tmp_path / "io" / "hyworld_in").mkdir(parents=True, exist_ok=True)
    (tmp_path / "io" / "hyworld_out").mkdir(parents=True, exist_ok=True)
    return tmp_path


# ------------------------------------------------------------------
# Preconditions: planner + backend roster contract
# ------------------------------------------------------------------


def test_planner_module_importable_torch_free():
    # Planner must stay pure stdlib so the canary can import it without
    # pulling diffusers/torch on a GPU-less CI host.
    import otr_v2.hyworld.planner  # noqa: F401
    assert "torch" not in sys.modules or os.environ.get("OTR_ALLOW_TORCH") == "1"


def test_all_seven_backends_registered():
    # Day 1-7 roster frozen; Day 8-14 add none.  Day 10 canary depends
    # on this exact list being dispatchable.
    expected = {
        "placeholder_test",
        "flux_anchor",
        "pulid_portrait",
        "flux_keyframe",
        "ltx_motion",
        "wan21_loop",
        "florence2_sdxl_comp",
    }
    assert expected.issubset(set(list_backends()))


def test_scene_01_outline_well_formed():
    outline = _scene_01_outline()
    assert outline["runtime_s"] == 48.0
    beats = outline["scenes"][0]["beats"]
    kinds = [b["kind"] for b in beats]
    # Every kind family exercised exactly once.
    assert sorted(kinds) == [
        "close_up", "establishing", "insert", "keyframe",
        "loop", "motion",
    ]


# ------------------------------------------------------------------
# Planner pass: outline -> jobs
# ------------------------------------------------------------------


def test_planner_covers_scene_01_runtime():
    outline = _scene_01_outline()
    result = planner_mod.plan_episode(outline)

    # Runtime coverage: planned total is at least the target.
    assert result.total_duration_s >= result.target_runtime_s
    # Sanity: every job names a registered backend.
    registered = set(list_backends())
    for job in result.jobs:
        assert job.backend in registered


def test_planner_emits_every_expected_backend_for_scene_01():
    outline = _scene_01_outline()
    result = planner_mod.plan_episode(outline)

    emitted = {job.backend for job in result.jobs}
    # Every Day 1-7 non-test backend should be emitted exactly from
    # this outline's beat roster (placeholder_test is never emitted
    # by kind-inference -- it's test-only).
    expected = {
        "flux_anchor", "pulid_portrait", "flux_keyframe",
        "ltx_motion", "wan21_loop", "florence2_sdxl_comp",
    }
    assert expected.issubset(emitted), (
        f"planner missed backends: {expected - emitted}"
    )


def test_planner_respects_C4_on_motion_and_loop_in_scene_01():
    outline = _scene_01_outline()
    result = planner_mod.plan_episode(outline)
    for job in result.jobs:
        if job.backend in ("ltx_motion", "wan21_loop"):
            assert job.duration_s <= planner_mod.MAX_MOTION_DURATION_S, (
                f"C4 violated by {job.shot_id}: {job.duration_s}"
            )


# ------------------------------------------------------------------
# Per-backend stub-mode pass
# ------------------------------------------------------------------


def _planner_jobs_by_backend(outline: dict) -> dict[str, list]:
    result = planner_mod.plan_episode(outline)
    grouped: dict[str, list] = {}
    for job in result.jobs:
        grouped.setdefault(job.backend, []).append(job)
    return grouped


def _shot_payload_for(job) -> dict:
    """Expand a PlannerJob into a shotlist shot dict the backends expect.

    Each Day 1-7 backend reads a superset schema; we populate what each
    backend actually looks up and leave the rest alone.
    """
    shot = {
        "shot_id": job.shot_id,
        "env_prompt": job.prompt,
        "camera": "",
        "duration_sec": job.duration_s,
    }
    if job.character:
        shot["character"] = job.character
    if job.refs:
        shot["refs"] = list(job.refs)
    if job.mask_prompt:
        shot["mask_prompt"] = job.mask_prompt
    if job.insert_prompt:
        shot["insert_prompt"] = job.insert_prompt
    if job.backend == "ltx_motion":
        shot["motion_prompt"] = job.prompt
    if job.backend == "wan21_loop":
        shot["loop_prompt"] = job.prompt
    return shot


@pytest.mark.parametrize("backend_name", sorted(_BACKEND_MATRIX.keys()))
def test_backend_stub_pass_for_scene_01(
    backend_name, stub_env, canary_root, monkeypatch,
):
    """Each Day 1-7 backend runs Scene 01's relevant shots in stub mode.

    Asserts:
      1. STATUS.json ends with READY (no stub-mode error path).
      2. Every shot gets its expected per-file roster.
      3. No zero-byte files (deterministic stubs must emit real bytes).
    """
    outline = _scene_01_outline()
    grouped = _planner_jobs_by_backend(outline)
    jobs = grouped.get(backend_name, [])
    if not jobs:
        pytest.skip(f"scene 01 produced no {backend_name} jobs")

    # Each backend gets its own job_id so outputs don't cross-pollute.
    job_id = f"canary_{backend_name}"
    job_dir = _job_dir_for(canary_root, job_id)
    out_dir = _out_dir_for(canary_root, job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    shots = [_shot_payload_for(j) for j in jobs]
    _write_shotlist(job_dir, shots)

    backend = resolve(backend_name)
    backend.run(job_dir)

    status_path = out_dir / "STATUS.json"
    assert status_path.exists(), (
        f"{backend_name}: STATUS.json missing at {status_path}"
    )
    status = json.loads(status_path.read_text(encoding="utf-8"))
    assert status.get("status") == "READY", (
        f"{backend_name}: status != READY, got {status!r}"
    )

    expected_files = _BACKEND_MATRIX[backend_name]["outputs"]
    for shot in shots:
        shot_dir = out_dir / shot["shot_id"]
        assert shot_dir.is_dir(), (
            f"{backend_name}/{shot['shot_id']}: shot dir missing"
        )
        for fname in expected_files:
            fpath = shot_dir / fname
            assert fpath.exists(), (
                f"{backend_name}/{shot['shot_id']}/{fname}: missing"
            )
            assert fpath.stat().st_size > 0, (
                f"{backend_name}/{shot['shot_id']}/{fname}: zero bytes"
            )


# ------------------------------------------------------------------
# VHS post-processor pass over the combined canary outputs
# ------------------------------------------------------------------


def test_vhs_postproc_over_full_canary_run(stub_env, canary_root):
    """Run every backend once then apply VHS postproc to the combined
    motion/loop outputs.

    Asserts:
      1. At least one motion.mp4 and one loop.mp4 was emitted.
      2. vhs.apply_vhs_to_job_dir succeeds and writes a summary file.
      3. Every video clip produced a *_vhs.mp4 sibling in stub copy mode.
    """
    outline = _scene_01_outline()
    grouped = _planner_jobs_by_backend(outline)

    # Combine all backends into one shared job_id so VHS sees a single
    # out_dir containing every shot.
    job_id = "canary_full"
    job_dir_shared = _job_dir_for(canary_root, job_id)
    out_dir_shared = _out_dir_for(canary_root, job_id)
    out_dir_shared.mkdir(parents=True, exist_ok=True)

    for backend_name, jobs in grouped.items():
        if backend_name not in _BACKEND_MATRIX:
            # placeholder_test isn't produced by scene_01 but guard anyway.
            continue
        # Per-backend subdir so each backend's load_shotlist finds its
        # own shotlist; outputs all land under the shared out_dir via
        # out_dir_for's otr_root/io/hyworld_out/<job_id> convention.
        per_backend_job_id = f"{job_id}"
        # Each backend writes into the SAME shared out_dir using its
        # own shotlist.  To avoid each backend overwriting STATUS.json,
        # we snapshot the shotlist per backend-call and let each one
        # stamp the most recent status.
        shots = [_shot_payload_for(j) for j in jobs]
        _write_shotlist(job_dir_shared, shots)
        backend = resolve(backend_name)
        backend.run(job_dir_shared)
        # Confirm this backend's STATUS is READY before moving on.
        status = json.loads(
            (out_dir_shared / "STATUS.json").read_text(encoding="utf-8")
        )
        assert status.get("status") == "READY", (
            f"{backend_name}: {status!r}"
        )

    # Confirm the canary actually produced video clips to VHS over.
    motion_clips = list(out_dir_shared.rglob("motion.mp4"))
    loop_clips = list(out_dir_shared.rglob("loop.mp4"))
    assert motion_clips, "scene 01 produced no motion.mp4"
    assert loop_clips, "scene 01 produced no loop.mp4"

    # Also confirm still roster: at least one render.png and keyframe.png
    # somewhere under the shared out_dir.
    assert list(out_dir_shared.rglob("render.png")), \
        "scene 01 produced no render.png"
    assert list(out_dir_shared.rglob("keyframe.png")), \
        "scene 01 produced no keyframe.png"
    assert list(out_dir_shared.rglob("composite.png")), \
        "scene 01 produced no composite.png"

    # VHS postproc: force stub mode so ffmpeg isn't required.
    summary = vhs_mod.apply_vhs_to_job_dir(
        out_dir_shared,
        force_stub=True,
    )
    assert (out_dir_shared / "vhs_postproc_summary.json").exists(), \
        "VHS summary missing"
    assert summary.get("count", -1) >= 2, (
        f"VHS count < 2 clips: {summary!r}"
    )

    # Every video clip should have a *_vhs.mp4 sibling.
    for src in motion_clips + loop_clips:
        vhs_out = src.with_name(src.stem + "_vhs.mp4")
        assert vhs_out.exists(), f"missing VHS pair: {vhs_out}"
        assert vhs_out.stat().st_size > 0, f"zero-byte VHS: {vhs_out}"


# ------------------------------------------------------------------
# End-to-end gate: the "no black frames, audio aligned" Day 10 ROADMAP bar
# ------------------------------------------------------------------


def test_no_zero_byte_outputs_after_full_canary(stub_env, canary_root):
    """Day 10 gate: after a full canary pass, no output anywhere in
    io/hyworld_out/ is zero bytes.  Stand-in for the 'no black frames'
    check since we can't decode pixels in a torch-free test."""
    outline = _scene_01_outline()
    grouped = _planner_jobs_by_backend(outline)

    job_id = "canary_nobytes"
    job_dir = _job_dir_for(canary_root, job_id)
    out_dir = _out_dir_for(canary_root, job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    for backend_name, jobs in grouped.items():
        if backend_name not in _BACKEND_MATRIX:
            continue
        shots = [_shot_payload_for(j) for j in jobs]
        _write_shotlist(job_dir, shots)
        resolve(backend_name).run(job_dir)

    zero_byte = []
    for path in out_dir.rglob("*"):
        if path.is_file() and path.stat().st_size == 0:
            zero_byte.append(str(path.relative_to(out_dir)))
    assert not zero_byte, f"zero-byte outputs: {zero_byte}"


def test_canary_is_deterministic_across_two_runs(stub_env, tmp_path):
    """Same outline + same env should produce byte-identical stub
    outputs across two back-to-back runs under the *same* absolute
    paths.  Validates the 'no stagnation, no duplicate stills'
    framework (the opposite failure -- nondeterminism -- breaks the
    non-repetition window).

    Note: several backends (flux_keyframe, wan21_loop, etc.) hash
    on the absolute anchor path in stub mode for layout-lock
    invariance, so two runs under different tmp roots would
    legitimately differ.  We instead run twice under the same root,
    nuking outputs between passes, to prove the backends are
    themselves deterministic rather than path-agnostic.
    """
    import shutil

    outline = _scene_01_outline()
    jobs = sorted(
        planner_mod.plan_episode(outline).jobs,
        key=lambda j: j.shot_id,
    )

    root = tmp_path / "canary_det_shared"
    (root / "io" / "hyworld_in").mkdir(parents=True, exist_ok=True)
    (root / "io" / "hyworld_out").mkdir(parents=True, exist_ok=True)

    def _run_once() -> dict[str, bytes]:
        job_id = "canary_det"
        job_dir = _job_dir_for(root, job_id)
        out_dir = _out_dir_for(root, job_id)
        # Nuke any prior outputs so the second pass is a clean redo.
        if job_dir.exists():
            shutil.rmtree(job_dir)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        grouped: dict[str, list] = {}
        for j in jobs:
            grouped.setdefault(j.backend, []).append(j)

        for backend_name, group in grouped.items():
            if backend_name not in _BACKEND_MATRIX:
                continue
            shots = [_shot_payload_for(j) for j in group]
            _write_shotlist(job_dir, shots)
            resolve(backend_name).run(job_dir)

        # Collect every produced file (except STATUS/meta which carry
        # per-run reason strings and timestamps that can vary).
        snapshot: dict[str, bytes] = {}
        for path in sorted(out_dir.rglob("*")):
            if not path.is_file():
                continue
            name = path.name
            if name in ("STATUS.json", "meta.json"):
                continue
            snapshot[str(path.relative_to(out_dir))] = path.read_bytes()
        return snapshot

    snap_a = _run_once()
    snap_b = _run_once()

    assert set(snap_a) == set(snap_b), (
        f"file roster differs: a-b={set(snap_a) - set(snap_b)} "
        f"b-a={set(snap_b) - set(snap_a)}"
    )
    for rel, ba in snap_a.items():
        bb = snap_b[rel]
        assert ba == bb, f"non-deterministic output: {rel}"
