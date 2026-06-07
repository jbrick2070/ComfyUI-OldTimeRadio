"""A-S4 CPU tests -- the latentsync lipsync-overlay sidecar adapter (default-OFF).

LatentSync runs in an isolated cu128 Path-B sidecar; its library + worker are
absent in the pytest sandbox, so every test here exercises the COLD path: the
adapter is registered + dark + gated, fails closed without the flag / install,
takes + releases the AS-3 shared lease correctly, and its pure request/clip
helpers are right. The live load + the VRAM<=14.5 GB boundary are the GPU smoke
(operator), NOT covered here.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from nodes._otr_shared import gpu_residency as gr
from nodes._otr_shared import role_compat as rc
from nodes._otr_video_engines import registry as vreg
from nodes._otr_video_engines import schemas as sc
from nodes._otr_video_engines.eng_latentsync import LatentSyncEngine

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _eng():
    return vreg.get_engine("latentsync")


# --------------------------------------------------------------------------- #
# Registration + default-OFF / dark
# --------------------------------------------------------------------------- #
def test_latentsync_registered_and_dark():
    assert vreg.is_registered("latentsync")
    eng = _eng()
    assert isinstance(eng, LatentSyncEngine)
    assert eng.family == "lipsync_overlay" and eng.family in sc.FAMILIES
    assert eng.default_roles == ()                 # not a default for any role -> dark
    assert eng.requires_flag == "OTR_ENABLE_LATENTSYNC"
    assert eng.required_inputs == ("base_clip_ref", "audio_ref")
    assert eng.commercial_clean is False           # OpenRAIL++ weights (verify-at-build)
    # Appears in the FULL static dropdown (V-6) even though it is gated.
    assert "latentsync" in vreg.all_engine_names()


def test_registry_gates_latentsync_until_flag(monkeypatch):
    monkeypatch.delenv("OTR_ENABLE_LATENTSYNC", raising=False)
    for role in ("announcer_visual", "character_video"):
        with pytest.raises(vreg.EngineUnusable):
            vreg.assert_usable("latentsync", role)         # dark -> GATED_BY_FLAG
    monkeypatch.setenv("OTR_ENABLE_LATENTSYNC", "1")
    for role in ("announcer_visual", "character_video"):
        assert vreg.assert_usable("latentsync", role) == "latentsync"
    # A role it does NOT serve fails closed regardless of the flag.
    with pytest.raises(vreg.EngineUnusable):
        vreg.assert_usable("latentsync", "music_visual")


def test_latentsync_role_fit_lipsync_needs_base_and_audio():
    desc = {"engine_id": "latentsync", "roles": _eng().roles,
            "required_inputs": _eng().required_inputs}
    # Only the roles that can supply BOTH a base clip and the speech audio.
    assert rc.engine_fits_role(desc, "announcer_visual") is True
    assert rc.engine_fits_role(desc, "character_video") is True
    for role in ("music_visual", "scene_broll", "background_abstract"):
        assert rc.engine_fits_role(desc, role) is False
    # And the shared filter excludes it from a base-only role.
    descs = [desc]
    assert rc.filter_engines_for_role("scene_broll", descs) == []
    assert "latentsync" in rc.filter_engines_for_role("character_video", descs)


# --------------------------------------------------------------------------- #
# Fail-closed usability (no heavy import; CPU box)
# --------------------------------------------------------------------------- #
def test_assert_usable_fail_closed_flag_then_install(monkeypatch):
    eng = _eng()
    monkeypatch.delenv("OTR_ENABLE_LATENTSYNC", raising=False)
    with pytest.raises(vreg.EngineUnusable) as ei:
        eng.assert_usable(host_caps={}, profile={})
    assert ei.value.reason == vreg.EngineUsabilityReason.GATED_BY_FLAG
    # Flag set but the Path-B venv / worker absent -> MISSING_MODEL (still closed).
    monkeypatch.setenv("OTR_ENABLE_LATENTSYNC", "1")
    monkeypatch.setenv("OTR_LATENTSYNC_VENV", str(REPO_ROOT / "_no_such_py.exe"))
    monkeypatch.setenv("OTR_LATENTSYNC_WORKER", str(REPO_ROOT / "_no_such_worker.py"))
    with pytest.raises(vreg.EngineUnusable) as ei2:
        eng.assert_usable(host_caps={}, profile={})
    assert ei2.value.reason == vreg.EngineUsabilityReason.MISSING_MODEL


def test_load_fails_closed_without_install(monkeypatch):
    monkeypatch.setenv("OTR_LATENTSYNC_VENV", str(REPO_ROOT / "_no_such_py.exe"))
    monkeypatch.setenv("OTR_LATENTSYNC_WORKER", str(REPO_ROOT / "_no_such_worker.py"))
    eng = LatentSyncEngine()
    with pytest.raises(RuntimeError) as ei:
        eng.load()
    assert "not installed" in str(ei.value).lower()


# --------------------------------------------------------------------------- #
# AS-3 shared lease: prepare takes it; a load failure must not strand it
# --------------------------------------------------------------------------- #
def test_prepare_releases_lease_on_load_failure(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))     # isolate the default lockdir
    monkeypatch.setenv("OTR_LATENTSYNC_VENV", str(tmp_path / "_no_py.exe"))
    monkeypatch.setenv("OTR_LATENTSYNC_WORKER", str(tmp_path / "_no_worker.py"))
    eng = LatentSyncEngine()
    assert not gr.is_held()                       # clean to start
    with pytest.raises(RuntimeError):
        eng.prepare(host_caps={}, profile={}, session_ctx={})
    assert not gr.is_held()                       # lease released, never stranded


def test_unload_and_teardown_idempotent_no_worker(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    eng = LatentSyncEngine()
    eng.unload()                                  # no worker -> no crash
    eng.unload()                                  # idempotent
    eng.teardown(None)                            # no lease, no worker -> no raise
    eng.teardown({"lease": None})


# --------------------------------------------------------------------------- #
# Pure helpers (no worker, no heavy import)
# --------------------------------------------------------------------------- #
def test_build_worker_request_from_dict():
    out = _eng()._build_worker_request(
        {"shot_id": "s1", "base_clip_ref": "C:/base.mp4",
         "audio_ref": {"path": "C:/voice.wav", "content_hash": "abc"},
         "timing": {"target_frame_count": 50},
         "seed_bundle": {"request_seed": 7}},
        "C:/out.mp4")
    assert out == {
        "base_clip": "C:/base.mp4", "audio_path": "C:/voice.wav",
        "out_path": "C:/out.mp4", "fps": 25, "target_frame_count": 50,
        "audio_sample_rate": 16000, "seed": 7,
    }


def test_build_worker_request_from_videorequest():
    vr = sc.VideoRequest(
        request_id="r", shot_id="s2", role="announcer_visual",
        family_hint="lipsync_overlay", profile_id="p",
        base_clip_ref="C:/b.mp4", audio_ref=sc.AudioRef(path="C:/a.wav"),
        timing=sc.Timing(target_frame_count=33),
        canvas=sc.Canvas(w=832, h=480, fps=25),
        seed_bundle=sc.SeedBundle(request_seed=3),
    )
    out = _eng()._build_worker_request(vr, "C:/o.mp4")
    assert out["base_clip"] == "C:/b.mp4" and out["audio_path"] == "C:/a.wav"
    assert out["target_frame_count"] == 33 and out["seed"] == 3


def test_clip_from_worker_is_silent_bt709():
    clip = _eng()._clip_from_worker(
        {"ok": True, "out_path": "C:/o.mp4", "frame_count": 40}, {"shot_id": "s9"})
    assert clip["has_audio"] is False                  # V-1: mux owns audio
    assert clip["pixel_format"] == "yuv420p"
    assert clip["matrix"] == clip["transfer"] == clip["color_primaries"] == "bt709"
    assert clip["frame_count"] == 40 and clip["fps"] == 25
    assert clip["engine_id"] == "latentsync" and clip["family"] == "lipsync_overlay"
    assert clip["path"] == "C:/o.mp4" and clip["clip_id"] == "s9"


def test_ref_path_handles_str_dict_and_none():
    eng = _eng()
    assert eng._ref_path("C:/x.mp4") == "C:/x.mp4"
    assert eng._ref_path({"path": "C:/y.wav"}) == "C:/y.wav"
    assert eng._ref_path(None) == "" and eng._ref_path({}) == ""


# --------------------------------------------------------------------------- #
# Cold-import: the adapter + shared sidecar pull NO heavy lib (V-12)
# --------------------------------------------------------------------------- #
def test_cold_import_adapter_no_heavy_libs():
    code = (
        "import sys;"
        "import nodes._otr_video_engines.eng_latentsync;"
        "import nodes._otr_shared.sidecar;"
        "heavy=[m for m in ('torch','transformers','diffusers') if m in sys.modules];"
        "print('HEAVY', heavy);"
        "sys.exit(1 if heavy else 0)"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, f"heavy libs pulled at import:\n{r.stdout}\n{r.stderr}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
