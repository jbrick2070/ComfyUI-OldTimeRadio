"""CPU tests for the operator GPU-smoke probe (scripts/otr_video_gpu_smoke).

The probe is a readiness + scaffold tool: the live forward is the operator's
NotImplementedError GPU slice, so the probe must NEVER report a render as a pass.
These tests pin the CPU-checkable surface -- the readiness verdict reflects the
real flag / install / assert_usable state, the real fallback chain resolves and
converges to the radio floor, the request build runs, and a render attempt is
reported honestly (gpu_slice_not_implemented / not_ready), never as 'rendered'.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
SMOKE_SRC = REPO_ROOT / "scripts" / "otr_video_gpu_smoke.py"


def _load():
    spec = importlib.util.spec_from_file_location("otr_video_gpu_smoke", SMOKE_SRC)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SMK = _load()
_FLAGS = ["OTR_ENABLE_HUMO", "OTR_ENABLE_LTX_VIDEO",
          "OTR_ENABLE_WAN_I2V"]


@pytest.fixture(autouse=True)
def _flags_off(monkeypatch):
    for f in _FLAGS:
        monkeypatch.delenv(f, raising=False)


def _check(report, name):
    return next(c for c in report["checks"] if c["name"] == name)


# --------------------------------------------------------------------------- #
# readiness verdict reflects reality (flag off -> NOT READY)
# --------------------------------------------------------------------------- #
def test_humo_report_not_ready_without_deps():
    # No flag in the smoke verdict (C5 -- "registry IS the menu"): the flag_set
    # ready-assert is GONE. humo is NOT READY in the sandbox because its deps (the
    # HuMo wrapper + ckpts) are absent, not because of any opt-in flag -- the
    # verdict reflects registration + dep-pilot import-clean + assert_usable only.
    r = SMK.run_smoke("humo")
    assert r["engine"] == "humo" and r["ready"] is False
    assert _check(r, "registered")["ok"] is True
    assert all(c["name"] != "flag_set" for c in r["checks"])   # no flag gate


def test_assert_usable_advances_to_missing_model_with_flag_on(monkeypatch):
    monkeypatch.setenv("OTR_ENABLE_HUMO", "1")
    monkeypatch.setenv("OTR_HUMO_CKPT", str(REPO_ROOT / "_no_such_ckpt.safetensors"))
    r = SMK.run_smoke("humo")
    au = _check(r, "assert_usable")
    assert au["ok"] is False and "missing_model" in au["detail"]   # ladder advanced


# --------------------------------------------------------------------------- #
# the real fallback chain + the LOUD restamp demonstration (humo)
# --------------------------------------------------------------------------- #
def test_humo_fallback_chain_and_demo():
    r = SMK.run_smoke("humo")
    assert _check(r, "fallback_chain")["detail"] == \
        "humo -> humo_1.7B -> still_motion"
    demo = r["fallback_demo"]
    assert demo["chain"] == ["humo", "humo_1.7B", "still_motion"]
    assert demo["converges_to_floor"] is True
    assert len(demo["decisions"]) == 2
    assert all(d["block_class"] == "hard" and d["failure_kind"] == "oom"
               and d["video_revision"] == 1 for d in demo["decisions"])
    assert all("frozen audio untouched" in line for line in demo["logs"])


def test_request_build_runs_for_each_engine():
    for engine in ("humo", "ltx_video", "wan_i2v"):
        r = SMK.run_smoke(engine)
        assert _check(r, "request_build")["ok"] is True


# --------------------------------------------------------------------------- #
# ltx_video shape: Sage state surfaced, no fallback chain (terminal)
# --------------------------------------------------------------------------- #
def test_ltx_has_sage_check_and_no_fallback_chain():
    r = SMK.run_smoke("ltx_video")
    assert _check(r, "sage_state")["ok"] is True
    assert [c for c in r["checks"] if c["name"] == "fallback_chain"] == []
    assert "fallback_demo" not in r                # only humo demos the fallback


# --------------------------------------------------------------------------- #
# --run-render is honest: never reports 'rendered' on the CPU box / unwired slice
# --------------------------------------------------------------------------- #
def test_run_render_is_honest_never_a_fake_pass(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_GPU_LEASE_DIR", str(tmp_path))
    r = SMK.run_smoke("humo", run_render=True)
    ra = r["render_attempt"]
    assert ra["status"] in {"gpu_slice_not_implemented", "not_ready",
                            "render_failed"}
    assert ra["status"] != "rendered"             # the GPU slice is the operator's


# --------------------------------------------------------------------------- #
# main exit codes: NOT READY on the CPU box -> non-zero (honest)
# --------------------------------------------------------------------------- #
def test_main_returns_nonzero_when_not_ready(capsys):
    assert SMK.main(["--engine", "humo"]) == 1
    out = capsys.readouterr().out
    assert "NOT READY" in out and "next steps" in out


def test_main_json_mode_runs_all(capsys):
    rc = SMK.main(["--engine", "all", "--json"])
    out = capsys.readouterr().out
    assert rc == 1                                 # nothing installed on this box
    assert '"engine": "humo"' in out and '"engine": "wan_i2v"' in out


def test_smoke_source_is_ascii_no_em_dash():
    src = SMOKE_SRC.read_text(encoding="utf-8")
    assert chr(0x2014) not in src                  # em-dash (U+2014) forbidden
    src.encode("ascii")                            # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
