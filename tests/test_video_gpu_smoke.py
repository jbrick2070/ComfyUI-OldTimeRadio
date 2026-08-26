"""CPU tests for the operator GPU-smoke probe (scripts/otr_video_gpu_smoke).

The probe is a readiness + scaffold tool: the live forward is the operator's
NotImplementedError GPU slice, so the probe must NEVER report a render as a pass.
These tests pin the CPU-checkable surface -- the readiness verdict reflects the
real flag / install / assert_usable state, the request build runs, and a render
attempt is reported honestly (gpu_slice_not_implemented / not_ready), never as
'rendered'. NO FALLBACKS (2026-07-02 rip): the chain/demo surface is gone.
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
#: The opt-in flags the probe's engines still read. OTR_ENABLE_WAN_I2V left this
#: list with the Wan 2.2 14B I2V retirement (2026-08-26) -- the flag now gates
#: nothing, so clearing it proved nothing. The remaining two are the probe's
#: whole ENGINES table, which is what the sweeps below walk.
_FLAGS = ["OTR_ENABLE_HUMO", "OTR_ENABLE_LTX_VIDEO"]


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
# NO FALLBACKS (2026-07-02): no chain check, no restamp demo -- for ANY engine
# --------------------------------------------------------------------------- #
#: The probe's in-process engines, which is exactly ``SMK.ENGINES``. The third
#: member was `wan_i2v` (Wan 2.2 14B I2V) until its retirement on 2026-08-26;
#: the probe row went with the adapter, so a sweep still naming it would only be
#: proving a KeyError. NOT replaced by `wan_ti2v` -- the 5B TI2V was never in
#: this probe's table and giving it a row here would assert readiness metadata
#: (install hint, Sage verdict) nobody wrote for it.
_PROBE_ENGINES = ("humo", "ltx_video")


def test_the_probe_table_is_exactly_the_engines_these_sweeps_walk():
    """The tripwire under the two sweeps below.

    They iterate a LITERAL tuple, so an engine added to (or dropped from) the
    probe would silently go unswept while both sweeps stayed green -- which is
    how a hand-kept list rots. Comparing against the probe's own table makes
    that a named failure here instead.
    """
    assert sorted(SMK.ENGINES) == sorted(_PROBE_ENGINES)


def test_no_fallback_chain_check_or_demo_for_any_engine():
    for engine in _PROBE_ENGINES:
        r = SMK.run_smoke(engine)
        assert [c for c in r["checks"] if c["name"] == "fallback_chain"] == []
        assert "fallback_demo" not in r


def test_request_build_runs_for_each_engine():
    for engine in _PROBE_ENGINES:
        r = SMK.run_smoke(engine)
        assert _check(r, "request_build")["ok"] is True


# --------------------------------------------------------------------------- #
# ltx_video shape: Sage state surfaced
# --------------------------------------------------------------------------- #
def test_ltx_has_sage_check():
    r = SMK.run_smoke("ltx_video")
    assert _check(r, "sage_state")["ok"] is True


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
    # `--engine all` must emit EVERY row of the probe table, not just the first.
    # The second name here was `wan_i2v` until the 14B retired (2026-08-26);
    # `ltx_video` is the other surviving in-process row, so the "more than one
    # report" property this line exists to hold is unchanged.
    for engine in _PROBE_ENGINES:
        assert '"engine": "%s"' % engine in out


def test_smoke_source_is_ascii_no_em_dash():
    src = SMOKE_SRC.read_text(encoding="utf-8")
    assert chr(0x2014) not in src                  # em-dash (U+2014) forbidden
    src.encode("ascii")                            # ASCII-only source


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
