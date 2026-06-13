"""R1 -- in-tree temp allocator for video-engine intermediates.

The soak hygiene gate (assert_no_stray_writes) failed every full leg because
engine intermediate .mp4s were minted with bare tempfile.mktemp/mkstemp (no
dir=), leaking otr_*-prefixed files into the system temp dir. The fix routes
every intermediate through nodes/_otr_video_engines/_tmp.py::otr_engine_tmp_mp4,
which lands them under the in-tree tmp tier (otr/episodes/_shared/tmp) the OH-3
janitor sweeps.

Two layers of coverage:
  * behavior of otr_engine_tmp_mp4 (in-tree path, non-existent on return, no
    system-temp leak, fail-closed in production, OTR_TEST_MODE escape);
  * an AST forbidden-temp ban over the video-engine package so no engine can
    re-introduce a bare mktemp/mkstemp or a gettempdir()-joined otr_* leaf.

Hermetic: no GPU, no ffmpeg, no real render.
"""
from __future__ import annotations

import ast
import os
import sys
import tempfile
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import nodes._otr_paths as _paths  # noqa: E402
from nodes._otr_video_engines import _tmp  # noqa: E402

_ENGINES_DIR = _REPO_ROOT / "nodes" / "_otr_video_engines"


def _raise(*_a, **_k):
    raise RuntimeError("otr_shared_tmp_dir unavailable (test)")


# ---------------------------------------------------------------------------
# Behavior of the allocator
# ---------------------------------------------------------------------------


def test_returns_in_tree_nonexistent_path(tmp_path, monkeypatch):
    shared = tmp_path / "otr" / "episodes" / "_shared" / "tmp"
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", lambda: shared)
    p = _tmp.otr_engine_tmp_mp4("otr_floor_test_")
    assert p.endswith(".mp4")
    assert Path(p).parent == shared, (
        f"intermediate not routed in-tree: {p!r} not under {shared!r}"
    )
    assert not os.path.exists(p), "path must NOT exist on return (mktemp contract)"


def test_no_otr_leak_into_system_temp(tmp_path, monkeypatch):
    shared = tmp_path / "shared_tmp"
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", lambda: shared)
    _tmp.otr_engine_tmp_mp4("otr_floor_leakcheck_")
    sysd = tempfile.gettempdir()
    leaked = [n for n in os.listdir(sysd) if n.startswith("otr_floor_leakcheck_")]
    assert leaked == [], f"otr_* artifact leaked into system temp dir: {leaked!r}"


def test_fail_closed_in_production(monkeypatch):
    """If the in-tree dir cannot be resolved and OTR_TEST_MODE is unset,
    the allocator RAISES rather than silently leak to the system temp dir."""
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", _raise)
    monkeypatch.delenv("OTR_TEST_MODE", raising=False)
    with pytest.raises(RuntimeError):
        _tmp.otr_engine_tmp_mp4("otr_fc_")


def test_test_mode_allows_tempfile_default(monkeypatch):
    """OTR_TEST_MODE permits the tempfile default dir when the in-tree dir
    cannot be resolved -- headless CPU unit tests with no ComfyUI output."""
    monkeypatch.setattr(_paths, "otr_shared_tmp_dir", _raise)
    monkeypatch.setenv("OTR_TEST_MODE", "1")
    p = _tmp.otr_engine_tmp_mp4("otr_tm_")
    assert p.endswith(".mp4")
    assert not os.path.exists(p)


# ---------------------------------------------------------------------------
# AST forbidden-temp ban over the video-engine package
# ---------------------------------------------------------------------------


def _call_name(call: ast.Call):
    fn = call.func
    if isinstance(fn, ast.Attribute):
        return fn.attr
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _is_gettempdir(node) -> bool:
    return isinstance(node, ast.Call) and _call_name(node) == "gettempdir"


def _otr_str_arg(call: ast.Call) -> bool:
    for arg in list(call.args) + [kw.value for kw in call.keywords]:
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            if arg.value.lower().startswith("otr"):
                return True
    return False


def test_no_engine_mints_otr_temp_outside_helper():
    """No module in the video-engine package (except the sanctioned _tmp.py
    allocator) may call tempfile.mktemp/mkstemp, nor join gettempdir() with an
    otr_*-prefixed leaf. Every engine intermediate must go through
    otr_engine_tmp_mp4 so the soak hygiene gate stays green."""
    offenders: list[str] = []
    for py in sorted(_ENGINES_DIR.glob("*.py")):
        if py.name == "_tmp.py":
            continue
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = _call_name(node)
            if name in ("mktemp", "mkstemp"):
                offenders.append(
                    f"{py.name}:{node.lineno} bare tempfile.{name}() -- "
                    f"use otr_engine_tmp_mp4")
            if name == "join" and _otr_str_arg(node):
                # os.path.join(..., "otr_*") with a gettempdir() arg = the leak.
                if any(_is_gettempdir(a) for a in node.args):
                    offenders.append(
                        f"{py.name}:{node.lineno} gettempdir()+otr_* leaf -- "
                        f"use _in_tree_tmp_dir")
    assert not offenders, (
        "video-engine temp leak re-introduced (R1):\n  " + "\n  ".join(offenders)
    )
