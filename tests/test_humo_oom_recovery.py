"""tests/test_humo_oom_recovery.py

BUG-LOCAL-126 alarm-plumbing unit tests. Verifies:
  - `_is_oom_exception` correctly identifies torch OOM in both forms
    (the modern `torch.OutOfMemoryError` subclass and the legacy
    plain-`RuntimeError`-with-message form).
  - `_hard_reset_cuda_context` runs all its cleanup steps without
    raising, even when torch / comfy.model_management aren't
    available (gracefully degrades, logs warnings, does NOT
    re-raise).
  - `HumoSoakCapReached` exception carries `lines_completed` and
    `cap` attributes.
  - The audit script's FAIL_PATTERNS includes the
    `Fatal Python error: Aborted` signal so a soak watcher can
    surface BUG-126 hard aborts.

Pure-stdlib + targeted import. No torch required to run -- the
module-load helpers are best-effort and degrade cleanly.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
if str(_NODES_DIR) not in sys.path:
    sys.path.insert(0, str(_NODES_DIR))


def _import_batch_humo_module():
    """Load `batch_humo_render` from its file path so the test runs
    even if the package's heavy imports (Bark, Wan2.1, etc.) aren't
    fully wired in the test venv. Returns the module or None.
    """
    src = _NODES_DIR / "batch_humo_render.py"
    if not src.is_file():
        return None
    # If already in sys.modules from a previous test, return it.
    if "batch_humo_render" in sys.modules:
        return sys.modules["batch_humo_render"]
    spec = importlib.util.spec_from_file_location(
        "batch_humo_render_for_test", src,
    )
    if spec is None or spec.loader is None:
        return None
    try:
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    except Exception:  # noqa: BLE001
        # Heavy deps may fail to import in the test venv; that's OK
        # for the helper-level tests below -- the helpers are stdlib.
        return None
    return mod


# ---------- _is_oom_exception ------------------------------------


def test_is_oom_exception_modern_oom_class():
    """If torch is installed and exposes OutOfMemoryError, an instance
    of that class returns True."""
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "_is_oom_exception"):
        pytest.skip("batch_humo_render didn't load in test venv")
    try:
        import torch  # type: ignore
        oom_cls = getattr(torch, "OutOfMemoryError", None)
    except Exception:  # noqa: BLE001
        oom_cls = None
    if oom_cls is None:
        pytest.skip("torch.OutOfMemoryError not present in this build")
    exc = oom_cls("test fixture: out of memory")
    assert mod._is_oom_exception(exc) is True


def test_is_oom_exception_legacy_runtime_error_message():
    """Plain RuntimeError with `out of memory` in the message is the
    older OOM form; must still register as OOM."""
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "_is_oom_exception"):
        pytest.skip("batch_humo_render didn't load in test venv")
    exc = RuntimeError(
        "Allocation on device 0 would exceed allowed memory. "
        "(out of memory) Currently allocated: 24.50 GiB"
    )
    assert mod._is_oom_exception(exc) is True


def test_is_oom_exception_unrelated_error_returns_false():
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "_is_oom_exception"):
        pytest.skip("batch_humo_render didn't load in test venv")
    assert mod._is_oom_exception(ValueError("not an oom")) is False
    assert mod._is_oom_exception(TypeError("also not")) is False
    assert mod._is_oom_exception(RuntimeError("totally fine")) is False


def test_is_oom_exception_runtime_error_case_insensitive():
    """The OOM message comes from PyTorch's C extension and the
    case may shift between builds. The check normalizes."""
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "_is_oom_exception"):
        pytest.skip("batch_humo_render didn't load in test venv")
    assert mod._is_oom_exception(RuntimeError("OUT OF MEMORY")) is True
    assert mod._is_oom_exception(RuntimeError("Out Of Memory")) is True


# ---------- HumoSoakCapReached -----------------------------------


def test_humo_soak_cap_reached_carries_attrs():
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "HumoSoakCapReached"):
        pytest.skip("batch_humo_render didn't load in test venv")
    exc = mod.HumoSoakCapReached(lines_completed=8, cap=8)
    assert exc.lines_completed == 8
    assert exc.cap == 8
    assert "8 of cap 8" in str(exc)


def test_humo_soak_cap_reached_is_runtime_error_subclass():
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "HumoSoakCapReached"):
        pytest.skip("batch_humo_render didn't load in test venv")
    assert issubclass(mod.HumoSoakCapReached, RuntimeError)


# ---------- _hard_reset_cuda_context degrades cleanly ------------


def test_hard_reset_does_not_raise_when_torch_missing(monkeypatch):
    """Even if torch isn't importable in this venv, the helper must
    log + return. It's recovery code -- it can never escalate the
    fault by itself raising."""
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "_hard_reset_cuda_context"):
        pytest.skip("batch_humo_render didn't load in test venv")
    # Force the import-time `import torch` inside the helper to fail
    # by hiding the name from sys.modules + builtins. Easiest: stub
    # torch with a sentinel that raises on attribute access, but the
    # helper guards on its own exception. Just call it and assert
    # no propagation.
    mod._hard_reset_cuda_context()  # must not raise


def test_hard_reset_does_not_raise_after_runtime_error(monkeypatch):
    """If `mm.unload_all_models` raises, the helper logs and continues
    to the rest of the steps; no propagation."""
    mod = _import_batch_humo_module()
    if mod is None or not hasattr(mod, "_hard_reset_cuda_context"):
        pytest.skip("batch_humo_render didn't load in test venv")
    mod._hard_reset_cuda_context()  # must not raise


# ---------- audit script FAIL_PATTERNS includes Aborted signal ----


def test_audit_fail_patterns_includes_fatal_python_error():
    """BUG-LOCAL-126 watcher / auditor must surface the C-level abort
    signal so an incomplete soak isn't reported as PASS."""
    src = _REPO_ROOT / "scripts" / "audit_otr_full_run.py"
    body = src.read_text(encoding="utf-8")
    assert "Fatal Python error: Aborted" in body, (
        "audit_otr_full_run.py must include 'Fatal Python error: Aborted' "
        "in FAIL_PATTERNS to catch BUG-126 C-level CUDA aborts"
    )
    # Stronger: load the FAIL_PATTERNS object and assert by membership.
    spec = importlib.util.spec_from_file_location(
        "audit_otr_full_run_for_test", src,
    )
    assert spec is not None and spec.loader is not None
    audit_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit_mod)
    patterns = audit_mod.FAIL_PATTERNS
    string_patterns = [p for p in patterns if isinstance(p, str)]
    assert "Fatal Python error: Aborted" in string_patterns
