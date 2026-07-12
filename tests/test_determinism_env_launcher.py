"""R0a (b) tests: determinism env contract, the canonical headless launcher, the
scoped ``deterministic_inference`` CM, and the model-loader TF32 flip."""
from pathlib import Path

import random

import pytest
import torch

from nodes._otr_determinism import (
    REQUIRED_DETERMINISM_ENV,
    apply_module_determinism_defaults,
    assert_determinism_env_ready,
    determinism_env_status,
    deterministic_inference,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
_LAUNCH_CMD = REPO_ROOT / "scripts" / "_otr_soak_server_launch.cmd"
_LOADER = REPO_ROOT / "nodes" / "_otr_model_loader.py"


# --- env contract ---------------------------------------------------------

def test_required_env_keys():
    assert set(REQUIRED_DETERMINISM_ENV) == {
        "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
        "NVIDIA_TF32_OVERRIDE", "TOKENIZERS_PARALLELISM",
    }


def test_assert_env_ready_passes_when_all_set(monkeypatch):
    for k, v in REQUIRED_DETERMINISM_ENV.items():
        monkeypatch.setenv(k, v)
    assert assert_determinism_env_ready() is True


def test_assert_env_ready_raises_when_missing(monkeypatch):
    for k, v in REQUIRED_DETERMINISM_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(RuntimeError):
        assert_determinism_env_ready()
    # non-strict reports without raising
    assert assert_determinism_env_ready(strict=False) is False


def test_env_status_reports_actual(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "7")  # wrong value
    ok, actual = determinism_env_status()["PYTHONHASHSEED"]
    assert ok is False and actual == "7"


# --- launchers: values present AND set before python (C-1 ordering) --------

def test_launcher_exists():
    assert _LAUNCH_CMD.is_file(), f"missing launcher: {_LAUNCH_CMD}"


def test_launcher_sets_all_env_before_python():
    text = _LAUNCH_CMD.read_text(encoding="utf-8")
    lines = text.splitlines()

    def _is_comment(ln: str) -> bool:
        s = ln.strip()
        return s.upper().startswith("REM") or s.startswith("#")

    def first_code_line_containing(token):
        for i, ln in enumerate(lines):
            if not _is_comment(ln) and token in ln:
                return i
        return None

    # The real invocation line (a comment header may also mention main.py).
    py_idx = first_code_line_containing("main.py")
    assert py_idx is not None, "launcher must invoke main.py"
    for key, val in REQUIRED_DETERMINISM_ENV.items():
        key_idx = first_code_line_containing(key)
        assert key_idx is not None, f"{_LAUNCH_CMD.name} does not set {key}"
        assert val in lines[key_idx], (
            f"{_LAUNCH_CMD.name}: {key} must be set to {val}"
        )
        assert key_idx < py_idx, (
            f"{_LAUNCH_CMD.name}: {key} must be exported before python"
        )


def test_launcher_hydrates_remote_llm_keys_before_python():
    text = _LAUNCH_CMD.read_text(encoding="utf-8")
    main_index = text.index("main.py")
    for key in (
        "OPENROUTER_API_KEY", "OTR_GOOGLE_API_KEY", "GEMINI_API_KEY",
        "GOOGLE_API_KEY",
    ):
        key_index = text.index(f"GetEnvironmentVariable('{key}','User')")
        assert key_index < main_index


# --- model-loader TF32 flip pinned ----------------------------------------

def test_model_loader_tf32_disabled():
    src = _LOADER.read_text(encoding="utf-8")
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in src
    assert "torch.backends.cudnn.allow_tf32 = False" in src
    assert "allow_tf32 = True" not in src


# --- apply_module_determinism_defaults ------------------------------------

def test_apply_module_defaults_disables_tf32():
    prev_mm = torch.backends.cuda.matmul.allow_tf32
    prev_cudnn = torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        apply_module_determinism_defaults()
        assert torch.backends.cuda.matmul.allow_tf32 is False
        assert torch.backends.cudnn.allow_tf32 is False
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev_mm
        torch.backends.cudnn.allow_tf32 = prev_cudnn


# --- scoped deterministic_inference CM ------------------------------------

def test_cm_enables_then_restores_determinism_flag():
    prev = torch.are_deterministic_algorithms_enabled()
    with deterministic_inference(123):
        assert torch.are_deterministic_algorithms_enabled() is True
    assert torch.are_deterministic_algorithms_enabled() == prev


def test_cm_restores_rng_state():
    prev_py = random.getstate()
    prev_torch = torch.get_rng_state()
    with deterministic_inference(123):
        random.random()
        torch.rand(5)
    assert random.getstate() == prev_py
    assert torch.equal(torch.get_rng_state(), prev_torch)


def test_cm_same_seed_same_draw():
    with deterministic_inference(7):
        a = torch.rand(4)
    with deterministic_inference(7):
        b = torch.rand(4)
    assert torch.equal(a, b)


def test_cm_different_seed_different_draw():
    with deterministic_inference(7):
        a = torch.rand(4)
    with deterministic_inference(8):
        b = torch.rand(4)
    assert not torch.equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
