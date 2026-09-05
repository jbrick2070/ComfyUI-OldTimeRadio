"""R0a (b) tests: determinism env contract, the canonical headless launcher, the
scoped ``deterministic_inference`` CM, and the model-loader TF32 flip."""
from pathlib import Path

import random

import pytest
import torch

from nodes._otr_determinism import REQUIRED_DETERMINISM_ENV, determinism_env_status, deterministic_inference

REPO_ROOT = Path(__file__).resolve().parent.parent
_LAUNCH_CMD = REPO_ROOT / "scripts" / "_otr_soak_server_launch.cmd"
_LOADER = REPO_ROOT / "nodes" / "_otr_model_loader.py"


# --- env contract ---------------------------------------------------------

def test_required_env_keys():
    assert set(REQUIRED_DETERMINISM_ENV) == {
        "CUBLAS_WORKSPACE_CONFIG", "PYTHONHASHSEED",
        "NVIDIA_TF32_OVERRIDE", "TOKENIZERS_PARALLELISM",
    }


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


def test_launcher_hydrates_image_engine_weight_paths_before_python():
    """lumina_image and flux2_klein read an ABSOLUTE weights path out of the
    environment and fail CLOSED on it -- `os.getenv(...)` plus
    `os.path.isfile(...)` with no folder_paths fallback. The image dispatcher
    does not degrade either: a missing engine raises ImageRenderError
    "NO FALLBACK", so an unhydrated boot does not lose one picture, it kills
    the whole episode. A detached cmd does not inherit `setx` User env -- the
    same gotcha that already forced OTR_BLENDER_EXE into this launcher after
    mesh_stage silently fell back to still_parallax.

    z_image_turbo is deliberately absent: it ranks and auto-discovers its own
    unet, so it needs no variable to survive.
    """
    text = _LAUNCH_CMD.read_text(encoding="utf-8")
    main_index = text.index("main.py")
    for key in ("OTR_LUMINA_CKPT", "OTR_FLUX2_KLEIN_CKPT"):
        key_index = text.index(f"GetEnvironmentVariable('{key}','User')")
        assert key_index < main_index, (
            f"{_LAUNCH_CMD.name}: {key} must be hydrated before python starts")


# --- model-loader TF32 flip pinned ----------------------------------------

def test_model_loader_tf32_disabled():
    src = _LOADER.read_text(encoding="utf-8")
    assert "torch.backends.cuda.matmul.allow_tf32 = False" in src
    assert "torch.backends.cudnn.allow_tf32 = False" in src
    assert "allow_tf32 = True" not in src


# --- apply_module_determinism_defaults ------------------------------------


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
