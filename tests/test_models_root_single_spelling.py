"""One spelling of "where this type lives": ``_otr_models_root.model_type_dir``.

``wan_shared.configured_models_root()`` used to be a second spelling of the
models root, and its docstring claimed "the same override chain" for weeks while
the two functions disagreed. It was deleted on 2026-09-25 when the per-type owner
arrived; every former caller asks ``model_type_dir`` now. A claim of sameness is
proven by an equality, under every env state, or it is prose.
"""
from __future__ import annotations

import pathlib

import pytest

from nodes import _otr_models_root as models_root
from nodes._otr_video_engines import wan_shared as ws

_ENV = ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT")
REPO = pathlib.Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("pinned", ["OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT", None])
def test_outside_comfy_the_type_dir_is_the_root_plus_the_type(pinned, monkeypatch, tmp_path):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    if pinned:
        monkeypatch.setenv(pinned, str(tmp_path / "weights"))
    got = models_root.model_type_dir("diffusion_models")
    assert got == models_root._models_root() / "diffusion_models"
    if pinned:
        assert got == tmp_path / "weights" / "diffusion_models"


def test_no_second_spelling_is_left_in_nodes():
    offenders = [p.relative_to(REPO).as_posix()
                 for p in sorted((REPO / "nodes").rglob("*.py"))
                 if "configured_models_root" in p.read_text(encoding="utf-8")]
    assert offenders == [], offenders


def test_the_module_stays_cold_import_clean():
    """V-12: module scope imports only the stdlib + motion_common. The owner
    is imported lazily inside the function, never at module scope."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(ws))
    top_level = {
        (n.module or "") for n in tree.body
        if isinstance(n, ast.ImportFrom)
    }
    assert not any("_otr_models_root" in m for m in top_level), top_level
