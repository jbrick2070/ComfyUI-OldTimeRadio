"""``wan_shared.configured_models_root()`` IS ``_otr_gguf_backend._models_root()``.

The docstring claimed "the same override chain" for weeks while the two
functions disagreed: one existence-gated the legacy literal and then asked
``folder_paths``, the other returned the literal unconditionally. A claim of
sameness is proven by an equality, under every env state, or it is prose.
"""
from __future__ import annotations

import pytest

from nodes import _otr_gguf_backend as gguf
from nodes._otr_video_engines import wan_shared as ws

_ENV = ("OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT")


@pytest.mark.parametrize("pinned", ["OTR_COMFYUI_MODELS_ROOT", "COMFYUI_MODELS_ROOT", None])
def test_one_spelling_under_each_env_state(pinned, monkeypatch, tmp_path):
    for key in _ENV:
        monkeypatch.delenv(key, raising=False)
    if pinned:
        monkeypatch.setenv(pinned, str(tmp_path / "weights"))
    assert ws.configured_models_root() == str(gguf._models_root())
    if pinned:
        assert ws.configured_models_root() == str(tmp_path / "weights")


def test_the_first_env_pin_wins_over_the_second(monkeypatch, tmp_path):
    monkeypatch.setenv("OTR_COMFYUI_MODELS_ROOT", str(tmp_path / "first"))
    monkeypatch.setenv("COMFYUI_MODELS_ROOT", str(tmp_path / "second"))
    assert ws.configured_models_root() == str(tmp_path / "first")
    assert ws.configured_models_root() == str(gguf._models_root())


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
    assert not any("_otr_gguf_backend" in m for m in top_level), top_level
