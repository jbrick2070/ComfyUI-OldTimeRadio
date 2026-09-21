# -*- coding: utf-8 -*-
r"""The models root must never invent ``C:\ComfyUI-Models`` off Windows.

MEASURED, NOT IMAGINED. On 2026-09-21 a RunPod Linux box was found with a
directory literally named ``C:\ComfyUI-Models`` INSIDE the repo, holding
3.7 GB of real weights: the AnimateDiff v3 motion module, its domain adapter
and an SD 1.5 checkpoint. On POSIX that string is not an absolute path, so it
resolved relative to the working directory and a fetch wrote gigabytes into
it.

THE CHAIN, because the obvious culprit was not the culprit. ``_models_root``
already guarded the Windows literal on ``is_dir()`` and already preferred
ComfyUI's ``folder_paths``. What it did NOT guard was its FINAL fallback,
which returned that literal unconditionally. ``folder_paths`` is importable
only from inside a running ComfyUI -- which is exactly when it is absent, a
fetch script is running -- so the function returned the literal, and
``scripts/otr_fetch_lane_weights.py`` never reached its own sibling-directory
fallback written for precisely this case, because the call above it had
already produced an answer.

THE OPERATOR'S RULE this encodes (2026-09-21): ``C:\ComfyUI-Models`` is his
5080's layout and nobody else's. Everyone else gets ComfyUI's own models
directory.

Returning a wrong root is worse than returning nothing: nothing stops, and a
wrong root downloads gigabytes to a place the user never chose and will not
find. So off Windows the last resort raises instead.
"""
from __future__ import annotations

import os
import pathlib

import pytest

from nodes import _otr_gguf_backend as g


def test_an_explicit_env_var_always_wins(monkeypatch, tmp_path):
    """Step 1, and the escape hatch every pod run relies on."""
    monkeypatch.setattr(g.otr_env, "get",
                        lambda k, *a, **kw: (str(tmp_path)
                                             if k == "OTR_COMFYUI_MODELS_ROOT"
                                             else None))
    assert g._models_root() == pathlib.Path(str(tmp_path))


def test_off_windows_it_refuses_rather_than_naming_the_windows_path(monkeypatch):
    """The whole point: no env, no tree, no ComfyUI, no sibling -> raise.

    Every earlier step is disabled so the LAST one is what answers. On POSIX
    it must decline, because the only thing it could return is a relative
    directory named after a Windows drive.
    """
    monkeypatch.setattr(g.otr_env, "get", lambda *a, **kw: None)
    monkeypatch.setattr(pathlib.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(g.os, "name", "posix")

    with pytest.raises(g.ModelsRootUnresolved) as exc:
        g._models_root()
    msg = str(exc.value)
    assert "OTR_COMFYUI_MODELS_ROOT" in msg, (
        "the refusal must tell the reader the knob that fixes it")
    assert "ComfyUI-Models" in msg, (
        "and name the path it declined to invent, so the message is "
        "recognisable to whoever finds the stray directory")


def test_on_windows_the_literal_is_still_the_last_resort(monkeypatch):
    """The reference machine keeps working.

    Its own tree exists, so in practice step 2 answers long before this. This
    pins that the FINAL step is unchanged on Windows, which is what makes the
    POSIX guard above safe to add.
    """
    monkeypatch.setattr(g.otr_env, "get", lambda *a, **kw: None)
    monkeypatch.setattr(pathlib.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(os.path, "isdir", lambda p: False)
    monkeypatch.setattr(g.os, "name", "nt")

    assert str(g._models_root()).endswith("ComfyUI-Models")


def test_the_sibling_models_dir_is_preferred_over_the_literal(monkeypatch, tmp_path):
    """Step 4, which is what makes a Linux or Docker box work with no env var.

    A ComfyUI checkout puts ``custom_nodes/`` beside ``models/``. That sibling
    has to be consulted BEFORE the literal, because the literal is reachable
    off Windows and the sibling is the correct answer there.
    """
    monkeypatch.setattr(g.otr_env, "get", lambda *a, **kw: None)
    monkeypatch.setattr(pathlib.Path, "is_dir", lambda self: False)
    monkeypatch.setattr(g.os, "name", "posix")
    monkeypatch.setattr(os.path, "isdir", lambda p: str(p).endswith("models"))

    got = g._models_root()
    assert str(got).endswith("models"), got
    assert "ComfyUI-Models" not in str(got), (
        "the sibling must win; returning the literal here is the defect")
