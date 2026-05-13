"""C3 (S24, 2026-05-13) -- MusicGen strict ImportError behavior.

Mirrors AudioGen S17.2 (IMP-19) for MusicGen. Default behavior:
transformers/MusicGen ImportError raises RuntimeError. Opt-in
``allow_silence_fallback`` widget keeps the silence-fill path for
smoke tests; that path stamps ``_render_status="fallback_silence"``
on every cue dict so the writeback below carries the degraded state.

Source-level pins so collection works without transformers.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest


_MUSICGEN_SRC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "musicgen_theme.py"
).read_text(encoding="utf-8")


def test_strict_path_raises_runtime_error():
    """The ImportError branch must raise RuntimeError when the
    ``allow_silence_fallback`` widget is False (the default)."""
    pat = re.compile(
        r"if not allow_silence_fallback:\s*\n"
        r"\s*log\.error\([^)]*\)\s*\n"
        r"\s*raise RuntimeError\(msg\) from exc",
        re.MULTILINE,
    )
    assert pat.search(_MUSICGEN_SRC), (
        "C3: MusicGen ImportError must raise RuntimeError in strict "
        "mode (allow_silence_fallback=False). Either the raise was "
        "removed or the branch shape changed."
    )


def test_fallback_path_stamps_render_status():
    """The opt-in fallback path must tag each cue dict with
    ``_render_status='fallback_silence'`` so the per-cue writeback
    can stamp the ledger row."""
    pat = re.compile(
        r'cues\[cue_id\]\["_render_status"\] = "fallback_silence"',
        re.MULTILINE,
    )
    assert pat.search(_MUSICGEN_SRC), (
        "C3: MusicGen fallback path must stamp _render_status="
        "fallback_silence on every cue dict; missing."
    )


def test_input_types_has_allow_silence_fallback():
    """The widget must be declared in INPUT_TYPES().optional with
    BOOLEAN type, default False."""
    pat = re.compile(
        r'"allow_silence_fallback":\s*\("BOOLEAN",\s*\{"default":\s*False\}\)'
    )
    assert pat.search(_MUSICGEN_SRC), (
        "C3: allow_silence_fallback widget not in INPUT_TYPES, "
        "or its declaration shape changed."
    )


def test_render_signature_accepts_allow_silence_fallback():
    """The render() signature must accept allow_silence_fallback
    with default False so positional callers don't break."""
    from nodes.musicgen_theme import MusicGenTheme
    sig = inspect.signature(MusicGenTheme.render)
    params = sig.parameters
    assert "allow_silence_fallback" in params, (
        "C3: render() missing allow_silence_fallback kwarg."
    )
    assert params["allow_silence_fallback"].default is False, (
        "C3: allow_silence_fallback default must be False for "
        "strict-failure semantics."
    )
