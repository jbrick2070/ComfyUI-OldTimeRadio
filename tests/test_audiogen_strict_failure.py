"""S17.2 (IMP-19): AudioGen strict failure mode.

Default behavior: transformers/AudioGen ImportError raises
RuntimeError so production never silently substitutes silence
(Directive 1 breach). Opt-in fallback path honors per-cue
render_queue durations.
"""
from __future__ import annotations

import inspect
import re
import pathlib

import pytest


# Pull the module source so we can inspect the failure-path code
# without importing torch/transformers (which adds startup cost +
# may break in some CI envs).
_AUDIOGEN_SRC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "batch_audiogen_generator.py"
).read_text(encoding="utf-8")


def test_strict_failure_raises_runtime_error_on_import_error():
    """The strict path (allow_silence_fallback=False, default) must
    raise RuntimeError when transformers/AudioGen ImportError fires.

    Source-level assertion: the strict path is present and the
    silent-silence fallback is gated.
    """
    # The two markers that distinguish the strict path
    assert "raise RuntimeError(msg) from exc" in _AUDIOGEN_SRC, (
        "S17.2 contract: ImportError must convert to RuntimeError "
        "on the strict path. The 'raise RuntimeError(msg) from exc' "
        "line is missing or moved -- check that the strict path is "
        "still wired."
    )
    assert "if not allow_silence_fallback" in _AUDIOGEN_SRC, (
        "S17.2 contract: silence fallback is gated by the "
        "allow_silence_fallback kwarg. The gate is missing."
    )


def test_fallback_silence_uses_per_cue_duration_not_default():
    """The opt-in fallback path indexes render_queue[idx][duration]
    instead of the bug-source default_duration the prior code used."""
    # The bug: prior code did
    #   torch.zeros(..., int(AUDIOGEN_SAMPLE_RATE * default_duration))
    # The S17.2 fix indexes the actual per-cue duration:
    #   item_dur = float(render_queue[idx]["duration"])
    assert 'render_queue[idx]["duration"]' in _AUDIOGEN_SRC, (
        "S17.2 fallback path no longer reads per-cue duration from "
        "render_queue. The default_duration regression is back."
    )


def test_fallback_silence_stamps_status_on_ledger_row():
    """The fallback path must stamp sfx_render_status="fallback_silence"
    on the render result so downstream consumers can branch."""
    assert '"fallback_silence"' in _AUDIOGEN_SRC, (
        "S17.2: fallback path must stamp sfx_render_status with "
        "the fallback_silence string. Status missing."
    )


def test_input_types_has_allow_silence_fallback():
    """The widget must be declared in INPUT_TYPES().optional so the
    user can opt in via the workflow."""
    # Source-level grep: declaration shape uses BOOLEAN default False.
    pat = re.compile(
        r'"allow_silence_fallback":\s*\("BOOLEAN",\s*\{"default":\s*False\}\)'
    )
    assert pat.search(_AUDIOGEN_SRC), (
        "S17.2: allow_silence_fallback widget not in INPUT_TYPES, "
        "or its declaration shape changed."
    )


def test_generate_signature_accepts_allow_silence_fallback():
    """The generate() signature accepts the kwarg with default
    False -- positional callers must not be silently affected."""
    from nodes import batch_audiogen_generator as mod
    sig = inspect.signature(mod.BatchAudioGenGenerator.generate)
    params = sig.parameters
    assert "allow_silence_fallback" in params
    assert params["allow_silence_fallback"].default is False
