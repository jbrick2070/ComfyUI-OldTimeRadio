"""S18.1 + S18.3 + S18.4 -- ProcSFX writeback convention pins.

S18.1: sfx_wav_path is "" not None on disk-write failure.
S18.3: strict_writeback opt-in raises on ledger write failure.
S18.4: sfx_render_status is stamped alongside sfx_wav_path.
"""
from __future__ import annotations

import inspect
import pathlib
import re

import pytest


# Source-level pins so test collection works without torch.
_PROCSFX_SRC = (
    pathlib.Path(__file__).resolve().parent.parent
    / "nodes" / "batch_procedural_sfx.py"
).read_text(encoding="utf-8")


def test_wav_path_default_is_empty_string_not_none():
    """S18.1: the wav_path local must default to "" not None.

    The render-loop initializer at line ~240 was
    ``wav_path: Optional[str] = None`` before S18.1; the §6.16
    convention requires "" for unset optional strings.
    """
    # Source-level pin: the new initializer phrasing.
    pat = re.compile(r'wav_path:\s*str\s*=\s*""')
    assert pat.search(_PROCSFX_SRC), (
        "S18.1: wav_path local must be initialized to \"\" not None. "
        "Either the initializer was reverted, or the typing changed."
    )


def test_writeback_or_empty_pattern():
    """The write-back block must coerce wav_path to "" if falsy."""
    pat = re.compile(r'r\.get\("wav_path"\)\s*or\s*""')
    assert pat.search(_PROCSFX_SRC), (
        "S18.1: writeback must coerce wav_path to \"\" if falsy. "
        "The ``r.get(\"wav_path\") or \"\"`` line is missing or moved."
    )


def test_sfx_render_status_in_writeback():
    """S18.4: sfx_render_status must land on the ledger row."""
    assert '"sfx_render_status":' in _PROCSFX_SRC, (
        "S18.4: sfx_render_status field absent from writeback dict."
    )


def test_strict_writeback_kwarg_in_signature():
    """S18.3: generate() accepts strict_writeback with default False."""
    from nodes.batch_procedural_sfx import BatchProceduralSFX
    sig = inspect.signature(BatchProceduralSFX.generate)
    params = sig.parameters
    assert "strict_writeback" in params, (
        "S18.3: strict_writeback kwarg missing from generate()."
    )
    assert params["strict_writeback"].default is False, (
        "S18.3: strict_writeback default must be False for "
        "soft-rollout compatibility."
    )


def test_strict_writeback_input_types_optional():
    """The widget must be declared in INPUT_TYPES().optional."""
    pat = re.compile(
        r'"strict_writeback":\s*\("BOOLEAN",\s*\{"default":\s*False\}\)'
    )
    assert pat.search(_PROCSFX_SRC), (
        "S18.3: strict_writeback widget not in INPUT_TYPES().optional "
        "or shape changed."
    )


def test_strict_writeback_raises_on_failure_path():
    """Source-level pin: the strict branch must raise RuntimeError."""
    assert "if strict_writeback:" in _PROCSFX_SRC
    # The raise must be reachable from inside the except block
    pat = re.compile(
        r"if strict_writeback:\s*\n\s*raise RuntimeError",
        re.MULTILINE,
    )
    assert pat.search(_PROCSFX_SRC), (
        "S18.3: strict branch must raise RuntimeError, not just log."
    )
