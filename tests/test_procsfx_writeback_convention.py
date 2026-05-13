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


# ---------------------------------------------------------------------
# C4 (S24, 2026-05-13) -- fallback_default_type + stale-comment cleanup
# ---------------------------------------------------------------------


def test_resolver_tracks_matched_flag():
    """C4: the resolver loop must set ``matched = True`` on every
    keyword / alias hit so the default-path can be distinguished from
    a successful semantic match."""
    assert "matched = False" in _PROCSFX_SRC, (
        "C4: resolver matched-flag initializer missing or moved."
    )
    # At least 6 ``matched = True`` lines: one per keyword branch
    # (door_knock, sci_fi_beep, white_noise, explosion, theremin)
    # plus one inside the ``for t in available_types`` loop.
    n_true = _PROCSFX_SRC.count("matched = True")
    assert n_true >= 6, (
        f"C4: expected >=6 matched=True branches; got {n_true}. "
        "A future refactor may have folded the alias chain; verify "
        "every keyword path still flips the flag."
    )


def test_fallback_default_type_status_stamped_when_no_match():
    """C4: when the resolver falls through without matching, the
    render_status must take the literal ``fallback_default_type``."""
    pat = re.compile(
        r"if matched:\s*\n"
        r"\s*render_status = \"ok\"\s*\n"
        r"\s*else:\s*\n"
        r"\s*render_status = \"fallback_default_type\"",
        re.MULTILINE,
    )
    assert pat.search(_PROCSFX_SRC), (
        "C4: matched-flag must select between 'ok' and "
        "'fallback_default_type' render_status. Either the branch "
        "shape changed or the status string differs."
    )


def test_stale_sfx_wav_path_None_doc_scrubbed():
    """C4 (b): every docstring / comment mention of
    ``sfx_wav_path=None`` must be replaced with the §6.16 convention
    ``sfx_wav_path=""``."""
    assert "sfx_wav_path=None" not in _PROCSFX_SRC, (
        "C4: stale sfx_wav_path=None mention surfaced. Per S18.1, "
        "the disk-write-failure stamp is \"\" not None."
    )


def test_stale_dur_range_literal_scrubbed():
    """C4 (c): the bare ``[0.25, 12.0]`` G7-old-range literal in the
    `dur_s` outline comment must be replaced with the symbolic
    SFX_DUR_MIN_S / SFX_DUR_MAX_S form. A forensic-only mention
    inside a "previously [0.25, 12.0]" anchor is allowed."""
    # Find every literal occurrence.
    hits = re.findall(r"\[0\.25,\s*12\.0\]", _PROCSFX_SRC)
    for h in hits:
        # Each hit must be inside a "previously [0.25, 12.0]" anchor.
        # The anchor must appear within 30 characters of the hit.
        idx = _PROCSFX_SRC.find(h)
        context = _PROCSFX_SRC[max(0, idx - 40):idx + len(h)]
        assert "previously" in context, (
            f"C4: bare [0.25, 12.0] reference at offset {idx} lacks "
            f"the 'previously' forensic anchor. Use SFX_DUR_MIN_S / "
            f"SFX_DUR_MAX_S symbolic in active code."
        )
