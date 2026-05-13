"""S18.2 (IMP-20 part 2): post-freeze §6.16 audit walker.

Pins:
  - audit returns violations list (default soft mode)
  - audit raises ValueError when strict=True and violations exist
  - clean ledger returns empty list / passes strict
  - sfx_render_status is in the audited field set (S18.4 wiring)
"""
from __future__ import annotations

import pytest

from nodes._otr_ledger_consumers import (
    ALLOWED_SFX_RENDER_STATUS,
    _OPTIONAL_STRING_FIELDS,
    audit_post_freeze_writeback,
)


def _make_ledger(*lines: dict) -> dict:
    return {"lines": list(lines), "meta": {}}


def test_audit_flags_none_in_string_field():
    led = _make_ledger({
        "line_id": "L001",
        "speaker": "ALICE",
        "sfx_wav_path": None,
    })
    violations = audit_post_freeze_writeback(led)
    assert len(violations) == 1
    assert "L001" in violations[0]
    assert "sfx_wav_path" in violations[0]


def test_audit_passes_when_clean():
    led = _make_ledger({
        "line_id": "L001",
        "speaker": "ALICE",
        "sfx_wav_path": "",
    })
    assert audit_post_freeze_writeback(led) == []


def test_audit_strict_raises_on_violation():
    led = _make_ledger({"line_id": "L001", "sfx_wav_path": None})
    with pytest.raises(ValueError) as exc:
        audit_post_freeze_writeback(led, strict=True)
    assert "1" in str(exc.value)
    assert "L001" in str(exc.value)


def test_audit_strict_clean_passes():
    led = _make_ledger({"line_id": "L001", "sfx_wav_path": ""})
    audit_post_freeze_writeback(led, strict=True)  # no raise


def test_audit_handles_multiple_violations():
    led = _make_ledger(
        {"line_id": "L001", "sfx_wav_path": None, "sfx_engine": None},
        {"line_id": "L002", "audio_wav_path": None},
    )
    violations = audit_post_freeze_writeback(led)
    assert len(violations) == 3


def test_audit_keyword_only_strict():
    """strict is keyword-only -- positional callers must raise."""
    led = _make_ledger()
    with pytest.raises(TypeError):
        audit_post_freeze_writeback(led, True)  # positional


def test_sfx_render_status_in_audited_fields():
    """S18.4 wired sfx_render_status as a new optional string. The
    audit walker must cover it."""
    assert "sfx_render_status" in _OPTIONAL_STRING_FIELDS


def test_audit_handles_empty_ledger():
    """No lines -> no violations."""
    assert audit_post_freeze_writeback({"lines": []}) == []
    assert audit_post_freeze_writeback({}) == []


def test_audit_skips_non_dict_lines():
    """Defensive: malformed entries don't crash the walker."""
    led = {"lines": [None, "string", {"line_id": "L001"}]}
    # Should not raise on the None / string entries.
    violations = audit_post_freeze_writeback(led)
    assert violations == []


# ---------------------------------------------------------------------
# C5 (S24, 2026-05-13) -- ALLOWED_SFX_RENDER_STATUS enum check
# ---------------------------------------------------------------------


_EXPECTED_ENUM = frozenset({
    "",
    "ok",
    "ok_cache",
    "error",
    "fallback_silence",
    "fallback_output_shape",
    "fallback_default_type",
    "skipped",
})


def test_allowed_sfx_render_status_membership():
    """The enum frozenset must contain the 8 values producers stamp.
    Adding or removing a value requires updating this test in
    lockstep so a future producer-side change can't silently
    introduce an unaudited status."""
    assert ALLOWED_SFX_RENDER_STATUS == _EXPECTED_ENUM, (
        f"C5: ALLOWED_SFX_RENDER_STATUS drift. Expected "
        f"{sorted(_EXPECTED_ENUM)!r}; got "
        f"{sorted(ALLOWED_SFX_RENDER_STATUS)!r}."
    )


@pytest.mark.parametrize("status", sorted(_EXPECTED_ENUM))
def test_audit_accepts_every_enum_value(status):
    """Every value in the enum must pass the walker in both soft and
    strict mode."""
    led = _make_ledger({"line_id": "L001", "sfx_render_status": status})
    assert audit_post_freeze_writeback(led) == []
    audit_post_freeze_writeback(led, strict=True)  # no raise


def test_audit_rejects_typo_in_strict_mode():
    """A typo like 'fallback_silnce' must raise in strict mode."""
    led = _make_ledger({
        "line_id": "L001",
        "sfx_render_status": "fallback_silnce",
    })
    with pytest.raises(ValueError) as exc:
        audit_post_freeze_writeback(led, strict=True)
    assert "ALLOWED_SFX_RENDER_STATUS" in str(exc.value)
    assert "fallback_silnce" in str(exc.value)


def test_audit_rejects_typo_in_soft_mode():
    """Soft mode collects the violation without raising."""
    led = _make_ledger({
        "line_id": "L001",
        "sfx_render_status": "fallback_silnce",
    })
    violations = audit_post_freeze_writeback(led)
    assert len(violations) == 1
    assert "fallback_silnce" in violations[0]


def test_audit_only_enum_checks_sfx_render_status():
    """The enum check applies ONLY to sfx_render_status. Other
    optional-string fields stay string-shape-only (only None counts
    as a violation; arbitrary strings pass)."""
    # An arbitrary string in another optional-string field must NOT
    # be a violation, only sfx_render_status uses the enum.
    led = _make_ledger({
        "line_id": "L001",
        "sfx_engine": "garbage_engine_name",
        "sfx_render_status": "ok",
    })
    assert audit_post_freeze_writeback(led) == []
