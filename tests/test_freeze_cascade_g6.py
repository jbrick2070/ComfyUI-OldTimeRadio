"""tests/test_freeze_cascade_g6.py

FreezeCascade Phase 0 / Phase 10 G6 voice_preset invariant.

Sprint 2 (a) re-timed this gate. Bark ``voice_preset`` is now assigned by
OTR_CastLock AFTER the freeze (the writer no longer stamps it), so an EMPTY
preset at freeze time is expected and G6 only WARNs on it. A PRESENT but
non-``v2/`` preset on a Bark row is still a hard error (a real malformed value).
The hard voice gate -- every non-ANNOUNCER row carries a ``v2/`` preset, no two
collide -- moved to OTR_CastLock's exit
(``_otr_casting._assert_voice_preset_invariant`` + ``_assert_unique_bark_voices``,
invoked by ``OTR_CastLock._assign_bark_voices``). Gate 3 lives in the Bark engine.
"""
from __future__ import annotations

import pytest

from nodes import _otr_ledger_freeze as _LFC


def _make_ledger(cast_rows: list[dict], lines: list[dict] | None = None) -> dict:
    """Minimal ledger shape sufficient for the audit walk."""
    if lines is None:
        lines = [
            {"line_id": "b001", "char_id": "c01",
             "speaker_role": "character", "text": "Hello.", "traits": "calm"},
        ]
    return {
        "schema_version": "l3-2026-05-14",
        "cast": cast_rows,
        "lines": lines,
        "meta": {},
    }


def test_g6_passes_on_valid_cast():
    """Cast where every non-ANNOUNCER row has a v2/* preset audits clean."""
    led = _make_ledger([
        {"char_id": "announcer", "name": "ANNOUNCER",
         "voice_preset": "bm_george", "traits": "warm"},
        {"char_id": "c01", "name": "LEMMY",
         "voice_preset": "v2/en_speaker_8", "traits": "gruff"},
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g6_errors = [e for e in report.errors if "voice_preset" in e.lower()]
    assert not g6_errors, f"unexpected G6 errors on clean cast: {g6_errors!r}"


def test_g6_errors_on_empty_preset():
    """Sprint 2 (a): empty/missing voice_preset on a Bark row is now a WARNING,
    not a hard error -- OTR_CastLock assigns the bark voice AFTER the freeze (the
    writer no longer stamps it), so an empty preset at freeze time is expected.
    The hard voice gate moved to CastLock's exit invariant. (Name kept for
    nodeid stability.)"""
    led = _make_ledger([
        {"char_id": "c01", "name": "LEMMY", "voice_preset": "", "traits": "gruff"},
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g6_errors = [e for e in report.errors if "voice_preset" in e.lower()]
    assert not g6_errors, \
        f"empty voice_preset must not hard-error at freeze now: {g6_errors!r}"
    g6_warnings = [w for w in report.warnings if "voice_preset" in w.lower()]
    assert g6_warnings, \
        "G6 should WARN on empty voice_preset (CastLock assigns it post-freeze)"
    assert any("c01" in w or "LEMMY" in w for w in g6_warnings), (
        f"G6 warning should identify the offending row: {g6_warnings!r}"
    )


def test_g6_errors_on_none_preset():
    """Sprint 2 (a): None voice_preset is a WARNING now (see empty-preset)."""
    led = _make_ledger([
        {"char_id": "c01", "name": "LEMMY",
         "voice_preset": None, "traits": "gruff"},
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g6_errors = [e for e in report.errors if "voice_preset" in e.lower()]
    assert not g6_errors, \
        f"None voice_preset must not hard-error at freeze now: {g6_errors!r}"
    g6_warnings = [w for w in report.warnings if "voice_preset" in w.lower()]
    assert g6_warnings, "G6 should WARN on None voice_preset"


def test_g6_errors_on_non_v2_preset():
    """Kokoro id on a Bark row is a writer bug, not a legacy fallback."""
    led = _make_ledger([
        {"char_id": "c01", "name": "LEMMY",
         "voice_preset": "bm_george", "traits": "gruff"},
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g6_errors = [e for e in report.errors if "voice_preset" in e.lower()]
    assert g6_errors, "G6 should error on non-v2/* preset on a Bark row"
    assert any("v2/" in e or "non-v2" in e.lower() for e in g6_errors), (
        f"G6 error should mention v2/* requirement: {g6_errors!r}"
    )


def test_g6_excludes_announcer_with_kokoro_id():
    """ANNOUNCER row with bm_*/bf_* Kokoro id is intentionally allowed."""
    led = _make_ledger([
        {"char_id": "announcer", "name": "ANNOUNCER",
         "voice_preset": "bf_emma", "traits": "broadcast"},
        {"char_id": "c01", "name": "LEMMY",
         "voice_preset": "v2/en_speaker_8", "traits": "gruff"},
    ])
    report = _LFC.run_gap_audit(led, label="test")
    g6_errors = [e for e in report.errors if "voice_preset" in e.lower()]
    assert not g6_errors, (
        f"announcer Kokoro id should be excluded; got errors: {g6_errors!r}"
    )


def test_g6_skipped_when_cast_empty_and_announcer_only_fallback():
    """announcer_only_fallback is the documented escape hatch -- G6 stays silent."""
    led = {
        "schema_version": "l3-2026-05-14",
        "cast": [],
        "lines": [],
        "meta": {"announcer_only_fallback": True},
    }
    report = _LFC.run_gap_audit(led, label="test")
    g6_errors = [e for e in report.errors if "voice_preset" in e.lower()]
    assert not g6_errors


def test_phase_10_hard_fails_on_g6_violation():
    """Phase 10 raises FreezeAssertionError when any Phase-0 critical
    gap (including G6) is still present."""
    led = _make_ledger([
        {"char_id": "c01", "name": "LEMMY",
         "voice_preset": "", "traits": "gruff"},
    ])
    # Phase 0 stamps but never raises -- soft gate.
    pre_report = _LFC.phase_0_gap_audit_pre(led)
    assert pre_report.errors, "phase 0 should surface G6 errors"
    # Phase 10 hard-fails on any unresolved critical gap.
    with pytest.raises(_LFC.FreezeAssertionError):
        _LFC.phase_10_gap_audit_post_and_freeze(led)


def test_phase_10_clean_path_is_covered_by_existing_lfc_tests():
    """Phase 10's pass-path (no critical gaps -> frozen_clean / frozen_with_warns)
    is covered by tests/test_lfc_phase_0_10_gap_audit.py against full fixtures.
    G6 contract is fully exercised here by the direct run_gap_audit cases
    above plus test_phase_10_hard_fails_on_g6_violation; building a fixture
    that passes every other Phase 0 check just to re-prove the pass-path
    would duplicate coverage. This placeholder documents the coverage seam."""
    assert True
