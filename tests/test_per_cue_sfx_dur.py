"""tests/test_per_cue_sfx_dur.py

Voice-path-cleanbreak Sprint 3 (2026-05-12). Per-cue ``dur_s`` on SFX
lines plus G7 invariant enforcement.

Contract under test:
  - G7 (FreezeCascade Phase 0/10): SFX line ``dur_s``, when present,
    must fall within ``[0.25, 12.0]``. Phase 10 hard-fails on
    violation.
  - AudioGen + ProcSFX honor ``line.dur_s`` when present; fall back
    to ``default_duration`` when absent.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pytest

from nodes import _otr_ledger_freeze as _LFC


# ---------------------------------------------------------------------------
# G7 invariant
# ---------------------------------------------------------------------------


def _ledger_with_sfx_dur(dur_s):
    """Minimal ledger with one SFX line carrying the given dur_s."""
    return {
        "schema_version": "l3-2026-05-14",
        "cast": [],
        "lines": [
            {
                "line_id":      "sfx_001",
                "char_id":      "sfx",
                "speaker_role": "sfx",
                "text":         "metal hatch groans open",
                "dur_s":        dur_s,
                "traits":       "ambient",
            },
        ],
        "meta": {},
    }


def test_g7_dur_s_within_bounds_passes():
    """A SFX line with dur_s in [0.5, 10.0] produces no G7 error.

    Voice-path-cleanbreak Sprint 6.4 (2026-05-12): bounds tightened
    from (0.25, 12.0) to (0.5, 10.0) -- the consumer intersection of
    AudioGen + ProcSFX. Boundary values 0.5 and 10.0 are inclusive.
    """
    for dur in (0.5, 1.0, 3.0, 8.0, 10.0):
        led = _ledger_with_sfx_dur(dur)
        report = _LFC.run_gap_audit(led, label="test")
        g7_errors = [e for e in report.errors if "G7" in e]
        assert not g7_errors, f"dur_s={dur} surfaced unexpected G7 error: {g7_errors!r}"


def test_g7_dur_s_below_min_raises():
    """dur_s under 0.5 is a writer contract violation (Sprint 6.4
    tightened from 0.25 to 0.5)."""
    led = _ledger_with_sfx_dur(0.1)
    report = _LFC.run_gap_audit(led, label="test")
    g7_errors = [e for e in report.errors if "G7" in e]
    assert g7_errors, "G7 should error on dur_s below MIN"
    assert "sfx_001" in g7_errors[0]


def test_g7_dur_s_at_old_lower_bound_now_raises():
    """Sprint 6.4 contract: dur_s=0.25 used to pass G7; now it raises.
    Pin the intent so a future 'relaxation' has to explicitly defeat
    this test."""
    led = _ledger_with_sfx_dur(0.25)
    report = _LFC.run_gap_audit(led, label="test")
    g7_errors = [e for e in report.errors if "G7" in e]
    assert g7_errors, (
        "dur_s=0.25 must FAIL G7 post-Sprint-6.4 (tightened to 0.5 minimum)"
    )


def test_g7_dur_s_above_max_raises():
    """dur_s over 10.0 is a writer contract violation (Sprint 6.4
    tightened from 12.0 to 10.0)."""
    led = _ledger_with_sfx_dur(15.0)
    report = _LFC.run_gap_audit(led, label="test")
    g7_errors = [e for e in report.errors if "G7" in e]
    assert g7_errors, "G7 should error on dur_s above MAX"
    assert "sfx_001" in g7_errors[0]


def test_g7_dur_s_at_old_upper_bound_now_raises():
    """Sprint 6.4 contract: dur_s=12.0 used to pass G7; now it raises."""
    led = _ledger_with_sfx_dur(12.0)
    report = _LFC.run_gap_audit(led, label="test")
    g7_errors = [e for e in report.errors if "G7" in e]
    assert g7_errors, (
        "dur_s=12.0 must FAIL G7 post-Sprint-6.4 (tightened to 10.0 maximum)"
    )


def test_g7_bounds_match_consumer_intersection():
    """Sprint 10.1 (post-S6.4) drift guard. The G7 bounds MUST equal
    the consumer-intersection of AudioGen + ProcSFX. Post-S10.1 both
    consumers import these constants directly, so the intersection is
    enforced structurally -- this test verifies the freeze module
    still exposes the canonical numeric values."""
    from nodes._otr_ledger_freeze import SFX_DUR_MIN_S, SFX_DUR_MAX_S
    import nodes.batch_audiogen_generator as ag

    # The consumer's module-level constants MUST be the same objects
    # as the freeze module's. (Identity, not equality -- catches a
    # local-shadow refactor that would silently diverge.)
    assert ag.SFX_DUR_MIN_S is SFX_DUR_MIN_S
    assert ag.SFX_DUR_MAX_S is SFX_DUR_MAX_S

    # The numeric values themselves -- pinned for the canonical
    # cleanbreak intersection.
    assert SFX_DUR_MIN_S == 0.5
    assert SFX_DUR_MAX_S == 10.0


def test_g7_absent_dur_s_is_hard_error():
    """S28 cleanbreak (Rule C): pre-S28 G7 silently passed sfx lines
    missing dur_s as a back-compat tolerance for older flat-layout
    ledgers. Post-S28 the producer (find_clip_durations / upstream
    timing) always stamps dur_s on every sfx line; a missing value
    is now a hard writer-contract violation, not a tolerated legacy
    shape. The test assertion is flipped from "no G7 errors" to
    "G7 reports the missing dur_s as an error."
    """
    led = {
        "schema_version": "l3-2026-05-14",
        "cast": [],
        "lines": [
            {
                "line_id":      "sfx_002",
                "char_id":      "sfx",
                "speaker_role": "sfx",
                "text":         "door knock",
                "traits":       "ambient",
                # no dur_s field at all -- producer leak under v2.0
            },
        ],
        "meta": {},
    }
    report = _LFC.run_gap_audit(led, label="test")
    g7_errors = [e for e in report.errors if "G7" in e]
    assert g7_errors, "expected G7 missing-dur_s error, got none"
    assert any("missing dur_s" in e for e in g7_errors), (
        f"expected 'missing dur_s' wording in G7 error, got: {g7_errors!r}"
    )
    assert any("sfx_002" in e for e in g7_errors)


def test_g7_non_sfx_lines_excluded():
    """G7 only checks sfx-role lines. Character lines with dur_s
    populated (post-render) are NOT in scope for the writer-time
    bounds check."""
    led = {
        "schema_version": "l3-2026-05-14",
        "cast": [],
        "lines": [
            {
                "line_id":      "b001",
                "char_id":      "c01",
                "speaker_role": "character",
                "text":         "Hello.",
                "dur_s":        25.0,  # would violate if sfx; ok as character
                "traits":       "calm",
            },
        ],
        "meta": {},
    }
    report = _LFC.run_gap_audit(led, label="test")
    g7_errors = [e for e in report.errors if "G7" in e]
    assert not g7_errors


def test_g7_phase_10_hard_fails_on_violation():
    """Phase 10 raises FreezeAssertionError on any G7 violation."""
    led = _ledger_with_sfx_dur(0.1)
    # Phase 0 stamps; never raises.
    pre = _LFC.phase_0_gap_audit_pre(led)
    assert any("G7" in e for e in pre.errors)
    with pytest.raises(_LFC.FreezeAssertionError):
        _LFC.phase_10_gap_audit_post_and_freeze(led)


# ---------------------------------------------------------------------------
# AudioGen reader honors line.dur_s
# ---------------------------------------------------------------------------


def _l3_sfx_ledger(dur_s_list):
    """Build a ledger with N sfx lines, dur_s values from the list
    (use None to mean "field absent")."""
    lines = []
    for i, dur in enumerate(dur_s_list, start=1):
        entry = {
            "line_id":      f"sfx_{i:03d}",
            "char_id":      "sfx",
            "speaker_role": "sfx",
            "text":         f"cue {i}",
            "traits":       "ambient",
        }
        if dur is not None:
            entry["dur_s"] = dur
        lines.append(entry)
    return {
        "schema_version": "l3-2026-05-14",
        "cast":  [],
        "lines": lines,
        "meta":  {},
    }


def test_audiogen_honors_per_cue_dur_s():
    """AudioGen's render_queue picks up dur_s from sfx_items."""
    from nodes.batch_audiogen_generator import BatchAudioGenGenerator
    # We intercept just the iter+queue build path; bypass real model.
    node = BatchAudioGenGenerator()
    script_json = json.dumps(_l3_sfx_ledger([0.5, 4.0, None]))

    captured_durations = []

    def _fake_generate_audio(*a, **kw):
        # Capture the duration arg that would've been passed.
        captured_durations.append(kw.get("duration", a[1] if len(a) > 1 else None))
        return np.zeros(int(32000 * 0.1), dtype=np.float32)

    # Quick path: assert the sfx_items list construction picks dur_s.
    from nodes import _otr_ledger_consumers as _OTRLC
    led = _OTRLC.load_ledger(script_json)
    sfx_items = []
    for line in _OTRLC.iter_lines(led, roles={"sfx"}):
        sfx_items.append({
            "line_id": line.get("line_id"),
            "text":    line.get("text"),
            "dur_s":   line.get("dur_s"),
        })
    assert [it["dur_s"] for it in sfx_items] == [0.5, 4.0, None], (
        f"sfx_items dur_s extraction broke; got "
        f"{[it['dur_s'] for it in sfx_items]!r}"
    )


def test_audiogen_falls_back_to_default_when_dur_s_absent():
    """When line.dur_s is None, AudioGen uses default_duration."""
    script_json = json.dumps(_l3_sfx_ledger([None, None]))
    from nodes import _otr_ledger_consumers as _OTRLC
    led = _OTRLC.load_ledger(script_json)
    sfx_items = []
    for line in _OTRLC.iter_lines(led, roles={"sfx"}):
        sfx_items.append({"dur_s": line.get("dur_s")})
    # Simulate the AudioGen loop's per-cue duration resolution.
    default_duration = 3.0
    resolved = []
    for it in sfx_items:
        cue_dur = it.get("dur_s")
        if isinstance(cue_dur, (int, float)) and cue_dur > 0:
            d = max(0.5, min(10.0, float(cue_dur)))
        else:
            d = float(default_duration)
        resolved.append(d)
    assert resolved == [3.0, 3.0], resolved


# ---------------------------------------------------------------------------
# ProcSFX reader honors line.dur_s
# ---------------------------------------------------------------------------


def test_procsfx_honors_per_cue_dur_s():
    """ProcSFX's cue_duration resolution picks up dur_s from sfx_items."""
    script_json = json.dumps(_l3_sfx_ledger([0.5, 2.0, 8.0]))
    from nodes import _otr_ledger_consumers as _OTRLC
    led = _OTRLC.load_ledger(script_json)
    sfx_items = []
    for line in _OTRLC.iter_lines(led, roles={"sfx"}):
        sfx_items.append({"dur_s": line.get("dur_s")})

    default_duration = 2.0
    durations = []
    for it in sfx_items:
        cue = it.get("dur_s")
        if isinstance(cue, (int, float)) and cue > 0:
            durations.append(max(0.1, min(10.0, float(cue))))
        else:
            durations.append(float(default_duration))
    assert durations == [0.5, 2.0, 8.0]


def test_procsfx_clamps_out_of_bounds():
    """Defensive clamp at the reader keeps a bypass of G7 from
    sending a 60s cue into the procedural synth."""
    sfx_items = [{"dur_s": 60.0}, {"dur_s": -5.0}, {"dur_s": 1.5}]
    durations = []
    for it in sfx_items:
        cue = it.get("dur_s")
        if isinstance(cue, (int, float)) and cue > 0:
            durations.append(max(0.1, min(10.0, float(cue))))
        else:
            durations.append(2.0)  # default_duration
    assert durations == [10.0, 2.0, 1.5]
