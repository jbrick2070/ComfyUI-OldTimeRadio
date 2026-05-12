"""tests/test_lfc_phase_0_10_gap_audit.py

LFC sprint Commit 1 -- regression coverage for Phase 0 + Phase 10 of
the Ledger Freeze Cascade.

Covers:
  * Clean ledger passes both phases (frozen_clean).
  * Phase 0 is warn-mode (NEVER raises) even on critical gaps.
  * Phase 10 raises FreezeAssertionError on critical gaps.
  * Phase 10 stamps cleanup_locked + freeze_timestamp + freeze_verdict.
  * Each ADR §7 per-line invariant violation is detected.
  * Each ADR §7 per-cast invariant violation is detected.
  * Each ADR §7 meta invariant violation is detected.
  * Each ADR §6.16 null-rejection invariant is detected.
  * announcer-only-fallback exemption works.
  * Both Ledger objects and bare dicts are accepted by the public API.
  * Phase 10 is idempotent on an already-frozen ledger.
"""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from nodes import _otr_ledger_freeze as _LFC


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _clean_ledger_data() -> dict:
    """Minimal-but-valid L3 ledger that should pass both phases clean."""
    return {
        "schema_version": _LFC.EXPECTED_SCHEMA_VERSION,
        "episode_id": "ep_test_001",
        "cast": [
            {
                "char_id": "c01",
                "name": "ALICE",
                "traits": "calm, methodical engineer",
                "voice_preset": "v2/en_speaker_6",
            },
            {
                "char_id": "c02",
                "name": "BOB",
                "traits": "skeptical analyst",
                "voice_preset": "v2/en_speaker_2",
            },
            {
                "char_id": "announcer",
                "name": "ANNOUNCER",
                "traits": "warm broadcast voice",
                "voice_preset": "v2/en_speaker_9",
            },
        ],
        "scenes": [],
        "shots": [],
        "beats": [
            {"beat_id": "b001"},
            {"beat_id": "b002"},
            {"beat_id": "b003"},
            {"beat_id": "b004"},
        ],
        "lines": [
            {
                "line_id": "l001",
                "beat_id": "b001",
                "speaker_role": "announcer",
                "char_id": "announcer",
                "text": "Tonight, on Signal Lost.",
                "word_count": 4,
                "char_count": len("Tonight, on Signal Lost."),
                "skip": False,
            },
            {
                "line_id": "l002",
                "beat_id": "b002",
                "speaker_role": "character",
                "char_id": "c01",
                "text": "Bob, the readings are off the chart.",
                "word_count": 7,
                "char_count": len("Bob, the readings are off the chart."),
                "skip": False,
            },
            {
                "line_id": "l003",
                "beat_id": "b003",
                "speaker_role": "character",
                "char_id": "c02",
                "text": "Then someone tampered with the calibration.",
                "word_count": 6,
                "char_count": len(
                    "Then someone tampered with the calibration."
                ),
                "skip": False,
            },
            {
                "line_id": "l004",
                "beat_id": "b004",
                "speaker_role": "announcer",
                "char_id": "announcer",
                "text": "Stay tuned.",
                "word_count": 2,
                "char_count": len("Stay tuned."),
                "skip": False,
            },
        ],
        "sfx": [],
        "music": [],
        "clips": [],
        "meta": {
            "episode_title": "The Calibration Drift",
            "style": "mission_control_procedural",
            "news": {"key_terms": ["calibration", "readings"]},
        },
    }


def _ledger_obj(data: dict) -> SimpleNamespace:
    """Stand-in for production_ledger.Ledger -- exposes `.data`."""
    return SimpleNamespace(data=data)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestCleanLedger:
    def test_run_gap_audit_clean(self):
        rep = _LFC.run_gap_audit(_clean_ledger_data(), label="pre")
        assert rep.errors == []
        assert rep.warnings == []
        assert rep.info["line_count"] == 4
        assert rep.info["voiced_beat_count"] == 4
        assert rep.is_clean is True

    def test_phase_0_clean_stamps_audit(self):
        data = _clean_ledger_data()
        rep = _LFC.phase_0_gap_audit_pre(data)
        assert rep.is_clean
        assert data["meta"]["gap_audit_pre"]["errors"] == []
        assert data["meta"]["gap_audit_pre"]["warnings"] == []
        # Phase 0 does NOT stamp the freeze lock.
        assert "cleanup_locked" not in data["meta"]
        assert "freeze_timestamp" not in data["meta"]
        assert "freeze_verdict" not in data["meta"]

    def test_phase_10_clean_stamps_freeze(self):
        data = _clean_ledger_data()
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert rep.is_clean
        meta = data["meta"]
        assert meta["cleanup_locked"] is True
        assert meta["freeze_verdict"] == "frozen_clean"
        # iso8601 UTC parses cleanly.
        ts = meta["freeze_timestamp"]
        assert isinstance(ts, str)
        parsed = datetime.fromisoformat(ts)
        assert parsed.tzinfo is not None

    def test_phase_10_warn_path_stamps_frozen_with_warns(self):
        data = _clean_ledger_data()
        # Drop meta.episode_title to trigger a soft warning.
        del data["meta"]["episode_title"]
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert rep.errors == []
        assert rep.warnings
        assert data["meta"]["freeze_verdict"] == "frozen_with_warns"
        assert data["meta"]["cleanup_locked"] is True

    def test_phase_10_accepts_ledger_obj(self):
        led = _ledger_obj(_clean_ledger_data())
        rep = _LFC.phase_10_gap_audit_post_and_freeze(led)
        assert rep.is_clean
        assert led.data["meta"]["cleanup_locked"] is True

    def test_phase_10_idempotent(self):
        data = _clean_ledger_data()
        rep1 = _LFC.phase_10_gap_audit_post_and_freeze(data)
        ts1 = data["meta"]["freeze_timestamp"]
        rep2 = _LFC.phase_10_gap_audit_post_and_freeze(data)
        ts2 = data["meta"]["freeze_timestamp"]
        assert rep1.is_clean
        assert rep2.is_clean
        # Allow timestamp to advance (idempotent stamp -- not byte-identical
        # but always-valid). Both verdicts should be frozen_clean.
        assert data["meta"]["freeze_verdict"] == "frozen_clean"
        assert ts1 <= ts2
        assert data["meta"]["cleanup_locked"] is True


# ---------------------------------------------------------------------------
# Phase 0 = warn-only contract
# ---------------------------------------------------------------------------


class TestPhase0NeverRaises:
    def test_phase_0_advances_on_critical_gap(self):
        data = _clean_ledger_data()
        # Remove a voiced line's char_id -- critical gap, but Phase 0
        # must still advance (warn-mode).
        data["lines"][1]["char_id"] = ""
        rep = _LFC.phase_0_gap_audit_pre(data)
        assert rep.errors  # critical detected
        # Phase 0 STAMPS but DOES NOT raise.
        assert data["meta"]["gap_audit_pre"]["errors"] == rep.errors
        assert "cleanup_locked" not in data["meta"]


# ---------------------------------------------------------------------------
# Phase 10 hard-fail
# ---------------------------------------------------------------------------


class TestPhase10HardFail:
    def test_voiced_line_missing_char_id_raises(self):
        data = _clean_ledger_data()
        data["lines"][1]["char_id"] = ""
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert ei.value.errors
        assert "char_id" in ei.value.errors[0]
        assert data["meta"]["freeze_verdict"] == "needs_full_rerun"
        # cleanup_locked was NEVER stamped (no prior success).
        assert "cleanup_locked" not in data["meta"]

    def test_duplicate_line_id_raises(self):
        data = _clean_ledger_data()
        data["lines"][1]["line_id"] = data["lines"][0]["line_id"]
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("duplicated" in e for e in ei.value.errors)

    def test_bad_speaker_role_raises(self):
        data = _clean_ledger_data()
        data["lines"][2]["speaker_role"] = "narrator"
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("narrator" in e for e in ei.value.errors)

    def test_skip_true_with_nonempty_text_raises(self):
        data = _clean_ledger_data()
        data["lines"][1]["skip"] = True
        # text intentionally NOT cleared -- inconsistent state
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any(
            "skip=True AND non-empty text" in e for e in ei.value.errors
        )

    def test_voiced_line_empty_text_no_skip_raises(self):
        data = _clean_ledger_data()
        data["lines"][2]["text"] = ""
        data["lines"][2]["char_count"] = 0
        data["lines"][2]["word_count"] = 0
        data["lines"][2]["skip"] = False
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any(
            "voiced, not skipped" in e and "empty text" in e
            for e in ei.value.errors
        )

    def test_empty_cast_no_announcer_only_raises(self):
        data = _clean_ledger_data()
        data["cast"] = []
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("cast is empty" in e for e in ei.value.errors)

    def test_bad_schema_version_raises(self):
        data = _clean_ledger_data()
        data["schema_version"] = "l2-2026-04-25"
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("schema_version" in e for e in ei.value.errors)

    def test_freeze_assertion_carries_report(self):
        data = _clean_ledger_data()
        data["lines"][1]["char_id"] = ""
        try:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        except _LFC.FreezeAssertionError as exc:
            assert isinstance(exc.report, _LFC.GapAuditReport)
            assert exc.report.has_critical_gaps
            assert exc.errors == exc.report.errors


# ---------------------------------------------------------------------------
# Null-rejection (§6.16)
# ---------------------------------------------------------------------------


class TestNullRejection:
    @pytest.mark.parametrize(
        "field", ["cast", "lines", "beats", "scenes", "shots",
                  "sfx", "music", "clips"],
    )
    def test_required_top_level_list_missing_raises(self, field):
        data = _clean_ledger_data()
        del data[field]
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any(field in e and "MISSING" in e for e in ei.value.errors)

    @pytest.mark.parametrize(
        "field", ["cast", "lines", "beats", "scenes", "shots",
                  "sfx", "music", "clips"],
    )
    def test_required_top_level_list_null_raises(self, field):
        data = _clean_ledger_data()
        data[field] = None
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any(field in e and "null" in e for e in ei.value.errors)

    @pytest.mark.parametrize(
        "field", ["cast", "lines", "beats", "scenes", "shots",
                  "sfx", "music", "clips"],
    )
    def test_required_top_level_list_wrong_type_raises(self, field):
        data = _clean_ledger_data()
        data[field] = {}
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any(
            field in e and "expected list" in e for e in ei.value.errors
        )

    def test_news_key_terms_null_raises(self):
        data = _clean_ledger_data()
        data["meta"]["news"]["key_terms"] = None
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("key_terms" in e for e in ei.value.errors)

    def test_news_key_terms_wrong_type_raises(self):
        data = _clean_ledger_data()
        data["meta"]["news"]["key_terms"] = "calibration"
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("key_terms" in e for e in ei.value.errors)

    def test_meta_wrong_type_raises(self):
        data = _clean_ledger_data()
        data["meta"] = []
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any(
            "meta" in e and "expected dict" in e for e in ei.value.errors
        )

    def test_tts_skip_reason_null_raises(self):
        data = _clean_ledger_data()
        data["lines"][1]["tts_skip_reason"] = None
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        # tts_skip_reason None on a non-skipped line is rejected.
        assert any("tts_skip_reason" in e for e in ei.value.errors)


# ---------------------------------------------------------------------------
# Per-cast invariants
# ---------------------------------------------------------------------------


class TestPerCastInvariants:
    def test_cast_row_missing_char_id_raises(self):
        data = _clean_ledger_data()
        data["cast"][0]["char_id"] = ""
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("char_id" in e for e in ei.value.errors)

    def test_cast_row_missing_name_raises(self):
        data = _clean_ledger_data()
        data["cast"][0]["name"] = ""
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("name" in e for e in ei.value.errors)

    def test_duplicate_char_id_raises(self):
        data = _clean_ledger_data()
        data["cast"][1]["char_id"] = data["cast"][0]["char_id"]
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("duplicated" in e for e in ei.value.errors)

    def test_announcer_only_fallback_allows_empty_cast(self):
        data = _clean_ledger_data()
        data["cast"] = []
        data["lines"] = [
            {
                "line_id": "l001",
                "beat_id": "b001",
                "speaker_role": "announcer",
                "char_id": "announcer",
                "text": "And we're back.",
                "word_count": 3,
                "char_count": len("And we're back."),
                "skip": False,
            },
        ]
        data["meta"]["announcer_only_fallback"] = True
        # Should not raise -- but the line still has a char_id, so it's
        # still gap-clean. (We only suppress the empty-cast critical.)
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert rep.is_clean or not rep.errors
        assert data["meta"]["cleanup_locked"] is True

    def test_unreferenced_char_id_is_warning_not_error(self):
        data = _clean_ledger_data()
        # Add an extra cast member nobody speaks for.
        data["cast"].append({
            "char_id": "c99",
            "name": "GHOST",
            "traits": "never appears",
            "voice_preset": "v2/en_speaker_4",
        })
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        # Soft gap -> freeze still completes.
        assert rep.errors == []
        assert any("c99" in w for w in rep.warnings)
        assert data["meta"]["freeze_verdict"] == "frozen_with_warns"

    def test_traits_wrong_type_raises(self):
        data = _clean_ledger_data()
        data["cast"][0]["traits"] = 42
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("traits" in e for e in ei.value.errors)


# ---------------------------------------------------------------------------
# Meta invariants
# ---------------------------------------------------------------------------


class TestMetaInvariants:
    def test_episode_title_wrong_type_raises(self):
        data = _clean_ledger_data()
        data["meta"]["episode_title"] = 123
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("episode_title" in e for e in ei.value.errors)

    def test_episode_title_missing_is_warning(self):
        data = _clean_ledger_data()
        del data["meta"]["episode_title"]
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert rep.errors == []
        assert any("episode_title" in w for w in rep.warnings)
        assert data["meta"]["freeze_verdict"] == "frozen_with_warns"

    def test_cleanup_passes_wrong_type_raises(self):
        data = _clean_ledger_data()
        data["meta"]["cleanup_passes"] = {"phase_0": True}
        with pytest.raises(_LFC.FreezeAssertionError) as ei:
            _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert any("cleanup_passes" in e for e in ei.value.errors)


# ---------------------------------------------------------------------------
# Word-count / char-count drift = WARN, not ERROR
# ---------------------------------------------------------------------------


class TestSoftGapsAdvanceCleanly:
    def test_word_count_drift_is_warning(self):
        data = _clean_ledger_data()
        # Synthetic drift.
        data["lines"][1]["word_count"] = 999
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert rep.errors == []
        assert any("word_count" in w for w in rep.warnings)
        assert data["meta"]["cleanup_locked"] is True

    def test_char_count_drift_is_warning(self):
        data = _clean_ledger_data()
        data["lines"][1]["char_count"] = 999
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert rep.errors == []
        assert any("char_count" in w for w in rep.warnings)

    def test_orphan_beat_reference_is_warning(self):
        data = _clean_ledger_data()
        data["lines"][1]["beat_id"] = "b_orphan"
        rep = _LFC.phase_10_gap_audit_post_and_freeze(data)
        assert rep.errors == []
        assert any("b_orphan" in w for w in rep.warnings)


# ---------------------------------------------------------------------------
# Report / dataclass surface
# ---------------------------------------------------------------------------


class TestGapAuditReport:
    def test_is_clean_true_when_no_errors_no_warnings(self):
        rep = _LFC.GapAuditReport(label="pre")
        assert rep.is_clean
        assert not rep.has_critical_gaps

    def test_is_clean_false_when_warnings(self):
        rep = _LFC.GapAuditReport(label="pre", warnings=["soft"])
        assert not rep.is_clean
        assert not rep.has_critical_gaps

    def test_has_critical_gaps_true_when_errors(self):
        rep = _LFC.GapAuditReport(label="pre", errors=["critical"])
        assert rep.has_critical_gaps
        assert not rep.is_clean

    def test_constants_exposed(self):
        assert _LFC.EXPECTED_SCHEMA_VERSION == "l3-2026-05-14"
        assert "frozen_clean" in _LFC.ALLOWED_FREEZE_VERDICTS
        assert "frozen_with_warns" in _LFC.ALLOWED_FREEZE_VERDICTS
        assert "needs_full_rerun" in _LFC.ALLOWED_FREEZE_VERDICTS
        assert "character" in _LFC.ALLOWED_SPEAKER_ROLES
        assert "announcer" in _LFC.ALLOWED_SPEAKER_ROLES
