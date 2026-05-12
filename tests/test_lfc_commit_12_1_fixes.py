"""tests/test_lfc_commit_12_1_fixes.py

LFC sprint Commit 12.1 -- regression coverage for the seven bugs +
three wiring gaps the go-forward QA flagged.

  B1  Phase 4.5 cross-run idempotency
  B2  VRAM watchdog single-call wiring at cascade entry
  B3  make_polish_generate_fn exported in __all__
  B4  make_polish_generate_fn fallback logged at WARNING
  B5  Cleanup_passes contiguity (stub_bypassed records for stub phases)
  B6  Phase 4.5 intra-run ID uniqueness
  B7  Terminal-verdict short-circuit moved before mutation phases
  W3  _no_ledger_error_json synthetic-error-state consistency
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_freeze_cascade as _LFC_ORCH  # noqa: E402
from nodes import _otr_ledger_reviewer as _OTRLR  # noqa: E402
from nodes import _otr_lfc_smart_suggestion as _SS  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _line(line_id, role, char_id, text, *, skip=False):
    return {
        "line_id": line_id, "beat_id": line_id,
        "speaker_role": role, "char_id": char_id,
        "text": text, "skip": skip,
        "char_count": len(text),
        "word_count": sum(1 for _ in text.split()),
    }


def _ledger(lines, cast=None):
    return SimpleNamespace(
        data={
            "schema_version": "l3-2026-05-14",
            "episode_id": "ep_12_1_test",
            "cast": cast or [
                {"char_id": "c01", "name": "ALICE",
                 "portrait_path": "ALICE.png"},
                {"char_id": "c02", "name": "BOB",
                 "portrait_path": "BOB.png"},
                {"char_id": "announcer", "name": "ANNOUNCER",
                 "portrait_path": "ANNOUNCER.png"},
            ],
            "scenes": [], "shots": [], "beats": [],
            "lines": lines, "sfx": [], "music": [], "clips": [],
            "meta": {
                "episode_title": "Commit 12.1 fixture",
                "style": "mission_control_procedural",
            },
        },
        episode_id="ep_12_1_test",
    )


def _stub_reviewer(verdict="clean_no_edits"):
    def fake_review(_fn, _led):
        return _OTRLR.ReviewerDisposition(
            verdict=verdict,
            pre_audit_violations=0, pre_audit_repairs_applied=0,
            doctor_edits_proposed=0, doctor_edits_applied=0,
            post_audit_violations=0, phantom_skip_count=0,
        )
    return fake_review


# ---------------------------------------------------------------------------
# B1 -- Phase 4.5 cross-run idempotency
# ---------------------------------------------------------------------------


class TestB1Phase45Idempotency:
    def test_re_running_phase_4_5_is_no_op(self):
        led = _ledger([
            _line("l001", "character", "c01", "Start the car, Bob."),
            _line("l002", "character", "c02",
                  "She slammed the door shut."),
        ])
        # First run: synthesizes 2 SFX beats.
        rep1 = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep1.sfx_suggested == 2
        assert len(led.data["lines"]) == 4  # 2 original + 2 synth

        # Second run on the SAME ledger: should be a no-op because the
        # canonical sfx_tag fields are now present and dedupe via
        # _existing_sfx_tags catches them.
        rep2 = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep2.sfx_suggested == 0
        assert rep2.music_suggested == 0
        assert len(led.data["lines"]) == 4

    def test_re_running_phase_4_5_with_music_is_no_op(self):
        led = _ledger([
            _line("l001", "character", "c01",
                  "Then the music swelled to a crescendo."),
        ])
        rep1 = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep1.music_suggested == 1
        # Second run: ledger now has the synthesized music_inter, so
        # _ledger_has_music_inter is True and music synthesis is
        # suppressed.
        rep2 = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep2.music_suggested == 0

    def test_existing_sfx_tags_prefers_canonical_field(self):
        # User-stamped SFX (no sfx_tag) AND auto_generated SFX (with
        # sfx_tag) both contribute to the existing set; dedupe
        # works against either form.
        scene_lines = [
            {"line_id": "x", "speaker_role": "sfx", "char_id": "",
             "text": "SFX: car_engine_start",
             "auto_generated": True, "sfx_tag": "car_engine_start"},
            {"line_id": "y", "speaker_role": "sfx", "char_id": "",
             "text": "user-typed phone_ring"},
        ]
        tags = _SS._existing_sfx_tags({}, scene_lines)
        assert "car_engine_start" in tags
        # User-stamped variant lands in the set as a snake_case
        # token derived from the text.
        assert any("phone_ring" in t for t in tags)


# ---------------------------------------------------------------------------
# B6 -- Phase 4.5 intra-run ID uniqueness
# ---------------------------------------------------------------------------


class TestB6Phase45IntraRunIdUniqueness:
    def test_two_synth_entries_get_distinct_ids(self):
        # Two SFX-triggering verbs in one scene + one music-triggering
        # verb. Three synth entries; all line_ids and beat_ids must
        # be unique.
        led = _ledger([
            _line("l001", "character", "c01",
                  "She slammed the door behind her."),
            _line("l002", "character", "c01",
                  "I'll start the car."),
            _line("l003", "character", "c02",
                  "And then the music swelled to a peak."),
        ])
        rep = _SS.phase_4_5_smart_suggestion(led, enable=True)
        assert rep.sfx_suggested == 2
        assert rep.music_suggested == 1

        synth = [
            ln for ln in led.data["lines"]
            if ln.get("auto_generated")
        ]
        assert len(synth) == 3
        line_ids = [ln["line_id"] for ln in synth]
        beat_ids = [ln["beat_id"] for ln in synth]
        assert len(set(line_ids)) == 3, f"duplicate line_ids: {line_ids}"
        assert len(set(beat_ids)) == 3, f"duplicate beat_ids: {beat_ids}"


# ---------------------------------------------------------------------------
# B3 -- make_polish_generate_fn in __all__
# ---------------------------------------------------------------------------


class TestB3PolishFnInAll:
    def test_make_polish_generate_fn_in_all(self):
        from nodes import _otr_model_loader as _OTRML
        assert "make_polish_generate_fn" in _OTRML.__all__

    def test_star_import_surfaces_polish_factory(self):
        # `from nodes._otr_model_loader import *` should expose the
        # polish factory.
        ns: dict = {}
        exec(
            "from nodes._otr_model_loader import *",
            ns, ns,
        )
        assert "make_polish_generate_fn" in ns


# ---------------------------------------------------------------------------
# B4 -- make_polish_generate_fn fallback at WARNING
# ---------------------------------------------------------------------------


class TestB4PolishFallbackLogLevel:
    def test_fallback_logs_at_warning(self):
        # The cascade node's run() calls make_polish_generate_fn and
        # logs at WARNING on failure. Inspect the source to confirm
        # the call site uses log.warning (not log.debug). The string
        # is split across source lines via implicit concatenation
        # (`"make_polish_generate_fn " "unavailable (%s); ..."`) so
        # we normalize the source to ASCII-collapse that pattern
        # before searching.
        import re as _re
        from nodes import OTR_LedgerFreezeCascade as _LFC_NODE
        src = Path(_LFC_NODE.__file__).read_text(encoding="utf-8")
        assert "make_polish_generate_fn(cache_entry)" in src
        # Collapse all whitespace runs into single spaces, then
        # collapse `" "` (quote-space-quote) into nothing so
        # implicit-concat string literals join up.
        flat = _re.sub(r"\s+", " ", src)
        flat = flat.replace('" "', "")
        marker = "make_polish_generate_fn unavailable"
        idx = flat.find(marker)
        assert idx != -1, (
            "polish fallback log message not found in normalized "
            "source; B4 fix may have regressed or the message was "
            "renamed"
        )
        context = flat[max(0, idx - 400): idx]
        # log.warning must appear AFTER the most-recent log.debug
        # (rightmost search position wins).
        warn_pos = context.rfind("log.warning")
        debug_pos = context.rfind("log.debug")
        assert warn_pos != -1, (
            f"expected log.warning before {marker!r}; got: {context!r}"
        )
        assert warn_pos > debug_pos, (
            "log.debug appears more recent than log.warning in the "
            "fallback context; B4 fix regressed"
        )


# ---------------------------------------------------------------------------
# B5 -- cleanup_passes contiguity for stub phases
# ---------------------------------------------------------------------------


class TestB5CleanupPassesContiguous:
    def test_stub_phases_stamp_records(self):
        led = _ledger([
            _line("l001", "character", "c01", "Hello there."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        passes = led.data["meta"]["cleanup_passes"]
        names = [p["phase_name"] for p in passes]
        # Every cascade phase has a record, including stubs.
        for required in (
            "phase_0_gap_audit_pre",
            "phase_3_per_line_polish",
            "phase_1_2_9_reviewer_composite",
            "phase_4_per_scene_coherence",
            "phase_4_5_smart_suggestion",
            "phase_5_voice_drift",
            "phase_6_episode_arc",
            "phase_7_audio_readiness",
            "phase_8_video_readiness",
            "phase_10_gap_audit_post_and_freeze",
        ):
            assert required in names, (
                f"missing phase record {required} in cleanup_passes; "
                f"got {names}"
            )

    def test_stub_phase_records_carry_stub_bypassed_reason(self):
        # Phase 4 + 5 wiring landed in commits 8 + 9 -- they now stamp
        # real records, not stub_bypassed. The remaining stub is
        # phase_6 (lands in commit 10).
        led = _ledger([
            _line("l001", "character", "c01", "Hello there."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        passes = led.data["meta"]["cleanup_passes"]
        for stub_name in (
            "phase_6_episode_arc",
        ):
            stub_rec = next(
                p for p in passes if p["phase_name"] == stub_name
            )
            failures = stub_rec.get("failures", [])
            assert any(
                "stub_bypassed" in (f.get("reason") or "")
                for f in failures
            ), f"missing stub_bypassed on {stub_name}"


# ---------------------------------------------------------------------------
# B7 -- terminal-verdict short-circuit before mutation phases
# ---------------------------------------------------------------------------


class TestB7TerminalShortCircuitOrdering:
    @pytest.mark.parametrize("terminal_verdict", [
        "cast_unrecoverable",
        "too_many_edits",
        "needs_full_rerun",
        "post_audit_failed",
    ])
    def test_terminal_verdict_does_not_mutate_text(self, terminal_verdict):
        # Build a ledger with text that Phase 7 WOULD normalize
        # (Dr. -> Doctor) if it ran. On terminal verdict the cascade
        # short-circuits before Phase 7, so the text stays unchanged.
        led = _ledger([
            _line("l001", "character", "c01", "Mr. Smith arrived."),
        ])
        original = copy.deepcopy(led.data["lines"])

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer(terminal_verdict)):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        # Text is byte-identical to the input.
        assert led.data["lines"] == original
        # freeze_verdict matches the terminal verdict.
        assert led.data["meta"]["freeze_verdict"] == terminal_verdict
        # No cleanup_locked stamp (no Phase 10 success).
        assert "cleanup_locked" not in led.data["meta"]

    def test_terminal_verdict_records_terminal_skipped_phases(self):
        led = _ledger([
            _line("l001", "character", "c01", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("needs_full_rerun")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        passes = led.data["meta"]["cleanup_passes"]
        names = [p["phase_name"] for p in passes]
        # All mutation phases AND Phase 10 should be present with
        # terminal_skipped reason.
        for skipped in (
            "phase_4_per_scene_coherence",
            "phase_4_5_smart_suggestion",
            "phase_5_voice_drift",
            "phase_6_episode_arc",
            "phase_7_audio_readiness",
            "phase_8_video_readiness",
            "phase_10_gap_audit_post_and_freeze",
        ):
            assert skipped in names
            rec = next(p for p in passes if p["phase_name"] == skipped)
            failures = rec.get("failures", [])
            assert any(
                "terminal_skipped" in (f.get("reason") or "")
                for f in failures
            ), f"missing terminal_skipped on {skipped}"

    def test_terminal_verdict_audio_readiness_marked_skipped(self):
        led = _ledger([
            _line("l001", "character", "c01", "Mr. Smith."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("cast_unrecoverable")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        ar = led.data["meta"]["audio_readiness"]
        assert ar.get("skipped") is True
        assert ar.get("skipped_reason") == "terminal_skipped"


# ---------------------------------------------------------------------------
# B2 -- VRAM watchdog single-call wiring
# ---------------------------------------------------------------------------


class TestB2VramWatchdogWiring:
    def test_vram_stamped_at_cascade_entry(self):
        led = _ledger([
            _line("l001", "character", "c01", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        meta = led.data["meta"]
        # Both stamps present + numeric.
        assert "vram_at_cascade_entry_gb" in meta
        assert isinstance(meta["vram_at_cascade_entry_gb"], float)
        assert meta["vram_at_cascade_entry_gb"] >= 0.0
        assert "lfc_vram_ceiling_gb" in meta
        assert meta["lfc_vram_ceiling_gb"] == 14.0

    def test_over_ceiling_warns_but_cascade_completes(self, caplog):
        led = _ledger([
            _line("l001", "character", "c01", "Hello."),
        ])
        # Patch vram_over_ceiling on the watchdog module attribute
        # the orchestrator imports lazily. Patching
        # _torch_vram_allocated_gb directly is fragile because
        # vram_over_ceiling captures the default measure_fn at
        # function-definition time. Replacing vram_over_ceiling
        # itself sidesteps the issue.
        from nodes import _otr_lfc_watchdog as _WD

        def fake_over(ceiling_gb=14.0, **_kw):
            return (True, 20.0)

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")), \
             patch.object(_WD, "vram_over_ceiling",
                          side_effect=fake_over):
            with caplog.at_level("WARNING", logger="OTR.freeze_cascade"):
                disp = _LFC_ORCH.run_freeze_cascade(
                    lambda *a, **k: "", led, vram_ceiling_gb=14.0,
                )
        # Cascade still completes successfully.
        assert disp.verdict in ("frozen_clean", "frozen_with_warns")
        # WARN log fired with the expected text.
        warn_records = [
            r for r in caplog.records
            if r.levelname == "WARNING"
            and "VRAM" in r.getMessage()
            and "over" in r.getMessage()
        ]
        assert warn_records, (
            f"expected VRAM over-ceiling WARN log; got "
            f"{[(r.levelname, r.getMessage()) for r in caplog.records]}"
        )
        # Meta stamped with the over-ceiling reading.
        assert led.data["meta"]["vram_at_cascade_entry_gb"] == 20.0

    def test_custom_ceiling_propagates_to_meta(self):
        led = _ledger([
            _line("l001", "character", "c01", "Hello."),
        ])
        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=_stub_reviewer("clean_no_edits")):
            _LFC_ORCH.run_freeze_cascade(
                lambda *a, **k: "", led, vram_ceiling_gb=12.5,
            )
        assert led.data["meta"]["lfc_vram_ceiling_gb"] == 12.5


# ---------------------------------------------------------------------------
# W3 -- _no_ledger_error_json consistency
# ---------------------------------------------------------------------------


class TestW3NoLedgerErrorJsonConsistency:
    def test_empty_input_returns_synthetic_error_shape(self):
        from nodes.OTR_LedgerFreezeCascade import _no_ledger_error_json
        out = _no_ledger_error_json("")
        parsed = json.loads(out)
        assert parsed["schema_version"] == "synthetic_error_state"
        assert parsed["meta"]["freeze_verdict"] == "needs_full_rerun"
        assert parsed["meta"]["freeze_disposition"]["skipped"] is True
        assert parsed["meta"]["freeze_disposition"][
            "skipped_reason"
        ] == "no_writer_produced_ledger"

    def test_braces_input_returns_synthetic_error_shape(self):
        from nodes.OTR_LedgerFreezeCascade import _no_ledger_error_json
        out = _no_ledger_error_json("{}")
        parsed = json.loads(out)
        assert parsed["schema_version"] == "synthetic_error_state"

    def test_non_empty_input_still_returns_synthetic_error_shape(self):
        # W3 fix: pre-12.1 this case returned the incoming JSON
        # unchanged. Post-12.1 it returns the synthetic-error-state
        # shape consistently.
        from nodes.OTR_LedgerFreezeCascade import _no_ledger_error_json
        incoming = json.dumps({"schema_version": "l3-2026-05-14",
                                "meta": {"reviewer_verdict": "improved"},
                                "lines": [{"line_id": "l001"}]})
        out = _no_ledger_error_json(incoming)
        parsed = json.loads(out)
        # Schema_version is the synthetic shape, NOT the incoming l3.
        assert parsed["schema_version"] == "synthetic_error_state"
        # freeze_verdict is the consistent error signal.
        assert parsed["meta"]["freeze_verdict"] == "needs_full_rerun"
        # Forensic detail preserved (truncated).
        assert parsed["meta"]["freeze_disposition"][
            "skipped_reason_detail"
        ]
        assert "schema_version" in parsed["meta"]["freeze_disposition"][
            "skipped_reason_detail"
        ]

    def test_long_incoming_input_truncated_to_200_chars(self):
        from nodes.OTR_LedgerFreezeCascade import _no_ledger_error_json
        incoming = "x" * 500
        out = _no_ledger_error_json(incoming)
        parsed = json.loads(out)
        detail = parsed["meta"]["freeze_disposition"][
            "skipped_reason_detail"
        ]
        assert len(detail) == 200
