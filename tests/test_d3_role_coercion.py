"""D3 (2026-06-22, story-quality lift) -- b011 role mis-stamp coercion.

A character line whose char_id is a real cast id must carry
speaker_role="character". The b011 "Chandra's Echo" defect: the reviewer's
role_mismatch repair honoured an LLM expected="announcer" on a c02 cast row.

Coverage: the coercion helper + cast_ids extractor (production_ledger),
set_lines (site b), the reviewer role_mismatch repair guard (site a, the
culprit), and the MANDATORY pre-freeze sweep in run_freeze_cascade (site c) +
its CI invariant. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import production_ledger as PL  # noqa: E402
from nodes import _otr_freeze_cascade as _LFC_ORCH  # noqa: E402
from nodes import _otr_ledger_reviewer as _OTRLR  # noqa: E402


# ---------------------------------------------------------------------------
# Unit: coerce_speaker_role_for_char_id + cast_ids_from_ledger
# ---------------------------------------------------------------------------

CAST = [
    {"char_id": "c01", "name": "MALI"},
    {"char_id": "c02", "name": "MANFRED"},
    {"char_id": "announcer", "name": "ANNOUNCER"},
]


class TestCastIdsFromLedger:
    def test_excludes_announcer_and_music(self):
        led = {"cast": CAST + [{"char_id": "music_inter", "name": "MUSIC"}]}
        assert PL.cast_ids_from_ledger(led) == {"c01", "c02"}

    def test_empty_safe(self):
        assert PL.cast_ids_from_ledger({}) == set()
        assert PL.cast_ids_from_ledger(None) == set()

    def test_announcer_in_cast_slot_excluded(self):
        # live-smoke 2026-06-22: the announcer is often stored as an ordinary
        # cast slot (char_id=c01, name=ANNOUNCER) rather than the "announcer"
        # sentinel. It must NOT count as a coercible character.
        led = {"cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "tts_model": "kokoro"},
            {"char_id": "c02", "name": "NORA ECKELS", "tts_model": "bark"},
        ]}
        assert PL.cast_ids_from_ledger(led) == {"c02"}

    def test_announcer_intro_cast_slot_not_coerced(self):
        # the announcer's intro line carries its cast char_id (c01) -> must stay
        # whatever role it has, never forced to "character".
        line = {"line_id": "b001", "char_id": "c01", "speaker_role": "announcer",
                "compose_flags": ["announcer_intro"]}
        cast_ids = PL.cast_ids_from_ledger({"cast": [
            {"char_id": "c01", "name": "ANNOUNCER", "tts_model": "kokoro"},
            {"char_id": "c02", "name": "NORA ECKELS", "tts_model": "bark"},
        ]})
        _out, changed = PL.coerce_speaker_role_for_char_id(line, cast_ids, "test")
        assert changed is False
        assert line["speaker_role"] == "announcer"


class TestCoerceHelper:
    def test_cast_charid_announcer_role_coerced(self):
        line = {"line_id": "b011", "char_id": "c02", "speaker_role": "announcer"}
        out, changed = PL.coerce_speaker_role_for_char_id(line, {"c01", "c02"}, "test")
        assert changed is True
        assert out["speaker_role"] == "character"
        flags = out.get("compose_flags") or []
        assert any(f.startswith("role_coerce:") and "prev=announcer" in f for f in flags)

    def test_legit_announcer_untouched(self):
        line = {"line_id": "b001", "char_id": "announcer", "speaker_role": "announcer"}
        out, changed = PL.coerce_speaker_role_for_char_id(line, {"c01", "c02"}, "test")
        assert changed is False
        assert out["speaker_role"] == "announcer"

    def test_non_cast_charid_noop(self):
        line = {"line_id": "x", "char_id": "c99", "speaker_role": "announcer"}
        _out, changed = PL.coerce_speaker_role_for_char_id(line, {"c01", "c02"}, "t")
        assert changed is False

    def test_music_sentinel_noop(self):
        line = {"line_id": "m", "char_id": "music_inter", "speaker_role": "music_inter"}
        _out, changed = PL.coerce_speaker_role_for_char_id(line, {"c01", "c02"}, "t")
        assert changed is False

    def test_already_character_noop(self):
        line = {"line_id": "b002", "char_id": "c01", "speaker_role": "character"}
        _out, changed = PL.coerce_speaker_role_for_char_id(line, {"c01", "c02"}, "t")
        assert changed is False

    def test_non_dict_safe(self):
        out, changed = PL.coerce_speaker_role_for_char_id("not a dict", {"c01"}, "t")
        assert changed is False and out == "not a dict"


# ---------------------------------------------------------------------------
# Site (b): production_ledger.set_lines
# ---------------------------------------------------------------------------

class TestSetLinesCoercion:
    def test_set_lines_coerces_cast_announcer_row(self, tmp_path):
        led = PL.Ledger(episode_id="ep_test_d3", out_dir=str(tmp_path))
        led.data["cast"] = list(CAST)
        led.set_lines([
            {"line_id": "b011", "char_id": "c02", "speaker_role": "announcer",
             "text": "I've sent a copy to The Chronicle."},
            {"line_id": "b001", "char_id": "announcer", "speaker_role": "announcer",
             "text": "Gather round, listeners."},
        ])
        rows = {r["line_id"]: r for r in led.data["lines"]}
        assert rows["b011"]["speaker_role"] == "character"
        assert rows["b001"]["speaker_role"] == "announcer"


# ---------------------------------------------------------------------------
# Site (a): reviewer role_mismatch repair guard (the culprit)
# ---------------------------------------------------------------------------

def _candidate_ledger():
    return {
        "schema_version": "l3-2026-05-14",
        "cast": list(CAST),
        "lines": [
            {"line_id": "b011", "char_id": "c02", "speaker_role": "announcer",
             "text": "I've sent a copy of your research to The Chronicle."},
            {"line_id": "b001", "char_id": "announcer", "speaker_role": "character",
             "text": "Gather round, listeners."},
        ],
        "meta": {},
    }


class TestReviewerRoleMismatchGuard:
    def test_rejects_announcer_on_cast_charid(self):
        led = _candidate_ledger()
        pre = _OTRLR.PreAuditReport(violations=[
            _OTRLR.CastViolation(line_id="b011", kind="role_mismatch",
                                 found="announcer", expected="announcer"),
        ])
        repaired = _OTRLR.apply_deterministic_cast_repairs(led, pre, list(CAST))
        rows = {r["line_id"]: r for r in led["lines"]}
        assert rows["b011"]["speaker_role"] == "character"
        assert repaired >= 1
        flags = rows["b011"].get("compose_flags") or []
        assert any(f.startswith("role_coerce:") for f in flags)

    def test_legit_announcer_repair_still_applies(self):
        # a real announcer row (char_id=announcer) flagged role_mismatch
        # expected=announcer must STILL be set to announcer (not coerced).
        led = _candidate_ledger()
        pre = _OTRLR.PreAuditReport(violations=[
            _OTRLR.CastViolation(line_id="b001", kind="role_mismatch",
                                 found="character", expected="announcer"),
        ])
        _OTRLR.apply_deterministic_cast_repairs(led, pre, list(CAST))
        rows = {r["line_id"]: r for r in led["lines"]}
        assert rows["b001"]["speaker_role"] == "announcer"


# ---------------------------------------------------------------------------
# Site (c): MANDATORY pre-freeze sweep in run_freeze_cascade (VERIFY item 1)
# ---------------------------------------------------------------------------

def _ledger_obj(data):
    obj = SimpleNamespace(data=data, episode_id=data.get("episode_id", "ep"))
    obj.save = lambda: None  # type: ignore[attr-defined]
    return obj


def _cascade_ledger_data():
    from nodes import _otr_ledger_freeze as _LFC
    return {
        "schema_version": _LFC.EXPECTED_SCHEMA_VERSION,
        "episode_id": "ep_test_d3",
        "cast": [
            {"char_id": "c01", "name": "MALI", "traits": "calm",
             "voice_preset": "v2/en_speaker_6"},
            {"char_id": "c02", "name": "MANFRED", "traits": "smug",
             "voice_preset": "v2/en_speaker_2"},
            {"char_id": "announcer", "name": "ANNOUNCER", "traits": "warm",
             "voice_preset": "v2/en_speaker_9"},
        ],
        "scenes": [], "shots": [],
        "beats": [{"beat_id": f"b00{i}"} for i in range(1, 5)],
        "lines": [
            {"line_id": "b001", "beat_id": "b001", "speaker_role": "announcer",
             "char_id": "announcer", "text": "Gather round, dear listeners.",
             "word_count": 4, "char_count": 28, "skip": False},
            {"line_id": "b002", "beat_id": "b002", "speaker_role": "character",
             "char_id": "c01", "text": "I have isolated the frequency at last.",
             "word_count": 7, "char_count": 37, "skip": False},
            # b011 culprit: cast char_id c02 but role announcer
            {"line_id": "b011", "beat_id": "b003", "speaker_role": "announcer",
             "char_id": "c02", "text": "I sent a copy to The Chronicle.",
             "word_count": 7, "char_count": 31, "skip": False},
            {"line_id": "b004", "beat_id": "b004", "speaker_role": "announcer",
             "char_id": "announcer", "text": "Stay tuned.",
             "word_count": 2, "char_count": 11, "skip": False},
        ],
        "music": [], "clips": [],
        "meta": {"episode_title": "Chandra's Echo", "style": "observatory_drama"},
    }


class TestPreFreezeSweep:
    def test_sweep_coerces_b011_keeps_announcer(self):
        led = _ledger_obj(_cascade_ledger_data())

        def fake_review(_fn, _led):
            return _OTRLR.ReviewerDisposition(
                verdict="clean_no_edits", pre_audit_violations=0,
                pre_audit_repairs_applied=0, doctor_edits_proposed=0,
                doctor_edits_applied=0, post_audit_violations=0,
            )

        with patch.object(_LFC_ORCH._OTRLR, "review_ledger",
                          side_effect=fake_review):
            _LFC_ORCH.run_freeze_cascade(lambda *a, **k: "", led)

        rows = {r["line_id"]: r for r in led.data["lines"]}
        # b011 coerced to character; legit announcer rows untouched
        assert rows["b011"]["speaker_role"] == "character"
        assert rows["b001"]["speaker_role"] == "announcer"
        assert rows["b004"]["speaker_role"] == "announcer"
        # audit stamped on meta + per-line breadcrumb
        rc = led.data["meta"].get("role_coercions")
        assert rc and rc["count"] == 1 and "b011" in rc["line_ids"]
        flags = rows["b011"].get("compose_flags") or []
        assert any(f.startswith("role_coerce:") for f in flags)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
