"""v4 campaign P1(v): outro cast-completeness guard.

Deterministic scan over the REAL final rows (character char_ids with a non-skipped
spoken line) + a lenient name-presence check on the last announcer line, an
authored keep-if-complete repair, and the opt-in G12 terminal. Inert for every
current bank. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_outro_guard as OG  # noqa: E402
from nodes import _otr_ledger_freeze as LF  # noqa: E402
from nodes import _otr_story_routing as SR  # noqa: E402


_CAST = [
    {"char_id": "c01", "name": "MALI"},
    {"char_id": "c02", "name": "MANFRED CHEN"},
    {"char_id": "c03", "name": "DR. NORA ECKELS"},
    {"char_id": "announcer", "name": "ANNOUNCER"},
]


def _led(outro_text, *, include_c03=True, c03_skip=False, flag=False):
    lines = [
        {"line_id": "b0", "speaker_role": "announcer", "char_id": "announcer",
         "text": "Gather round."},
        {"line_id": "b1", "speaker_role": "character", "char_id": "c01",
         "text": "I found it."},
        {"line_id": "b2", "speaker_role": "character", "char_id": "c02",
         "text": "Show me."},
    ]
    if include_c03:
        lines.append({"line_id": "b3", "speaker_role": "character",
                      "char_id": "c03", "text": "Careful.", "skip": c03_skip})
    lines.append({"line_id": "b9", "speaker_role": "announcer",
                  "char_id": "announcer", "text": outro_text})
    meta = {"source_bank": "x"}
    if flag:
        meta["outro_cast_complete"] = True
    return {"schema_version": "l3-2026-05-14", "cast": list(_CAST),
            "meta": meta, "lines": lines}


# ---------------------------------------------------------------------------
# 1. Pure scan
# ---------------------------------------------------------------------------

class TestFinalCastScan:
    def test_final_cast_excludes_announcer_and_order(self):
        led = _led("bye")
        assert OG.final_cast_names(led) == ["MALI", "MANFRED CHEN", "DR. NORA ECKELS"]

    def test_skipped_row_excluded(self):
        led = _led("bye", c03_skip=True)
        assert OG.final_cast_names(led) == ["MALI", "MANFRED CHEN"]

    def test_find_outro_is_last_announcer(self):
        led = _led("That was our show.")
        assert OG.find_outro_text(led) == "That was our show."

    def test_missing_none_when_all_named_full_or_token(self):
        # 'Manfred' credits 'MANFRED CHEN' (token); 'Eckels' credits 'DR. NORA
        # ECKELS' (token, title 'DR' ignored); 'Mali' credits 'MALI' (full).
        led = _led("Tonight: Mali, Manfred, and Eckels. Good night.")
        assert OG.missing_final_cast_names(led) == []

    def test_missing_reports_absent_names(self):
        led = _led("Good night, friends.")
        assert OG.missing_final_cast_names(led) == [
            "MALI", "MANFRED CHEN", "DR. NORA ECKELS"]

    def test_boundary_no_false_present(self):
        # 'normali' must NOT credit 'MALI'.
        led = _led("The situation was abnormally calm. Good night.")
        assert "MALI" in OG.missing_final_cast_names(led)

    def test_no_outro_line_skips(self):
        led = _led("")  # empty outro text -> no sign-off to check
        assert OG.missing_final_cast_names(led) == []

    def test_no_cast_safe(self):
        assert OG.final_cast_names({}) == []
        assert OG.missing_final_cast_names({}) == []


# ---------------------------------------------------------------------------
# 2. Authored repair
# ---------------------------------------------------------------------------

def _fn_returns(value):
    def _f(messages, *, temperature=None, max_new_tokens=None, stop=None):
        return value
    return _f


def _fn_raises():
    def _f(messages, *, temperature=None, max_new_tokens=None, stop=None):
        raise RuntimeError("slot boom")
    return _f


class TestAuthoredRepair:
    def test_rewrites_when_complete(self):
        led = _led("Good night.", flag=True)
        n = OG.apply_outro_completeness_repair(
            led, _fn_returns("Tonight's cast: Mali, Manfred Chen, and Nora Eckels."))
        assert n == 1
        row = led["lines"][-1]
        assert "Mali" in row["text"] and "Eckels" in row["text"]
        assert "outro_cast_repair" in row.get("compose_flags", [])
        assert OG.missing_final_cast_names(led) == []

    def test_keeps_original_when_still_incomplete(self):
        led = _led("Good night.", flag=True)
        original = led["lines"][-1]["text"]
        n = OG.apply_outro_completeness_repair(led, _fn_returns("Still nothing."))
        assert n == 0
        assert led["lines"][-1]["text"] == original
        assert "compose_flags" not in led["lines"][-1]

    def test_never_raises(self):
        led = _led("Good night.", flag=True)
        assert OG.apply_outro_completeness_repair(led, _fn_raises()) == 0
        assert led["lines"][-1]["text"] == "Good night."

    def test_noop_when_already_complete_or_disabled(self):
        led = _led("Mali, Manfred, and Eckels take a bow.", flag=True)
        assert OG.apply_outro_completeness_repair(led, _fn_returns("x")) == 0
        led2 = _led("Good night.", flag=True)
        assert OG.apply_outro_completeness_repair(led2, None) == 0


# ---------------------------------------------------------------------------
# 3. Parser validation of the opt-in flag
# ---------------------------------------------------------------------------

class TestParserValidation:
    @staticmethod
    def _row(defaults):
        return {
            "source_bank_id": "t", "label": "t", "source_kind": "custom",
            "interpreter": "", "fetcher": "",
            "default_story_model": "m", "default_story_pipeline": "p",
            "defaults": defaults, "required_seams": [], "runnable": False,
            "guide_ref": "",
        }

    def test_non_bool_rejected(self):
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"require_outro_cast_complete": "yes"}), "t")

    def test_bool_accepted(self):
        b = SR._parse_bank(self._row({"require_outro_cast_complete": True}), "t")
        assert b.defaults["require_outro_cast_complete"] is True


_CURRENT_BANKS = [
    "media_archive", "original_radio", "scifi_fable2", "scifi_codex",
    "public_domain_story_v3", "shakespeare_v3",
]


class TestCurrentBanksInert:
    @pytest.mark.parametrize("bank_id", _CURRENT_BANKS)
    def test_no_current_bank_opts_in(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        assert not (bank.defaults or {}).get("require_outro_cast_complete")


# ---------------------------------------------------------------------------
# 4. Deterministic G12 terminal
# ---------------------------------------------------------------------------

def _g12_errors(ledger_data):
    errors: list = []
    LF._check_g12_outro_cast_complete(ledger_data, errors, [])
    return errors


class TestG12Terminal:
    def test_fires_when_opted_in_and_missing(self):
        led = _led("Good night.", flag=True)
        errs = _g12_errors(led)
        assert errs and "G12" in errs[0]

    def test_inert_without_opt_in(self):
        led = _led("Good night.", flag=False)  # missing but not opted in
        assert _g12_errors(led) == []

    def test_ok_when_complete(self):
        led = _led("Mali, Manfred, and Eckels. Good night.", flag=True)
        assert _g12_errors(led) == []

    def test_no_outro_skips(self):
        led = _led("", flag=True)
        assert _g12_errors(led) == []

    def test_run_gap_audit_includes_g12(self):
        led = _led("Good night.", flag=True)
        report = LF.run_gap_audit(led, label="pre")
        assert any("G12" in e for e in report.errors)

    def test_phase_10_raises_with_g12(self):
        led = _led("Good night.", flag=True)
        with pytest.raises(LF.FreezeAssertionError) as ei:
            LF.phase_10_gap_audit_post_and_freeze(led)
        assert any("G12" in e for e in ei.value.errors)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
