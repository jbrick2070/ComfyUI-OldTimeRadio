"""v4 campaign P1(vii): literal placeholder-token guard.

Whole-value, token-boundary, punctuation/quote/case tolerant, over named fields
only -- so real words containing the letters never trip and a real sentence with
'X' in it is not flagged. Deterministic G13 terminal, opt-in, inert for every
current bank. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_placeholder_guard as PG  # noqa: E402
from nodes import _otr_ledger_freeze as LF  # noqa: E402
from nodes import _otr_story_routing as SR  # noqa: E402


class TestIsPlaceholderValue:
    def test_bare_tokens(self):
        for v in ["X", "x", "Y", "Z", "TBD", "tbd", "TODO", "N/A", "PLACEHOLDER"]:
            assert PG.is_placeholder_value(v), v

    def test_punctuation_and_quotes_stripped(self):
        for v in ["'X'", "X.", "(Y)", '"TBD"', "  x  ", "-X-", "X!"]:
            assert PG.is_placeholder_value(v), v

    def test_real_values_pass(self):
        for v in ["box", "Xavier", "Yara", "MALI", "xylophone", "Max",
                  "X marks the spot", "the answer is x squared", "", None]:
            assert not PG.is_placeholder_value(v), v


class TestFindPlaceholderFields:
    def _led(self):
        return {
            "meta": {"placeholder_guard": True},
            "cast": [
                {"char_id": "c01", "name": "X"},
                {"char_id": "c02", "name": "MALI"},
            ],
            "lines": [
                {"line_id": "b1", "speaker_role": "character", "char_id": "c01",
                 "speaker": "X", "text": "Hello there."},
                {"line_id": "b2", "speaker_role": "character", "char_id": "c02",
                 "speaker": "MALI", "text": "Y"},
                {"line_id": "m1", "speaker_role": "music_inter", "text": "X"},
                {"line_id": "b3", "speaker_role": "character", "char_id": "c02",
                 "speaker": "MALI", "text": "X marks the spot."},
            ],
        }

    def test_hits_named_fields_only(self):
        hits = PG.find_placeholder_fields(self._led())
        joined = " | ".join(hits)
        assert "cast[0].name='X'" in joined
        assert "b1].speaker='X'" in joined
        assert "b2].text='Y'" in joined
        # music row + real sentence + real name NOT flagged
        assert "m1" not in joined
        assert "b3" not in joined
        assert "MALI" not in joined

    def test_clean_ledger(self):
        led = {"cast": [{"char_id": "c01", "name": "MALI"}],
               "lines": [{"line_id": "b1", "speaker_role": "character",
                          "speaker": "MALI", "text": "A real line."}]}
        assert PG.find_placeholder_fields(led) == []

    def test_non_dict_safe(self):
        assert PG.find_placeholder_fields(None) == []


# ---------------------------------------------------------------------------
# Parser validation + current banks inert
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
            SR._parse_bank(self._row({"placeholder_guard": 1}), "t")

    def test_bool_accepted(self):
        b = SR._parse_bank(self._row({"placeholder_guard": True}), "t")
        assert b.defaults["placeholder_guard"] is True


_CURRENT_BANKS = [
    "media_archive", "original_radio", "scifi_news_pro",
    "public_domain_story_v3", "shakespeare_v3",
]


class TestCurrentBanksInert:
    @pytest.mark.parametrize("bank_id", _CURRENT_BANKS)
    def test_no_current_bank_opts_in(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        assert not (bank.defaults or {}).get("placeholder_guard")


# ---------------------------------------------------------------------------
# G13 terminal
# ---------------------------------------------------------------------------

def _g13_errors(ledger_data):
    errors: list = []
    LF._check_g13_placeholder_tokens(ledger_data, errors, [])
    return errors


class TestG13Terminal:
    def _led(self, flag):
        meta = {"source_bank": "x"}
        if flag:
            meta["placeholder_guard"] = True
        return {"schema_version": "l3-2026-05-14", "meta": meta,
                "cast": [{"char_id": "c01", "name": "X"}],
                "lines": [{"line_id": "b1", "speaker_role": "character",
                           "char_id": "c01", "speaker": "X", "text": "hi"}]}

    def test_fires_when_opted_in(self):
        errs = _g13_errors(self._led(True))
        assert errs and "G13" in errs[0]

    def test_inert_without_opt_in(self):
        assert _g13_errors(self._led(False)) == []

    def test_clean_ok(self):
        led = {"meta": {"placeholder_guard": True},
               "cast": [{"char_id": "c01", "name": "MALI"}],
               "lines": [{"line_id": "b1", "speaker_role": "character",
                          "speaker": "MALI", "text": "A real line."}]}
        assert _g13_errors(led) == []

    def test_run_gap_audit_includes_g13(self):
        report = LF.run_gap_audit(self._led(True), label="pre")
        assert any("G13" in e for e in report.errors)

    def test_phase_10_raises_with_g13(self):
        with pytest.raises(LF.FreezeAssertionError) as ei:
            LF.phase_10_gap_audit_post_and_freeze(self._led(True))
        assert any("G13" in e for e in ei.value.errors)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
