"""v4 campaign P1(viii): source-provenance normalizer + G14 publish gate.

Maps every lane's source_rights shape onto one normalized record, renders the
spoken-coda + printed-credit lines per status, and enforces the operator rule
that a research_only source BLOCKS publish (deterministic G14, opt-in, inert for
every current bank). Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_provenance as PROV  # noqa: E402
from nodes import _otr_ledger_freeze as LF  # noqa: E402
from nodes import _otr_story_routing as SR  # noqa: E402


class TestNormalize:
    def test_public_domain_us(self):
        p = PROV.normalize_provenance({
            "license_status": "public_domain_us",
            "source_label": "A Christmas Carol", "source_url": "http://x"})
        assert p["status"] == "public_domain_us"
        assert p["commercial_use_allowed"] is True
        assert p["blocks_publish"] is False
        assert p["source_label"] == "A Christmas Carol"

    def test_cc0(self):
        p = PROV.normalize_provenance({"license_status": "cc0"})
        assert p["status"] == "cc0" and p["blocks_publish"] is False

    def test_research_only_blocks(self):
        p = PROV.normalize_provenance({
            "license_status": "research_only", "source_label": "Restricted"})
        assert p["status"] == "research_only"
        assert p["commercial_use_allowed"] is False
        assert p["blocks_publish"] is True

    def test_shakespeare_noncommercial(self):
        p = PROV.normalize_provenance({
            "license_label": "Folger CC BY-NC", "commercial_use_allowed": False,
            "source_label": "Hamlet 1.1"})
        assert p["status"] == "licensed_noncommercial"
        assert p["commercial_use_allowed"] is False
        assert p["blocks_publish"] is False

    def test_shakespeare_commercial(self):
        p = PROV.normalize_provenance({
            "license_label": "CC0", "commercial_use_allowed": True})
        assert p["status"] == "licensed_commercial"

    def test_synthetic(self):
        p = PROV.normalize_provenance({"license_label": "synthetic original"})
        assert p["status"] == "synthetic" and p["blocks_publish"] is False

    def test_unknown_and_nondict_safe(self):
        assert PROV.normalize_provenance({})["status"] == "unknown"
        assert PROV.normalize_provenance(None)["status"] == "unknown"
        assert PROV.normalize_provenance({})["blocks_publish"] is False


class TestTemplates:
    def test_spoken_coda_per_status(self):
        for st in ("public_domain_us", "cc0", "research_only", "synthetic"):
            assert PROV.spoken_coda_line({"status": st})
        assert PROV.spoken_coda_line(
            {"status": "licensed_noncommercial", "source_label": "Hamlet",
             "license_label": "CC BY-NC"})
        assert PROV.spoken_coda_line({"status": "unknown"}) == ""

    def test_printed_credit_per_status(self):
        assert "public domain" in PROV.printed_credit_line(
            {"status": "public_domain_us", "source_label": "Carol"})
        assert "research use only" in PROV.printed_credit_line(
            {"status": "research_only", "source_label": "R"})
        assert "used under" in PROV.printed_credit_line(
            {"status": "licensed_noncommercial", "source_label": "H",
             "license_label": "CC BY-NC"})
        assert PROV.printed_credit_line({"status": "synthetic"})


# ---------------------------------------------------------------------------
# Parser + current banks inert
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
            SR._parse_bank(self._row({"provenance_normalize": "yes"}), "t")

    def test_bool_accepted(self):
        b = SR._parse_bank(self._row({"provenance_normalize": True}), "t")
        assert b.defaults["provenance_normalize"] is True


_CURRENT_BANKS = [
    "media_archive", "original_radio", "scifi_fable2", "scifi_codex",
    "public_domain_story_v3", "shakespeare_v3",
]


class TestCurrentBanksInert:
    @pytest.mark.parametrize("bank_id", _CURRENT_BANKS)
    def test_no_current_bank_opts_in(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        assert not (bank.defaults or {}).get("provenance_normalize")


# ---------------------------------------------------------------------------
# G14 publish gate
# ---------------------------------------------------------------------------

def _led(prov):
    meta = {"source_bank": "x"}
    if prov is not None:
        meta["provenance"] = prov
    return {"schema_version": "l3-2026-05-14", "meta": meta,
            "lines": [{"line_id": "b1", "speaker_role": "character", "text": "hi"}]}


def _g14_errors(ledger_data):
    errors: list = []
    LF._check_g14_provenance_publish(ledger_data, errors, [])
    return errors


class TestG14PublishGate:
    def test_blocks_research_only(self):
        led = _led({"status": "research_only", "blocks_publish": True,
                    "source_label": "Restricted"})
        errs = _g14_errors(led)
        assert errs and "G14" in errs[0]

    def test_public_domain_passes(self):
        led = _led({"status": "public_domain_us", "blocks_publish": False})
        assert _g14_errors(led) == []

    def test_inert_without_provenance(self):
        assert _g14_errors(_led(None)) == []

    def test_run_gap_audit_includes_g14(self):
        led = _led({"status": "research_only", "blocks_publish": True})
        report = LF.run_gap_audit(led, label="pre")
        assert any("G14" in e for e in report.errors)

    def test_phase_10_raises(self):
        led = _led({"status": "research_only", "blocks_publish": True})
        with pytest.raises(LF.FreezeAssertionError) as ei:
            LF.phase_10_gap_audit_post_and_freeze(led)
        assert any("G14" in e for e in ei.value.errors)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
