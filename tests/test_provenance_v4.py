"""v4 campaign P1(viii): source-provenance normalizer + the G14 publish report.

Maps every lane's source_rights shape onto one normalized record and renders the
spoken-coda + printed-credit lines per status. The operator rule that a
research_only source BLOCKS publish is still deterministic and still opt-in per
bank (ACTIVE on public_domain and shakespeare since 2026-08-04) -- but since
2026-08-15 it is ENFORCED at the publication boundary rather than by refusing to
freeze, so G14 here REPORTS and `tests/test_publication_eligibility.py` owns the
end-to-end proof. Pure / CPU. UTF-8 no BOM, SFW.
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
        assert PROV.spoken_coda_line({"status": "unknown"}) == ""

    def test_a_licensed_source_is_never_named_aloud(self):
        """Credit only -- the licensor does not get thanked on air.

        Operator ruling 2026-08-05: "I get it, thanks to Folger, but they didn't
        write it -- Shakespeare did." Folger publishes the edition; the play is
        Shakespeare's, so naming the licensor in the audio credits the wrong
        party to anyone listening.

        This assertion is INVERTED on purpose. It previously required a licensed
        source to speak "Tonight's tale was adapted from <source_label>". If it
        fails, someone put the licensor back on the air -- that is an operator
        decision, not a fix.
        """
        for label in ("Hamlet", "Folger Shakespeare", ""):
            for status in ("licensed_noncommercial", "licensed_commercial"):
                assert PROV.spoken_coda_line({
                    "status": status,
                    "source_label": label,
                    "license_label": "CC BY-NC 3.0",
                }) == "", "a licensed source was named aloud: %r" % label

    def test_the_credit_still_carries_what_the_audio_dropped(self):
        """Dropping it from the audio must not drop it from the record."""
        prov = {
            "status": "licensed_noncommercial",
            "source_label": "Folger Shakespeare",
            "license_label": "CC BY-NC 3.0",
            "commercial_use_allowed": False,
        }
        printed = PROV.printed_credit_line(prov)
        assert "Folger Shakespeare" in printed
        assert "CC BY-NC 3.0" in printed
        assert "NON-COMMERCIAL SOURCE" in PROV.noncommercial_notice(prov)

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
    "media_archive", "original", "scifi_news_pro",
    "public_domain", "shakespeare",
]


#: The FIDELITY banks adapt someone else's text, so their rights data has to
#: reach a human. They opted in 2026-08-04. Every other bank writes its own
#: material and stays inert.
_PROVENANCE_OPTED_IN = frozenset({"shakespeare", "public_domain"})


class TestOnlyTheFidelityBanksOptIn:
    """These tests pin WHICH banks opt in, so turning the flag on -- or losing
    it -- is a deliberate, visible act.

    Renamed from TestCurrentBanksInert 2026-08-05: the class asserted two ACTIVE
    banks while its own name said every bank was inert, and that contradiction
    was read as fact when specifying the spoken-citation fix. The normalizer was
    built opt-in and left switched off everywhere, which is why the announcer
    read a raw licence string aloud instead of the clean coda this module
    composes; public_domain and shakespeare opted in on 2026-08-04."""

    @pytest.mark.parametrize("bank_id", _CURRENT_BANKS)
    def test_only_the_fidelity_banks_opt_in(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        opted = bool((bank.defaults or {}).get("provenance_normalize"))
        assert opted is (bank_id in _PROVENANCE_OPTED_IN), (
            f"{bank_id}: provenance_normalize={opted}; expected "
            f"{bank_id in _PROVENANCE_OPTED_IN}"
        )

    def test_the_noncommercial_source_warns_and_the_free_one_does_not(self):
        """Folger is CC BY-NC. Nothing used to tell a publisher that."""
        folger = PROV.normalize_provenance({
            "source_label": "Folger Shakespeare",
            "license_label": "CC BY-NC 3.0 (Folger Shakespeare Library)",
            "commercial_use_allowed": False,
        })
        notice = PROV.noncommercial_notice(folger)
        assert "NON-COMMERCIAL SOURCE" in notice
        assert "Do not sell it" in notice
        # The licence identifier belongs in PRINT, never in the audio.
        assert "CC BY-NC" in PROV.printed_credit_line(folger)
        assert "CC BY-NC" not in PROV.spoken_coda_line(folger)

        gutenberg = PROV.normalize_provenance({
            "source_label": "Project Gutenberg",
            "license_status": "public_domain_us",
        })
        assert PROV.noncommercial_notice(gutenberg) == ""
        # A transcriber is not named aloud when the law does not require it.
        assert "Gutenberg" not in PROV.spoken_coda_line(gutenberg)


# ---------------------------------------------------------------------------
# G14 publish gate
# ---------------------------------------------------------------------------

def _led(prov):
    meta = {"source_bank": "x"}
    if prov is not None:
        meta["provenance"] = prov
    return {"schema_version": "l3-2026-05-14", "meta": meta,
            "lines": [{"line_id": "b1", "speaker_role": "character", "text": "hi"}]}


def _freezable_led(prov):
    """`_led` plus the structural keys Phase 10 requires.

    The two freeze tests below drive the REAL cascade phase, so the fixture has
    to be able to pass it on everything except the thing under test -- otherwise
    "it froze" and "it did not freeze" would both be answers about missing
    top-level lists rather than about provenance.
    """
    led = _led(prov)
    # `_led` pins the historical l3 schema on purpose for the coda tests; the
    # freeze asserts the CURRENT one, so read it from the module under test
    # rather than restating a version string that moves.
    led["schema_version"] = LF.EXPECTED_SCHEMA_VERSION
    led["meta"].update({"episode_title": "T", "style": "s"})
    led["cast"] = [{"char_id": "c01", "name": "NARRATOR"}]
    led["lines"] = [{"line_id": "b1", "char_id": "c01",
                     "speaker_role": "character", "text": "hi"}]
    for key in ("beats", "scenes", "shots", "music", "clips"):
        led[key] = []
    return led


def _g14_findings(ledger_data):
    """(errors, warnings) from the G14 check alone."""
    errors: list = []
    warnings: list = []
    LF._check_g14_provenance_publish(ledger_data, errors, warnings)
    return errors, warnings


class TestG14PublishGate:
    """G14 REPORTS; the publication boundary ENFORCES (2026-08-15, D5a).

    These assertions moved deliberately. G14 used to append to `errors`, which
    at Phase 10 means FreezeAssertionError -- so "a research_only source blocks
    publish" was carried out by destroying a finished render, leaving the
    operator without even the archival copy such a source IS cleared for. The
    rule is unchanged and still deterministic; it now lands at
    `OTR_MasterAudioMux`, which withholds the OBS copy on the durable
    publication-eligibility receipt. `tests/test_publication_eligibility.py`
    owns the end-to-end proof.
    """

    def test_research_only_is_reported_as_a_warning(self):
        led = _led({"status": "research_only", "blocks_publish": True,
                    "source_label": "Restricted"})
        errors, warnings = _g14_findings(led)
        assert errors == []
        assert warnings and "G14" in warnings[0]

    def test_public_domain_passes(self):
        led = _led({"status": "public_domain_us", "blocks_publish": False})
        assert _g14_findings(led) == ([], [])

    def test_inert_without_provenance(self):
        assert _g14_findings(_led(None)) == ([], [])

    def test_run_gap_audit_includes_g14(self):
        led = _led({"status": "research_only", "blocks_publish": True})
        report = LF.run_gap_audit(led, label="pre")
        assert any("G14" in w for w in report.warnings)
        assert not any("G14" in e for e in report.errors)

    def test_phase_10_freezes_instead_of_killing_the_render(self):
        led = _freezable_led({"status": "research_only", "blocks_publish": True})
        LF.phase_10_gap_audit_post_and_freeze(led)
        assert led["meta"]["freeze_verdict"] == "frozen_with_warns"

    def test_phase_10_stamps_the_block_where_publication_reads_it(self):
        led = _freezable_led({"status": "research_only", "blocks_publish": True})
        LF.phase_10_gap_audit_post_and_freeze(led)
        assert led["meta"]["publication_eligibility"]["eligible"] is False


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
