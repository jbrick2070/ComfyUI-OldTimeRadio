"""v4 campaign P1(i): validated scalar bank defaults replace the hardcoded
exact-id sets (visual-style pool / adaptation cast).

Asserts every runnable bank preserves its pre-migration behaviour (the former
base_source_bank_id / strict_v4_banks / (shakespeare, public_domain)
semantics, now DECLARED as validated `defaults` scalars) and that the parser
validates the surviving keys. The visual-STYLE pool
(media|adaptation|generic) is a separate axis from the source feed; this suite
covers only the style-pool and adaptation-cast routing axes. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_routing as SR  # noqa: E402
from nodes import _otr_style_catalog as SC  # noqa: E402


# (style_pool_class, propagate_adaptation_cast) expected per runnable bank.
EXPECTED = {
    "media_archive": ("media", False),
    "original": ("generic", False),
    "scifi_news_pro": ("generic", False),
    "public_domain": ("adaptation", True),
    "shakespeare": ("adaptation", True),
}


class TestScalarDefaultsMigration:
    @pytest.mark.parametrize("bank_id", sorted(EXPECTED))
    def test_bank_defaults_match_expected(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        d = bank.defaults or {}
        spc = str(d.get("style_pool_class", "generic") or "generic")
        prop = bool(d.get("propagate_adaptation_cast"))
        assert (spc, prop) == EXPECTED[bank_id]

    @pytest.mark.parametrize("bank_id", sorted(EXPECTED))
    def test_select_style_pool_matches_class(self, bank_id):
        spc = EXPECTED[bank_id][0]
        # The writer stamps style_pool_class into meta; select_style reads it.
        meta = {"source_bank": bank_id, "style_pool_class": spc}
        slug = SC.select_style("a science premise", meta, cast_seed="seed-123")
        assert slug in set(SC.all_slugs())
        if spc == "media":
            assert slug in set(SC.media_archive_slugs())
        elif spc == "adaptation":
            assert slug in set(SC.adaptation_slugs())

    def test_style_pool_class_missing_is_generic(self):
        # meta without style_pool_class must behave as generic (never an
        # adaptation-only slug) -- the default both the writer and select_style
        # fall back to.
        slug = SC.select_style("p", {"source_bank": "x"}, cast_seed="s")
        assert slug in set(SC.all_slugs())

    def test_style_pool_deterministic(self):
        meta = {"style_pool_class": "media"}
        a = SC.select_style("p", dict(meta), cast_seed="fixed-seed")
        b = SC.select_style("p", dict(meta), cast_seed="fixed-seed")
        assert a == b  # sha256(cast_seed)-keyed, C7-safe


class TestStoryScaffoldDefault:
    """defaults.story_scaffold (2026-08-03): the bank decides whether the
    pre-outline premise scaffold runs.

    Operator definition of the original bank: "make a random radio drama" --
    it may INVENT a premise and tag it, but never CHOOSE from the catalog.
    The specimen defect: tempests_chart (original, spark-deck lighthouse tale)
    carried meta.style='asteroid_mining_labor_dispute' and a cast member
    dressed as a "Mining Union Representative... hard hat with a union pin",
    because lock_cast was handed the catalog label as style."""

    def test_original_declares_scaffold_off(self):
        bank = SR.require_runnable_bank("original")
        assert (bank.defaults or {}).get("story_scaffold") == "off"

    def test_every_other_runnable_bank_keeps_the_scaffold(self):
        # Absent means "on" -- today's behaviour for every non-original bank.
        # Enumerated from the LIVE registry rather than the EXPECTED table,
        # which predates this key and omits scifi_news (QA 2026-08-03) -- a
        # hand-list drifts, the registry cannot.
        others = [b for b in SR.runnable_bank_ids() if b != "original"] \
            if hasattr(SR, "runnable_bank_ids") else None
        if others is None:
            others = [bid for bid in
                      ("media_archive", "scifi_news", "scifi_news_pro",
                       "public_domain", "shakespeare")]
        assert others, "registry returned no non-original runnable banks"
        for bank_id in others:
            bank = SR.require_runnable_bank(bank_id)
            assert (bank.defaults or {}).get("story_scaffold", "on") == "on", (
                f"{bank_id} unexpectedly declares story_scaffold off")

    def test_gate_semantics_absent_means_on(self):
        # The writer's gate: _style_grammar_on AND bank != "off". Absent/junk
        # values must not silently disable the scaffold for other banks.
        for defaults, expect_active in (({}, True),
                                        ({"story_scaffold": "on"}, True),
                                        ({"story_scaffold": "off"}, False)):
            bank_scaffold = str(defaults.get("story_scaffold", "on")).strip().lower()
            assert (bank_scaffold != "off") is expect_active


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

    def test_bad_style_pool_class_rejected(self):
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"style_pool_class": "scifi"}), "test")

    def test_non_bool_propagate_rejected(self):
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"propagate_adaptation_cast": "yes"}), "test")

    def test_valid_scalar_defaults_accepted(self):
        b = SR._parse_bank(self._row({
            "style_pool_class": "adaptation",
            "propagate_adaptation_cast": False,
        }), "test")
        assert b.defaults["style_pool_class"] == "adaptation"
        assert b.defaults["propagate_adaptation_cast"] is False

    def test_absent_keys_ok(self):
        b = SR._parse_bank(self._row({}), "test")
        assert b.defaults == {}

    def test_bad_story_scaffold_rejected(self):
        # Junk must fail LOUD at registry parse, not silently read as "on".
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"story_scaffold": "sometimes"}), "test")

    @pytest.mark.parametrize("value", ["on", "off"])
    def test_valid_story_scaffold_accepted(self, value):
        b = SR._parse_bank(self._row({"story_scaffold": value}), "test")
        assert b.defaults["story_scaffold"] == value


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
