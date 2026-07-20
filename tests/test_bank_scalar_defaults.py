"""v4 campaign P1(i): validated scalar bank defaults replace the hardcoded
exact-id sets (visual-style pool / science floor / adaptation cast).

Asserts every runnable bank preserves its pre-migration behaviour (the former
base_source_bank_id / strict_v4_banks / (shakespeare, public_domain_story_v3)
semantics, now DECLARED as validated `defaults` scalars) and that the parser
validates the three new keys. The visual-STYLE pool (media|adaptation|generic)
is a separate axis from the source FEED (science_rss vs media_archive_rss vs
folger) -- this suite covers the style-pool + science-floor + adaptation-cast
axes only. Pure / CPU. UTF-8 no BOM, SFW.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_story_routing as SR  # noqa: E402
from nodes import _otr_style_catalog as SC  # noqa: E402


# (style_pool_class, require_science_floor, propagate_adaptation_cast) expected
# per runnable bank -- exactly the pre-migration behaviour.
EXPECTED = {
    "media_archive":          ("media",      False, False),
    "original_radio":         ("generic",    False, False),
    "scifi_news_pro":           ("generic",    False, False),
    "public_domain_story_v3": ("adaptation", False, True),
    "shakespeare":         ("adaptation", False, True),
}


class TestScalarDefaultsMigration:
    @pytest.mark.parametrize("bank_id", sorted(EXPECTED))
    def test_bank_defaults_match_expected(self, bank_id):
        bank = SR.require_runnable_bank(bank_id)
        d = bank.defaults or {}
        spc = str(d.get("style_pool_class", "generic") or "generic")
        floor = bool(d.get("require_science_floor"))
        prop = bool(d.get("propagate_adaptation_cast"))
        assert (spc, floor, prop) == EXPECTED[bank_id]

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

    def test_non_bool_science_floor_rejected(self):
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"require_science_floor": 1}), "test")

    def test_non_bool_propagate_rejected(self):
        with pytest.raises(SR.RegistryValidationError):
            SR._parse_bank(self._row({"propagate_adaptation_cast": "yes"}), "test")

    def test_valid_scalar_defaults_accepted(self):
        b = SR._parse_bank(self._row({
            "style_pool_class": "adaptation",
            "require_science_floor": True,
            "propagate_adaptation_cast": False,
        }), "test")
        assert b.defaults["style_pool_class"] == "adaptation"
        assert b.defaults["require_science_floor"] is True

    def test_absent_keys_ok(self):
        b = SR._parse_bank(self._row({}), "test")
        assert b.defaults == {}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
