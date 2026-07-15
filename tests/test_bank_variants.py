"""Bake-off bank-variant helper + base-family wiring (Chunk 1).

Proves ``base_source_bank_id`` and the five family-behaviour call sites it feeds
(story_rules resolve + runnable coverage, style pool, strict-v4 science, the
adaptation cast-name membership) BEFORE any variant rows exist: a hypothetical
``<base>_v2``/``_v3`` id must resolve to its base family, while every real base
id stays inert (no regression). Pure Python; no GPU, no LLM, no network.

r3 ruling: base-mapping is FAMILY behaviour only. Provenance/owner_bank is NOT a
caller of this helper (content-authorship must prove the exact selected lane),
so no provenance assertion lives here.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_bank_variants as BV  # noqa: E402
from nodes import _otr_story_routing as ROUTING  # noqa: E402
from nodes import _otr_story_rules as RULES  # noqa: E402
from nodes import _otr_style_catalog as STYLE  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_registries():
    ROUTING._REGISTRY = None
    RULES._clear_caches()
    yield
    ROUTING._REGISTRY = None
    RULES._clear_caches()


# ---------------------------------------------------------------------------
# 1. Helper unit behaviour
# ---------------------------------------------------------------------------

class TestHelper:
    @pytest.mark.parametrize("variant,base", [
        ("shakespeare_v2", "shakespeare"),
        ("shakespeare_v3", "shakespeare"),
        ("science_news_v2", "science_news"),
        ("science_news_v3", "science_news"),
        ("media_archive_v3", "media_archive"),
        ("scifi_codex_v2", "scifi_codex"),
        ("scifi_sonnet_v3", "scifi_sonnet"),
        ("scifi_fable2_v2", "scifi_fable2"),
    ])
    def test_variant_strips_to_base(self, variant, base):
        assert BV.base_source_bank_id(variant) == base
        assert BV.is_bakeoff_variant(variant) is True

    @pytest.mark.parametrize("base", [
        "science_news", "media_archive", "public_domain_story", "shakespeare",
        "original_radio", "scifi_fable2", "scifi_codex", "scifi_sonnet",
        "custom_source_bank",
    ])
    def test_base_ids_are_inert(self, base):
        # scifi_fable2 ends in "2" but NOT "_v2" -> must pass through unchanged.
        assert BV.base_source_bank_id(base) == base
        assert BV.is_bakeoff_variant(base) is False

    def test_unknown_and_non_str_passthrough(self):
        assert BV.base_source_bank_id("totally_unknown") == "totally_unknown"
        assert BV.base_source_bank_id("") == ""
        assert BV.base_source_bank_id(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 2. story_rules base-map (resolve): a variant id resolves the base rules pack
# ---------------------------------------------------------------------------

class TestStoryRulesBaseMap:
    @pytest.mark.parametrize("base", [
        "science_news", "media_archive", "public_domain_story", "shakespeare",
        "original_radio", "scifi_fable2", "scifi_codex", "scifi_sonnet",
    ])
    def test_variant_resolves_base_rules(self, base):
        base_rules = RULES.resolve_story_rules(base)
        for suffix in ("_v2", "_v3"):
            variant_rules = RULES.resolve_story_rules(base + suffix)
            assert variant_rules is base_rules
            assert variant_rules.rules_id == base

    def test_unknown_bank_still_raises(self):
        with pytest.raises(RULES.UnknownStoryRulesError):
            RULES.resolve_story_rules("not_a_bank")


# ---------------------------------------------------------------------------
# 3. style pool base-map: a variant draws its base family's pool
# ---------------------------------------------------------------------------

class TestStylePoolBaseMap:
    @pytest.mark.parametrize("base", ["media_archive", "shakespeare",
                                      "public_domain_story"])
    def test_variant_style_matches_base(self, base):
        # Same seed + premise: the only difference is the source_bank suffix.
        premise, seed = "a quiet discovery in the archive", 4242
        base_slug = STYLE.select_style(premise, {"source_bank": base}, seed)
        for suffix in ("_v2", "_v3"):
            variant_slug = STYLE.select_style(
                premise, {"source_bank": base + suffix}, seed)
            assert variant_slug == base_slug


# ---------------------------------------------------------------------------
# 4. No regression: every existing base bank still resolves pack + rules
#    (the helper is inert for base ids everywhere)
# ---------------------------------------------------------------------------

class TestNoRegression:
    def test_all_registered_banks_resolve(self):
        ids = ROUTING.list_bank_ids()
        assert ids[-1] == "custom_source_bank"  # untouched at Chunk 1
        for bank_id in ids:
            bank = ROUTING.get_bank(bank_id)
            # pack resolution unaffected by the helper (variants get own packs)
            ROUTING.resolve_story_pack(bank_id)
            if bank.runnable:
                RULES.resolve_story_rules(bank_id)

    def test_registry_loads_clean(self):
        # _load_all() runnable-coverage gate runs here; the base-map keeps it
        # green for the current registry (no variant rows yet).
        assert RULES.list_rules_ids()  # non-empty => _load_all succeeded


# ---------------------------------------------------------------------------
# 5. Chunk 2: the 8 _v2 rows resolve (16 runnable / 17 visible)
# ---------------------------------------------------------------------------

_V2_BASE = {
    "science_news_v2": "science_news",
    "media_archive_v2": "media_archive",
    "public_domain_story_v2": "public_domain_story",
    "shakespeare_v2": "shakespeare",
    "original_radio_v2": "original_radio",
    "scifi_fable2_v2": "scifi_fable2",
    "scifi_codex_v2": "scifi_codex",
    "scifi_sonnet_v2": "scifi_sonnet",
}


class TestChunk2V2Rows:
    def test_counts_16_runnable_17_visible(self):
        ids = ROUTING.list_bank_ids()
        assert len(ids) == 17                      # 8 base + 8 _v2 + custom
        assert ids[-1] == "custom_source_bank"
        runnable = [b for b in ids if ROUTING.get_bank(b).runnable]
        assert len(runnable) == 16                 # only custom is non-runnable
        v2 = [b for b in ids if b.endswith("_v2")]
        assert len(v2) == 8

    @pytest.mark.parametrize("v2,base", sorted(_V2_BASE.items()))
    def test_v2_owns_its_pack_on_the_base_pipeline(self, v2, base):
        bank = ROUTING.get_bank(v2)
        base_bank = ROUTING.get_bank(base)
        assert bank.runnable is True
        # variant rows change ONLY default_story_model -> its own _v2 pack;
        # the pipeline mirrors the base family (B6).
        assert bank.default_story_pipeline == base_bank.default_story_pipeline
        pack = ROUTING.resolve_story_pack(v2)
        assert pack.source_bank_id == v2
        assert pack.story_model_id.endswith("_v2")
        assert pack.story_pipeline_id == base_bank.default_story_pipeline
        # family behaviour: a _v2 id resolves the BASE family's story rules.
        assert BV.base_source_bank_id(v2) == base
        assert RULES.resolve_story_rules(v2).rules_id == base
