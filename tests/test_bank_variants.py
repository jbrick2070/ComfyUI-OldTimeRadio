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
    def test_counts_24_runnable_25_visible(self):
        # chunk 4: 8 base + 8 _v2 + 8 _v3 + custom = 25 visible / 24 runnable.
        ids = ROUTING.list_bank_ids()
        assert len(ids) == 25
        assert ids[-1] == "custom_source_bank"
        runnable = [b for b in ids if ROUTING.get_bank(b).runnable]
        assert len(runnable) == 24                 # only custom is non-runnable
        assert len([b for b in ids if b.endswith("_v2")]) == 8
        assert len([b for b in ids if b.endswith("_v3")]) == 8

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


# ---------------------------------------------------------------------------
# 6. Chunk 4: the 8 _v3 lanes (own v3 pipeline; sci-fi own-runner, rest inline)
# ---------------------------------------------------------------------------

# v3 bank id -> (base, v3 pipeline id, lane kind)
_V3 = {
    "science_news_v3":        ("science_news",        "legacy_many_pass_v3",         "inline"),
    "media_archive_v3":       ("media_archive",       "legacy_many_pass_v3",         "inline"),
    "public_domain_story_v3": ("public_domain_story", "legacy_many_pass_v3",         "inline"),
    "shakespeare_v3":         ("shakespeare",         "legacy_many_pass_v3",         "inline"),
    "original_radio_v3":      ("original_radio",      "original_multi_pass_v3",      "inline"),
    "scifi_fable2_v3":        ("scifi_fable2",        "fable2_multipass_v3",         "runner"),
    "scifi_codex_v3":         ("scifi_codex",         "scifi_codex_circuit_v3",      "runner"),
    "scifi_sonnet_v3":        ("scifi_sonnet",        "sonnet_archive_multipass_v3", "runner"),
}


class TestChunk4V3Rows:
    def test_v3_count_and_order(self):
        ids = ROUTING.list_bank_ids()
        assert len([b for b in ids if b.endswith("_v3")]) == 8
        assert ids.index("science_news_v3") > ids.index("science_news_v2")
        assert ids[-1] == "custom_source_bank"

    @pytest.mark.parametrize("v3,base,pipe,kind",
                             sorted((k, *v) for k, v in _V3.items()))
    def test_v3_owns_pack_pipeline_and_lane(self, v3, base, pipe, kind):
        from nodes import OTR_LedgerScriptWriter as W
        bank = ROUTING.get_bank(v3)
        assert bank.runnable is True
        assert bank.default_story_pipeline == pipe
        pack = ROUTING.resolve_story_pack(v3)
        assert pack.source_bank_id == v3
        assert pack.story_model_id.endswith("_v3")
        assert pack.story_pipeline_id == pipe
        assert BV.base_source_bank_id(v3) == base
        assert RULES.resolve_story_rules(v3).rules_id == base
        if kind == "runner":
            assert pipe in W._RUNNER_BY_PIPELINE
            assert pipe not in W._LEGACY_INLINE_PIPELINES
            assert ROUTING.get_pipeline(pipe).executable is True
        else:
            assert pipe in W._LEGACY_INLINE_PIPELINES
            assert pipe in W._INLINE_V3_PIPELINES
            assert pipe not in W._RUNNER_BY_PIPELINE

    def test_v3_advisory_is_bounded_and_owns_one_field(self):
        from nodes import OTR_LedgerScriptWriter as W

        class _Led:
            def __init__(self):
                self.data = {"lines": [
                    {"line_id": "l1", "char_id": "c01", "speaker_role": "character",
                     "text": "The wording is exact.", "beat_id": "b1"},
                    {"line_id": "l2", "char_id": "c02", "speaker_role": "character",
                     "text": "But what could it mean.", "beat_id": "b2"},
                ], "meta": {}}

        led = _Led()
        before = [dict(r) for r in led.data["lines"]]
        W.run_v3_advisory(led, led.data["meta"], lane="scifi_sonnet_v3")
        rec = led.data["meta"]["scifi_sonnet_v3_advisory"]
        assert rec["status"] == "ok"
        assert rec["line_count"] == 2
        assert rec["focus"]["metric"] == "reader_alternation"
        # advisory-only: spoken rows untouched (no hole in the ledger).
        assert led.data["lines"] == before
        # exactly one owned meta field written.
        assert list(led.data["meta"].keys()) == ["scifi_sonnet_v3_advisory"]

    def test_v3_advisory_never_raises_on_bad_input(self):
        from nodes import OTR_LedgerScriptWriter as W
        meta = {}
        W.run_v3_advisory(None, meta, lane="science_news_v3")   # None led
        assert meta["science_news_v3_advisory"]["status"] in ("ok", "error")

        class _Bad:
            data = {"lines": "not a list", "meta": {}}
        bad = _Bad()
        W.run_v3_advisory(bad, bad.data["meta"], lane="media_archive_v3")
        assert "media_archive_v3_advisory" in bad.data["meta"]
