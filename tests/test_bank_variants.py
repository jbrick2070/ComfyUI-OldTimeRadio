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
        ("scifi_codex_v2", "scifi_codex"),
        ("scifi_fable2_v2", "scifi_fable2"),
    ])
    def test_variant_strips_to_base(self, variant, base):
        assert BV.base_source_bank_id(variant) == base
        assert BV.is_bakeoff_variant(variant) is True

    @pytest.mark.parametrize("base", [
        "science_news", "media_archive", "public_domain_story", "shakespeare",
        "original_radio", "scifi_fable2", "scifi_codex",
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
# 2. story_rules independence (resolve): each lane owns its rules by EXACT id
# ---------------------------------------------------------------------------
# Independence refactor 2026-07-17: the base_source_bank_id family-mapping is
# SEVERED for story_rules -- a _v3 lane resolves its OWN pack by exact id, NOT
# its base family's. The old "variant resolves base rules" pin is obsolete.

class TestStoryRulesIndependence:
    @pytest.mark.parametrize("bank_id", [
        "media_archive", "original_radio", "scifi_fable2",
        "public_domain_story_v3", "shakespeare_v3",
    ])
    def test_lane_resolves_its_own_rules(self, bank_id):
        assert RULES.resolve_story_rules(bank_id).rules_id == bank_id

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
# 5. Sonnet-bake-off rip 2026-07-18: four bake-off _v3 variants are RETIRED,
#    leaving two inline _v3 lanes -- roster is now 7 runnable / 8 visible
#    (scifi_codex_v4 from the 2026-07-17 v4 improvement campaign remains). The
#    _v2 family and the science_news family were already removed.
# ---------------------------------------------------------------------------

# v3 bank id -> (base, v3 pipeline id, lane kind). `base` is the pure
# base_source_bank_id strip (still a valid string op); rules now resolve by
# EXACT id, not by base family.
_V3 = {
    "public_domain_story_v3": ("public_domain_story", "legacy_many_pass_v3", "inline"),
    "shakespeare_v3":         ("shakespeare",         "legacy_many_pass_v3", "inline"),
}


class TestChunk4V3Rows:
    def test_roster_counts_6_runnable_7_visible(self):
        # sonnet-bake-off rip + v4 campaign + base scifi_codex rip (2026-07-19):
        # 3 base + 2 _v3 + 1 _v4 + custom = 7 visible / 6 runnable.
        ids = ROUTING.list_bank_ids()
        assert len(ids) == 7
        assert ids[-1] == "custom_source_bank"
        runnable = [b for b in ids if ROUTING.get_bank(b).runnable]
        assert len(runnable) == 6                  # only custom is non-runnable
        assert len([b for b in ids if b.endswith("_v2")]) == 0
        assert len([b for b in ids if b.endswith("_v3")]) == 2
        assert len([b for b in ids if b.endswith("_v4")]) == 1

    def test_v3_count_and_order(self):
        ids = ROUTING.list_bank_ids()
        assert len([b for b in ids if b.endswith("_v3")]) == 2
        # the _v3 lanes are seated after the base lanes, before custom.
        assert min(ids.index(b) for b in ids if b.endswith("_v3")) > \
            ids.index("scifi_fable2")
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
        # independence: a _v3 lane resolves its OWN rules by exact id, not base.
        assert RULES.resolve_story_rules(v3).rules_id == v3
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
        W.run_v3_advisory(led, led.data["meta"], lane="public_domain_story_v3")
        rec = led.data["meta"]["public_domain_story_v3_advisory"]
        assert rec["status"] == "ok"
        assert rec["line_count"] == 2
        assert rec["focus"]["metric"] == "compression_line_count"
        # advisory-only: spoken rows untouched (no hole in the ledger).
        assert led.data["lines"] == before
        # exactly one owned meta field written.
        assert list(led.data["meta"].keys()) == ["public_domain_story_v3_advisory"]

    def test_v3_advisory_never_raises_on_bad_input(self):
        from nodes import OTR_LedgerScriptWriter as W
        meta = {}
        W.run_v3_advisory(None, meta, lane="public_domain_story_v3")   # None led
        assert meta["public_domain_story_v3_advisory"]["status"] in ("ok", "error")

        class _Bad:
            data = {"lines": "not a list", "meta": {}}
        bad = _Bad()
        W.run_v3_advisory(bad, bad.data["meta"], lane="shakespeare_v3")
        assert "shakespeare_v3_advisory" in bad.data["meta"]


# ---------------------------------------------------------------------------
# 6. v4 improvement campaign 2026-07-17: scifi_codex_v4 is the FIRST fully
#    INDEPENDENT _v4 bank (own row + pack + rules by exact id + own pipeline;
#    the dedicated codex runner is reused DIRECTLY -- no v3 advisory wrapper).
#    `_v4` is NOT a bake-off variant suffix, so base_source_bank_id leaves it
#    unchanged (independence by construction).
# ---------------------------------------------------------------------------

class TestScifiCodexV4:
    def test_v4_is_not_a_bakeoff_variant(self):
        assert BV.base_source_bank_id("scifi_codex_v4") == "scifi_codex_v4"
        assert BV.is_bakeoff_variant("scifi_codex_v4") is False

    def test_v4_bank_wiring(self):
        from nodes import OTR_LedgerScriptWriter as W
        bank = ROUTING.get_bank("scifi_codex_v4")
        assert bank.runnable is True
        assert bank.default_story_pipeline == "scifi_codex_circuit_v4"
        assert bank.default_story_model == "scifi_codex_v4"
        assert bank.fetcher == "science_rss"
        pack = ROUTING.resolve_story_pack("scifi_codex_v4")
        assert pack.source_bank_id == "scifi_codex_v4"
        assert pack.story_model_id == "scifi_codex_v4"
        assert pack.story_pipeline_id == "scifi_codex_circuit_v4"
        # dedicated-runner lane, mapped DIRECTLY to the codex runner
        # (_run_scifi_codex_lane). base scifi_codex / scifi_codex_circuit were
        # ripped 2026-07-19; v4 owns the shared codex runner outright now.
        assert "scifi_codex_circuit_v4" in W._RUNNER_BY_PIPELINE
        assert "scifi_codex_circuit_v4" not in W._LEGACY_INLINE_PIPELINES
        assert (W._RUNNER_BY_PIPELINE["scifi_codex_circuit_v4"]
                is W._run_scifi_codex_lane)
        assert ROUTING.get_pipeline("scifi_codex_circuit_v4").executable is True

    def test_v4_owns_rules_by_exact_id(self):
        assert RULES.resolve_story_rules("scifi_codex_v4").rules_id == "scifi_codex_v4"

    def test_v4_science_floor_and_style_pool(self):
        bank = ROUTING.get_bank("scifi_codex_v4")
        assert bank.defaults.get("require_science_floor") is True
        assert bank.defaults.get("style_pool_class") == "generic"

    def test_v4_opt_in_gates_first_leg_posture(self):
        # The two deterministic gates codex's fail-closed compiler structurally
        # satisfies are ON; the gates with no codex-path authored repair stay
        # OFF for the first live leg (documented, vetoable in the bank guide_ref).
        bank = ROUTING.get_bank("scifi_codex_v4")
        assert bank.defaults.get("placeholder_guard") is True
        assert bank.defaults.get("scene_coherence_check") is True
        assert bank.defaults.get("genre_guard_spoken") in (None, False)
        assert bank.defaults.get("require_outro_cast_complete") in (None, False)
