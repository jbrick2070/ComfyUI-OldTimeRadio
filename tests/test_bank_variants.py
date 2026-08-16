"""Registry coverage for the five runnable story banks."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import _otr_bank_variants as BV
from nodes import _otr_story_routing as ROUTING
from nodes import _otr_style_catalog as STYLE


LIVE_BANKS = (
    "media_archive",
    "original",
    "scifi_news_pro",
    "public_domain",
    "shakespeare",
)

EXPECTED_PIPELINES = {
    "media_archive": "legacy_many_pass",
    "original": "original_multi_pass",
    "scifi_news_pro": "scifi_news_pro_multipass",
    "public_domain": "legacy_many_pass_adapt",
    "shakespeare": "legacy_many_pass_adapt",
}


@pytest.fixture(autouse=True)
def _fresh_registry():
    ROUTING._REGISTRY = None
    yield
    ROUTING._REGISTRY = None


def test_registry_exposes_exact_five_runnable_banks_plus_custom():
    ids = ROUTING.list_bank_ids()
    assert ids == LIVE_BANKS + ("custom_source_bank",)
    assert tuple(
        bank_id for bank_id in ids if ROUTING.get_bank(bank_id).runnable
    ) == LIVE_BANKS


@pytest.mark.parametrize("bank_id", LIVE_BANKS)
def test_each_live_bank_resolves_its_own_pack_and_pipeline(bank_id):
    bank = ROUTING.require_runnable_bank(bank_id)
    pack = ROUTING.resolve_story_pack(bank_id)
    pipeline = ROUTING.get_pipeline(bank.default_story_pipeline)
    assert pack.source_bank_id == bank_id
    assert pack.story_pipeline_id == EXPECTED_PIPELINES[bank_id]
    assert pipeline.story_pipeline_id == EXPECTED_PIPELINES[bank_id]


@pytest.mark.parametrize("bank_id", LIVE_BANKS)
def test_live_bank_ids_are_not_bakeoff_variants(bank_id):
    assert BV.base_source_bank_id(bank_id) == bank_id
    assert BV.is_bakeoff_variant(bank_id) is False


@pytest.mark.parametrize("bank_id", ("public_domain", "shakespeare"))
def test_adaptation_banks_draw_only_adaptation_styles(bank_id):
    allowed = set(STYLE.adaptation_slugs())
    for seed in range(20):
        chosen = STYLE.select_style(
            "Two people face the consequence of an old promise.",
            {"source_bank": bank_id, "style_pool_class": "adaptation"},
            seed,
        )
        assert chosen in allowed


def test_media_archive_draws_from_its_curated_style_pool():
    allowed = set(STYLE.media_archive_slugs())
    for seed in range(20):
        chosen = STYLE.select_style(
            "An archivist restores a forgotten broadcast.",
            {"source_bank": "media_archive", "style_pool_class": "media"},
            seed,
        )
        assert chosen in allowed
