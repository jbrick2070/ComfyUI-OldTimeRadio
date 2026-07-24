"""Registry contract for the scifi_news_pro fixed-topology lane."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes import OTR_LedgerScriptWriter as WRITER
from nodes import _otr_story_routing as ROUTING


REPO = Path(__file__).resolve().parents[1]
PACK_DIR = REPO / "nodes" / "story_packs" / "scifi_news_pro"
FABLE2_SEAMS = frozenset(
    {
        "fable2_dossier_system",
        "fable2_pitch_system",
        "fable2_treatment_system",
        "fable2_news_read_system",
        "fable2_script_system",
        "fable2_casting_system",
    }
)


@pytest.fixture(autouse=True)
def _fresh_registry():
    ROUTING._REGISTRY = None
    yield
    ROUTING._REGISTRY = None


def test_bank_routes_to_dedicated_runner():
    bank = ROUTING.require_runnable_bank("scifi_news_pro")
    assert bank.fetcher == "science_rss"
    assert bank.default_story_pipeline == "scifi_news_pro_multipass"
    assert WRITER._RUNNER_BY_PIPELINE[bank.default_story_pipeline] is (
        WRITER._run_fable2_lane
    )


def test_pipeline_has_fixed_one_story_topology():
    pipeline = ROUTING.get_pipeline("scifi_news_pro_multipass")
    assert pipeline.executable is True
    assert pipeline.declared_seams == FABLE2_SEAMS
    assert [row.pass_id for row in pipeline.passes] == [
        "dossier",
        "pitch",
        "treatment",
        "news_read",
        "script",
        "same_story_safety_cleanup",
        "final_draft_proof",
        "casting_voices",
        "assemble",
    ]


def test_pack_owns_exactly_the_six_model_seams():
    pack = ROUTING.resolve_story_pack("scifi_news_pro")
    assert pack.source_bank_id == "scifi_news_pro"
    assert pack.story_pipeline_id == "scifi_news_pro_multipass"
    assert set(pack.prompt_stages) == FABLE2_SEAMS
    assert all(text.strip() for text in pack.prompt_stages.values())


def test_frame_deck_is_well_formed_generation_guidance():
    deck = json.loads((PACK_DIR / "frame_deck.json").read_text(encoding="utf-8"))
    assert deck["schema_version"] == "fable2_deck_v1"
    assert len(deck["cards"]) >= 3
    assert len(deck["stances"]) >= 1
    assert len({card["name"] for card in deck["cards"]}) == len(deck["cards"])
    assert all(card["name"].strip() and card["shape"].strip() for card in deck["cards"])
    assert all(row["name"].strip() and row["note"].strip() for row in deck["stances"])


def test_registry_rejects_unknown_execution_slots():
    raw = {
        "story_pipeline_id": "bad_pipe",
        "label": "Bad",
        "executable": False,
        "requires_source_contract": False,
        "declared_seams": [],
        "passes": [
            {
                "pass_id": "p",
                "slot": "judge",
                "seam_refs": [],
                "description": "invalid slot",
            }
        ],
    }
    with pytest.raises(ROUTING.RegistryValidationError, match="slot"):
        ROUTING._parse_pipeline(raw, "test fixture")
