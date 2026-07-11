from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_scifi_codex as lane
from nodes import _otr_story_routing as routing


def _payload(words: int = 90) -> dict[str, str]:
    text = ("The observatory measured a quiet signal from a distant ice moon "
            "through a calibrated antenna during a cautious orbital survey. " * words)
    return {
        "headline": "Observatory measures a quiet signal",
        "summary": "Researchers report a cautious result.",
        "full_text": text,
        "source": "Test Wire",
        "date": "2026-07-11",
        "link": "https://example.invalid/story",
        "seed_text": text,
    }


def test_codex_pack_and_registry_contract():
    bank = routing.get_bank("scifi_codex")
    assert bank.runnable is True
    assert bank.default_story_pipeline == "scifi_codex_circuit"
    pack = routing.resolve_story_pack("scifi_codex")
    assert pack.story_model_id == "scifi_codex_v1"
    assert set(pack.prompt_stages) == set(routing.get_pipeline("scifi_codex_circuit").declared_seams)
    pack_path = Path(__file__).parents[1] / "nodes" / "story_packs" / "scifi_codex" / "scifi_codex_v1.json"
    assert json.loads(pack_path.read_text(encoding="utf-8"))["source_bank_id"] == "scifi_codex"


def test_payload_route_and_one_use_word_steer():
    env, steer = lane.validate_payload_envelope(_payload(), {"seed_source": "rss_fetch", "target_words": 721})
    assert env.source_mode == "rss"
    assert steer.requested_words == 721
    pinned = dict(_payload())
    pinned["seed_text"] = "A pinned premise with enough distinct words for testing."
    env, _ = lane.validate_payload_envelope(pinned, {"seed_source": "custom_premise", "target_words": 30})
    assert env.source_mode == "operator_pinned"
    with pytest.raises(lane.CodexPayloadRouteError):
        lane.validate_payload_envelope(_payload(), {"seed_source": "other", "target_words": 720})


def test_literal_fact_spans_and_reject_only_spoken_hygiene():
    payload = _payload()
    quote = payload["headline"][0:10]
    fact = lane.FactIndexV4(
        facts=[lane.FactV4(fact_id="F01", claim="a signal", source_spans=[lane.SourceSpanV4(field="headline", start=0, end=10, quote=quote)])],
        entities=[], numbers=[], tone="cautious", payload_sha256="0" * 64,
    )
    assert lane._validate_fact_index(fact, payload) is None
    broken = fact.model_copy(update={"facts": [fact.facts[0].model_copy(update={"source_spans": [lane.SourceSpanV4(field="headline", start=0, end=10, quote="wrong")]} )]})
    assert lane._validate_fact_index(broken, payload)
    assert lane._spoken_error("(pause) the signal arrives", "Iona")
    assert lane._spoken_error("NASA confirms it", "Iona")
    assert lane._spoken_error("Iona, listen", "Iona")
    assert lane._spoken_error("The signal arrives", "Iona") is None


def test_advisory_centers_do_not_require_requested_count():
    plan = lane.make_advisory_word_blueprint(719, ["b001", "b002", "b003"])
    assert plan.advisory_total_center == 719
    assert sum(x["advisory_word_center"] for x in plan.per_beat) == 719
