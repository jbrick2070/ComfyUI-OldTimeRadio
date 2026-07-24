"""Non-gating dramatic-state authoring coverage."""
from __future__ import annotations

import json

from nodes import _otr_dramatic_state_llm as B1


def _slot_returning(payload):
    def _slot(messages, *, temperature, max_new_tokens, **kwargs):
        return payload if isinstance(payload, str) else json.dumps(payload)
    return _slot


def _voice_slots():
    return ["d001", "d002", "d003", "d004"]


def test_first_structurally_valid_authored_state_is_preserved_exactly():
    long_value = "authored source context " * 100
    payload = {
        "character_a_wants": "same",
        "character_b_wants": "same",
        "dramatic_question": long_value,
        "ending_change": "E",
    }
    calls = 0

    def slot(messages, *, temperature, max_new_tokens, **kwargs):
        nonlocal calls
        calls += 1
        return json.dumps(payload)

    meta = {"news": {
        "key_terms": ["ancient dna"],
        "script_brief": "Scientists recover ancient DNA.",
    }}
    state = B1.derive_news_dramatic_state(
        meta=meta,
        cast_rows=[{"name": "Edna"}, {"name": "Marcus"}],
        voice_slot_ids=_voice_slots(),
        slot_fn=slot,
    )

    assert calls == 1
    assert meta["dramatic_state_source"] == "llm"
    assert state.character_a_wants == state.character_b_wants == "same"
    assert state.dramatic_question == long_value
    assert state.ending_change == "E"


def test_no_news_uses_source_neutral_fallback_without_model_call():
    calls = 0

    def slot(messages, **kwargs):
        nonlocal calls
        calls += 1
        return "unused"

    meta = {}
    state = B1.derive_news_dramatic_state(
        meta=meta,
        cast_rows=[],
        voice_slot_ids=_voice_slots(),
        slot_fn=slot,
    )

    assert calls == 0
    assert meta["dramatic_state_source"] == "fallback_no_news"
    assert "the event" in " ".join([
        state.character_a_wants,
        state.character_b_wants,
        state.dramatic_question,
    ])


def test_malformed_model_content_falls_back_without_failing_episode():
    meta = {"news": {
        "key_terms": ["the reactor"],
        "script_brief": "A reactor produces a strange signal.",
    }}
    state = B1.derive_news_dramatic_state(
        meta=meta,
        cast_rows=[],
        voice_slot_ids=_voice_slots(),
        slot_fn=_slot_returning("{ broken json"),
    )

    assert meta["dramatic_state_source"] == "fallback_llm_error"
    assert "the reactor" in " ".join([
        state.character_a_wants,
        state.character_b_wants,
        state.dramatic_question,
        state.ending_change,
    ])


def test_fallback_is_deterministic_for_same_inputs():
    def run():
        meta = {}
        return B1.derive_news_dramatic_state(
            meta=meta,
            cast_rows=[],
            voice_slot_ids=_voice_slots(),
            slot_fn=_slot_returning("unused"),
        )

    first, second = run(), run()
    assert first.model_dump() == second.model_dump()
