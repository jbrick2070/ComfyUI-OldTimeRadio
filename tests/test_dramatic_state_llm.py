"""Tests for nodes/_otr_dramatic_state_llm.py -- story-quality Phase 1, B1
(the news -> story spine).

Acceptance fixtures from the converged plan:
  * an ancient-DNA-style brief -> opposed wants about the EVENT (not aliens),
    with the _DEFAULT_* boilerplate ABSENT;
  * generic-but-schema-valid LLM wants -> the post-validator rejects -> the
    deterministic news-templated fallback;
  * news=None -> templated opposed wants + a non-empty turning-slot detail
    seeded into meta["news"]["key_terms"];
  * the two wants are always opposed (never identical).

The tests drive the REAL structured_call ladder through a fake slot_fn so the
parse + validate + post-validate path is exercised end to end.
"""

import json

from nodes import _otr_dramatic_state_llm as B1
from nodes import _otr_dramatic_state as DS


# The boilerplate constants B1 must NEVER emit.
_DEFAULTS = (DS._DEFAULT_A_WANTS, DS._DEFAULT_B_WANTS)


def _slot_returning(payload):
    """A fake slot_fn matching the codebase contract
    slot_fn(messages, *, temperature, max_new_tokens, **kw) -> str."""
    def _slot(messages, *, temperature, max_new_tokens, **kw):
        return payload if isinstance(payload, str) else json.dumps(payload)
    return _slot


def _voice_slots():
    return ["d001", "d002", "d003", "d004"]


def _assert_no_default_boilerplate(state):
    blob = " ".join([
        state.character_a_wants, state.character_b_wants,
        state.dramatic_question, state.ending_change,
    ]).lower()
    for d in _DEFAULTS:
        assert d.lower() not in blob, f"_DEFAULT boilerplate leaked: {d!r}"


def test_news_grounded_llm_wants_are_used():
    meta = {"news": {
        "key_terms": ["ancient dna", "neanderthal genome"],
        "script_brief": "Scientists recover ancient DNA from a Neanderthal bone.",
    }}
    payload = {
        "character_a_wants": "publish the ancient dna results immediately",
        "character_b_wants": "bury the ancient dna findings to avoid a panic",
        "dramatic_question": "Will the ancient dna evidence reach the public?",
        "ending_change": "The ancient dna truth is released, splitting the lab.",
    }
    state = B1.derive_news_dramatic_state(
        meta=meta, cast_rows=[{"name": "Edna"}, {"name": "Marcus"}],
        voice_slot_ids=_voice_slots(), slot_fn=_slot_returning(payload),
    )
    assert meta["dramatic_state_source"] == "llm"
    assert "ancient dna" in state.character_a_wants.lower()
    assert state.character_a_wants != state.character_b_wants
    assert state.costly_choice_beat.startswith("d")
    _assert_no_default_boilerplate(state)


def test_generic_valid_wants_are_rejected_then_fallback():
    # Schema-valid but with NO news term anywhere -> post-validator rejects
    # every attempt -> structured_call exhausts -> deterministic fallback.
    meta = {"news": {
        "key_terms": ["ancient dna"],
        "script_brief": "Scientists recover ancient DNA from a bone.",
    }}
    generic = {
        "character_a_wants": "honor the established commitment whatever the cost",
        "character_b_wants": "force a compromise that protects the status quo",
        "dramatic_question": "Will the principals make the costly choice in time?",
        "ending_change": "The situation closes meaningfully different.",
    }
    state = B1.derive_news_dramatic_state(
        meta=meta, cast_rows=[{"name": "Edna"}, {"name": "Marcus"}],
        voice_slot_ids=_voice_slots(), slot_fn=_slot_returning(generic),
    )
    assert state.character_a_wants != state.character_b_wants
    # the fallback is templated on the news term, not the generic LLM output
    blob = " ".join([state.character_a_wants, state.character_b_wants,
                     state.dramatic_question, state.ending_change]).lower()
    assert "ancient dna" in blob
    _assert_no_default_boilerplate(state)


def test_news_none_uses_templated_fallback_and_seeds_key_terms():
    meta = {}  # news=None / absent
    state = B1.derive_news_dramatic_state(
        meta=meta, cast_rows=[], voice_slot_ids=_voice_slots(),
        slot_fn=_slot_returning("not json at all"),
    )
    assert meta["dramatic_state_source"] == "fallback_no_news"
    # turning-slot detail floor: key_terms is now non-empty
    assert meta["news"]["key_terms"], "a key term must be seeded for news=None"
    seeded = meta["news"]["key_terms"][0]
    assert seeded in state.character_a_wants or seeded in state.dramatic_question
    assert state.character_a_wants != state.character_b_wants
    _assert_no_default_boilerplate(state)


def test_brief_only_seeds_key_term_from_noun_phrase():
    meta = {"news": {"key_terms": [], "script_brief":
                     "A coastal town debates a new tidal energy project."}}
    state = B1.derive_news_dramatic_state(
        meta=meta, cast_rows=[], voice_slot_ids=_voice_slots(),
        slot_fn=_slot_returning("garbage"),  # force fallback
    )
    assert meta["news"]["key_terms"], "key_terms seeded from the brief"
    assert state.character_a_wants != state.character_b_wants
    _assert_no_default_boilerplate(state)


def test_malformed_llm_output_degrades_to_fallback_never_raises():
    meta = {"news": {"key_terms": ["the reactor"], "script_brief": "x"}}
    state = B1.derive_news_dramatic_state(
        meta=meta, cast_rows=[], voice_slot_ids=_voice_slots(),
        slot_fn=_slot_returning("{ broken json"),
    )
    assert state.character_a_wants != state.character_b_wants
    assert "the reactor" in " ".join([
        state.character_a_wants, state.character_b_wants,
        state.dramatic_question, state.ending_change]).lower()


def test_fallback_wants_are_opposed_for_all_templates():
    for term in ["alpha", "beta gamma", "the reactor", "zeta", "q"]:
        a, b, q, e = B1._pick_templates(term)
        assert a != b
        assert term in a and term in b


def test_deterministic_for_same_inputs():
    def _run():
        meta = {"news": {"key_terms": ["the reactor"], "script_brief": "x"}}
        return B1.derive_news_dramatic_state(
            meta=meta, cast_rows=[], voice_slot_ids=_voice_slots(),
            slot_fn=_slot_returning("garbage"),
        )
    s1, s2 = _run(), _run()
    assert s1.character_a_wants == s2.character_a_wants
    assert s1.ending_change == s2.ending_change
