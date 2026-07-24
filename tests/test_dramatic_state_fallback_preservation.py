"""Positive preservation coverage for dramatic-state fallback vocabulary.

Accepted source terms pass through deterministic fallback without Python noun
classification. Safety remains owned by the shared prose safety boundary.
"""
from __future__ import annotations

from nodes._otr_dramatic_state_llm import _fallback_state, _pick_templates


def _meta(key_terms):
    return {"news": {"key_terms": list(key_terms), "script_brief": ""}}


def test_fallback_preserves_first_source_term():
    meta = _meta(["transgender people", "patient records"])
    state = _fallback_state(
        meta,
        ("d001", "d002"),
        "d001",
        arc_shape="setup_complication_resolution",
    )

    assert meta["dramatic_state_fallback_term"] == "transgender people"
    rendered = " ".join([
        state.character_a_wants,
        state.character_b_wants,
        state.dramatic_question,
        state.ending_change,
    ])
    assert "transgender people" in rendered


def test_template_selection_preserves_authored_term():
    term = "the residents"
    rendered = _pick_templates(term, "")
    assert all(term in field for field in rendered)

