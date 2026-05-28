"""Sprint 4.8 (2026-05-28) -- OTR_Stage1FanOut ComfyUI wrapper.

Pinned invariants:
  * Required inputs: news_seed STRING + num_characters INT +
    target_words INT.
  * Optional inputs: episode_title_hint STRING + technical_model
    STRING (forceInput; PD6 -- no model_id widget).
  * Five STRING outputs: candidate_a/b/c_json + winning_plan_json
    + fanout_audit_json.
  * Empty news_seed short-circuits with an audit error (no LLM
    call); operator sees the failure mode immediately.
  * The class is importable + registered in the package.
"""
from __future__ import annotations

import json

from nodes.OTR_Stage1FanOut import OTR_Stage1FanOut


# ---------------------------------------------------------------------------
# Surface invariants
# ---------------------------------------------------------------------------


def test_input_types_required_and_optional():
    schema = OTR_Stage1FanOut.INPUT_TYPES()
    req = schema.get("required", {})
    opt = schema.get("optional", {})
    assert set(req.keys()) == {
        "news_seed", "num_characters", "target_words",
    }
    assert "episode_title_hint" in opt
    assert "technical_model" in opt
    # PD6: no model_id widget anywhere.
    all_keys = list(req.keys()) + list(opt.keys())
    assert "model_id" not in all_keys
    assert "creative_model" not in all_keys
    # technical_model is forceInput-only.
    assert opt["technical_model"][1].get("forceInput") is True


def test_return_types_match_audit_contract():
    assert OTR_Stage1FanOut.RETURN_TYPES == (
        "STRING", "STRING", "STRING", "STRING", "STRING",
    )
    assert OTR_Stage1FanOut.RETURN_NAMES == (
        "candidate_a_json", "candidate_b_json", "candidate_c_json",
        "winning_plan_json", "fanout_audit_json",
    )


def test_node_class_is_importable():
    import nodes.OTR_Stage1FanOut as mod
    assert hasattr(mod, "OTR_Stage1FanOut")


# ---------------------------------------------------------------------------
# Empty news_seed short-circuits
# ---------------------------------------------------------------------------


def test_empty_news_seed_returns_failure_audit_no_llm_call():
    """No LLM is invoked when the news_seed is empty; the operator
    sees the failure mode immediately via fanout_audit_json."""
    node = OTR_Stage1FanOut()
    cand_a, cand_b, cand_c, winner, audit_json = node.run(
        news_seed="",
        num_characters=2,
        target_words=350,
    )
    assert cand_a == ""
    assert cand_b == ""
    assert cand_c == ""
    assert winner == ""
    audit = json.loads(audit_json)
    assert audit["ok"] is False
    assert "empty news_seed" in audit["error"]
    assert audit["per_knob_ok"] == []


def test_whitespace_only_news_seed_treated_as_empty():
    node = OTR_Stage1FanOut()
    _, _, _, _, audit_json = node.run(
        news_seed="   \n  \t  ",
        num_characters=2,
        target_words=350,
    )
    audit = json.loads(audit_json)
    assert audit["ok"] is False
    assert "empty news_seed" in audit["error"]
