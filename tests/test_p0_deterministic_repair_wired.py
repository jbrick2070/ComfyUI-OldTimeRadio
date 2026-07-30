"""P0's deterministic span repair is WIRED, not merely written.

tests/test_scifi_source_repair.py already proves repair_literal_source_metadata
does its job. That was never the problem. The problem, found in the live
45-word campaign on 2026-07-29, was that it had ZERO call sites: it was
imported into the codex lane in two places and invoked from none, while P0 --
which owned 15 of the 24 writer deaths in that campaign -- routed every
failure straight to an LLM repair owner.

WHAT THIS RUNG DOES AND DOES NOT ADD. `_span_ok` (_otr_scifi_codex.py:1158)
ALREADY self-heals the narrow case of wrong start/end when the quote is
literal text of its own declared field -- it does a `find` and snaps the
coordinates inside the validator. A first draft of this file used that case as
its fixture and proved nothing, because the pass never failed. The
deterministic rung earns its place on the cases `_span_ok` cannot reach:

  1. the quote is literal but the FIELD LABEL is wrong (`_span_ok` searches
     only the declared field and gives up),
  2. an evidence row is genuinely unsupported while OTHER rows are fine --
     today one bad row kills the whole index and the episode with it,
  3. malformed ids (F0 -> F01),
  4. an exact quote wider than the schema cap.

Case 2 is the operator's "degrade, never veto" ruling in miniature: drop the
row that cannot be supported, keep the ones that can, and let the episode
live.

UTF-8, no BOM, ASCII-only source. No GPU, no network.
"""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

import pytest

from nodes import _otr_scifi_codex as lane
from nodes._otr_scifi_codex import FactIndexV4, _validate_fact_index
from nodes._otr_scifi_p0_contract import MAX_QUOTE_CHARS
from nodes._otr_scifi_source_repair import repair_literal_source_metadata


PAYLOAD = {
    "headline": "short title",
    "full_text": "prefix exact evidence suffix",
}
ALLOWED = ("headline", "full_text")
DIGEST = "digest"


def _index(facts):
    return json.dumps({
        "facts": facts,
        "entities": [],
        "numbers": [],
        "tone": "measured",
        "payload_sha256": DIGEST,
    })


def _fact(fact_id, claim, field, start, end, quote):
    return {
        "fact_id": fact_id,
        "claim": claim,
        "source_spans": [{
            "field": field, "start": start, "end": end, "quote": quote,
        }],
    }


# CASE 2 -- one unsupported row beside a good one. Today the whole index is
# rejected and the episode dies for the sake of one row.
ONE_BAD_ROW = _index([
    _fact("F01", "unsupported claim", "full_text", 0, 5,
          "a paraphrase absent from the source"),
    _fact("F02", "supported claim", "full_text", 0, 7, "exact evidence"),
])

# CASE 1 -- literal quote, wrong field label. _span_ok searches "headline"
# only, never finds it, and gives up.
WRONG_FIELD = _index([
    _fact("F01", "the claim the model authored", "headline", 0, 14,
          "exact evidence"),
])

RECEIPT_PAYLOAD = {
    "headline": "shared evidence",
    "full_text": (
        "prefix exact evidence suffix shared evidence number evidence"
    ),
}
RECEIPT_PRIMARY = json.dumps({
    "facts": [
        _fact(
            "F01", "supported", "full_text", 0, 5, "exact evidence",
        ),
        _fact(
            "F02", "absent", "full_text", 0, 5,
            "not literal anywhere",
        ),
        _fact(
            "F03", "ambiguous", "summary", 0, 5, "shared evidence",
        ),
        {
            "fact_id": "F04",
            "claim": "malformed",
            "source_spans": "not-a-list",
        },
    ],
    "entities": [],
    "numbers": [
        {
            "number_id": "N01",
            "verbatim": "number",
            "fact_id": "F02",
            "source_span": {
                "field": "full_text",
                "start": 0,
                "end": 6,
                "quote": "number evidence",
            },
        },
        {
            "number_id": "N02",
            "verbatim": "missing",
            "fact_id": "F01",
            "source_span": {
                "field": "full_text",
                "start": 0,
                "end": 7,
                "quote": "missing number evidence",
            },
        },
    ],
    "tone": "measured",
    "payload_sha256": DIGEST,
})


def _p0_validator(value):
    return _validate_fact_index(
        value, PAYLOAD,
        allowed_source_fields=ALLOWED,
        expected_payload_sha256=DIGEST,
    )


def _p0_repair(failed_output, repair_receipt):
    return repair_literal_source_metadata(
        failed_output, FactIndexV4, PAYLOAD,
        zero_padded_ids=True, max_quote_chars=MAX_QUOTE_CHARS,
        repair_receipt=repair_receipt,
    )


def _receipt_validator(value):
    return _validate_fact_index(
        value,
        RECEIPT_PAYLOAD,
        allowed_source_fields=frozenset(RECEIPT_PAYLOAD),
        expected_payload_sha256=DIGEST,
    )


def _receipt_repair(failed_output, repair_receipt):
    return repair_literal_source_metadata(
        failed_output,
        FactIndexV4,
        RECEIPT_PAYLOAD,
        zero_padded_ids=True,
        max_quote_chars=MAX_QUOTE_CHARS,
        repair_receipt=repair_receipt,
    )


def _invoke(deterministic_repair_fn, primary_output,
            post_validator=_p0_validator, alternate_output=None):
    """Run the REAL ladder (structured_call is not stubbed) and report which
    rungs were reached."""
    seen = {"primary": 0, "alternate": 0}

    def primary(_messages, **_kwargs):
        seen["primary"] += 1
        return primary_output

    def alternate(_messages, **_kwargs):
        seen["alternate"] += 1
        return (
            primary_output
            if alternate_output is None else alternate_output
        )

    journal = {}
    result = lane.invoke_codex_structured(
        pass_id="P0",
        slot="technical",
        slot_fn=primary,
        pack=SimpleNamespace(
            prompt_stages={"codex_fact_index_system": "seam"}),
        seam_refs=("codex_fact_index_system",),
        artifact_inputs={},
        result_type=FactIndexV4,
        post_validator=post_validator,
        base_temperature=.20,
        structural_retry_temperature=.10,
        max_new_tokens=None,
        call_journal=journal,
        repair_slot_fn=alternate,
        repair_ledger_builder=lambda **_kw: [
            {"role": "user", "content": "repair"}],
        repair_owner_id="creative",
        deterministic_repair_fn=deterministic_repair_fn,
    )
    return result, seen, journal


def test_one_unsupported_row_no_longer_kills_the_whole_index():
    """The operator's ruling in miniature: drop what cannot be supported,
    keep what can, let the episode live."""
    result, seen, journal = _invoke(_p0_repair, ONE_BAD_ROW)

    assert isinstance(result, FactIndexV4)
    assert [f.fact_id for f in result.facts] == ["F02"]
    assert result.facts[0].claim == "supported claim"
    span = result.facts[0].source_spans[0]
    assert PAYLOAD[span.field][span.start:span.end] == span.quote
    assert seen["alternate"] == 0, "the deterministic rung must win first"
    call = journal["calls"][0]
    assert call["terminal_disposition"] == (
        "accepted_after_deterministic_repair"
    )
    assert call["repair_attempted"] is False
    assert call["deterministic_repair_receipt"]["drops"]
    assert call["accepted"] == result.model_dump(mode="json")


def test_without_the_wire_that_same_index_dies():
    """The counterfactual. If the hook is removed, this file fails loudly
    instead of quietly measuring nothing."""
    with pytest.raises(lane.CodexPassError):
        _invoke(None, ONE_BAD_ROW)


def test_every_deterministic_prune_reason_is_durable_in_the_journal():
    result, seen, journal = _invoke(
        _receipt_repair,
        RECEIPT_PRIMARY,
        post_validator=_receipt_validator,
    )

    assert [fact.fact_id for fact in result.facts] == ["F01"]
    assert result.numbers == []
    assert seen == {"primary": 1, "alternate": 0}
    call = journal["calls"][0]
    assert call["terminal_disposition"] == (
        "accepted_after_deterministic_repair"
    )
    receipt = call["deterministic_repair_receipt"]
    assert receipt["before_counts"] == {
        "facts": 4, "entities": 0, "numbers": 2,
    }
    assert receipt["after_counts"] == {
        "facts": 1, "entities": 0, "numbers": 0,
    }
    events = {
        (
            event["action"],
            event["row_id"],
            event["reason"],
        )
        for event in receipt["drops"]
    }
    assert events == {
        ("dropped_source_span", "F02", "quote_not_literal"),
        (
            "dropped_evidence_row",
            "F02",
            "no_literal_source_spans_remain",
        ),
        ("dropped_source_span", "F03", "ambiguous_source_field"),
        (
            "dropped_evidence_row",
            "F03",
            "no_literal_source_spans_remain",
        ),
        ("dropped_evidence_row", "F04", "malformed_evidence_row"),
        ("dropped_dependent_row", "N01", "fact_removed"),
        ("dropped_source_span", "N02", "quote_not_literal"),
        (
            "dropped_dependent_row",
            "N02",
            "no_literal_source_spans_remain",
        ),
    }
    serialized = json.dumps(receipt, sort_keys=True)
    assert "not literal anywhere" not in serialized
    assert "missing number evidence" not in serialized


def test_a_literal_quote_under_the_wrong_field_label_is_rehomed():
    """_span_ok searches only the declared field, so this is dead to it."""
    result, seen, journal = _invoke(_p0_repair, WRONG_FIELD)

    span = result.facts[0].source_spans[0]
    assert span.field == "full_text"
    assert PAYLOAD[span.field][span.start:span.end] == span.quote
    assert result.facts[0].claim == "the claim the model authored"
    assert seen["alternate"] == 0
    assert journal["calls"][0]["terminal_disposition"] == (
        "accepted_after_deterministic_repair"
    )


def test_without_the_wire_the_wrong_field_label_dies():
    with pytest.raises(lane.CodexPassError):
        _invoke(None, WRONG_FIELD)


def test_a_deterministic_repair_gets_no_privileges_an_llm_repair_lacks():
    """A repair that does not satisfy the pass's real validator is refused.
    The rung is a shortcut past the LLM, never past the contract."""
    with pytest.raises(lane.CodexPassError):
        _invoke(_p0_repair, ONE_BAD_ROW,
                post_validator=lambda _value: "rejected anyway")


def test_rejected_deterministic_candidate_does_not_publish_its_receipt():
    clean_alternate = _index([
        _fact(
            "F02",
            "alternate accepted claim",
            "full_text",
            7,
            21,
            "exact evidence",
        ),
    ])
    clean_candidate_checks = 0

    def validator(value):
        nonlocal clean_candidate_checks
        error = _p0_validator(value)
        if error is not None:
            return error
        clean_candidate_checks += 1
        if clean_candidate_checks == 2:
            return "reject the deterministic candidate on final validation"
        return None

    result, seen, journal = _invoke(
        _p0_repair,
        ONE_BAD_ROW,
        post_validator=validator,
        alternate_output=clean_alternate,
    )

    assert result.facts[0].claim == "alternate accepted claim"
    assert seen == {"primary": 1, "alternate": 1}
    call = journal["calls"][0]
    assert call["terminal_disposition"] == "accepted_after_repair"
    assert call["repair_attempted"] is True
    assert "deterministic_repair_receipt" not in call


def test_an_index_with_no_supportable_row_stays_loud():
    """Fail-closed: the rung repairs coordinates and prunes, it never invents
    evidence. Prune everything and schema min_length=1 still kills the pass,
    which is correct -- an index with no evidence is not a degraded index, it
    is no index."""
    all_bad = _index([
        _fact("F01", "claim", "full_text", 0, 8, "not in the source at all"),
    ])
    with pytest.raises(lane.CodexPassError):
        _invoke(_p0_repair, all_bad)


def test_the_production_p0_call_site_passes_the_hook():
    """THE ANTI-REGRESSION GUARD FOR THIS ENTIRE DEFECT CLASS.

    Every test above would still pass if run_scifi_codex_episode stopped
    wiring the hook -- which is precisely how repair_literal_source_metadata
    came to be imported twice and called never. Assert the real call site.
    """
    source = inspect.getsource(lane.run_scifi_codex_episode)
    assert "deterministic_repair_fn=" in source, (
        "run_scifi_codex_episode no longer wires a deterministic repair; "
        "P0's dominant live failure is back on the LLM repair owner"
    )
    assert "repair_literal_source_metadata(" in source, (
        "the deterministic hook is wired to something other than the tested "
        "literal-source repairer"
    )
    assert "repair_receipt=repair_receipt" in source, (
        "the production repairer no longer fills the pending receipt sink"
    )
    assert "deterministic_repair_fn=p0_deterministic_repair" in source, (
        "the production P0 call no longer wires the receipted repair closure"
    )
