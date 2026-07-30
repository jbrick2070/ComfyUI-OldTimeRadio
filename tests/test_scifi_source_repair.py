import json
import hashlib

from nodes._otr_scifi_codex import FactIndexV4
from nodes._otr_scifi_codex import RadioScoreV4, ScriptArtifactV4, _schema_instruction as codex_schema_instruction
from nodes._otr_json import parse_first_json_object
from nodes._otr_scifi_source_repair import repair_literal_source_metadata
from nodes._otr_scifi_p0_contract import (
    compact_p0_repair_context,
    p0_contract_instruction,
)
from nodes._otr_structured_call import schema_required_paths


def test_repair_reindexes_exact_quote_and_zero_pads_ids_without_touching_claim():
    payload = {"full_text": "prefix exact evidence suffix"}
    raw = (
        '{"facts":[{"fact_id":"F0","claim":"keep this claim",'
        '"source_spans":[{"field":"full_text","start":0,"end":12,'
        '"quote":"exact evidence"}]}],"entities":[],"numbers":[],'
        '"tone":"measured","payload_sha256":"digest"}'
    )
    repaired = repair_literal_source_metadata(raw, FactIndexV4, payload, zero_padded_ids=True)
    assert repaired is not None
    assert repaired.facts[0].fact_id == "F01"
    assert repaired.facts[0].claim == "keep this claim"
    span = repaired.facts[0].source_spans[0]
    assert payload[span.field][span.start:span.end] == span.quote


def test_repair_refuses_paraphrased_quote():
    payload = {"full_text": "the literal source sentence"}
    raw = (
        '{"facts":[{"fact_id":"F0","claim":"claim",'
        '"source_spans":[{"field":"full_text","start":0,"end":8,'
        '"quote":"a paraphrase"}]}],"entities":[],"numbers":[],'
        '"tone":"measured","payload_sha256":"digest"}'
    )
    assert repair_literal_source_metadata(raw, FactIndexV4, payload, zero_padded_ids=True) is None


def test_repair_drops_unsupported_fact_but_keeps_literal_fact():
    payload = {"full_text": "literal supported evidence"}
    receipt = {}
    raw = (
        '{"facts":['
        '{"fact_id":"F0","claim":"unsupported", "source_spans":[{"field":"full_text","start":0,"end":5,"quote":"paraphrase"}]},'
        '{"fact_id":"F1","claim":"supported", "source_spans":[{"field":"full_text","start":0,"end":7,"quote":"literal"}]}],'
        '"entities":[],"numbers":[],"tone":"measured","payload_sha256":"digest"}'
    )
    repaired = repair_literal_source_metadata(
        raw,
        FactIndexV4,
        payload,
        zero_padded_ids=True,
        repair_receipt=receipt,
    )
    assert repaired is not None
    assert [fact.fact_id for fact in repaired.facts] == ["F02"]
    assert receipt["before_counts"]["facts"] == 2
    assert receipt["after_counts"]["facts"] == 1
    assert receipt["drops"] == [
        {
            "action": "dropped_source_span",
            "collection": "facts",
            "row_id": "F01",
            "span_index": 0,
            "field": "full_text",
            "quote_sha256": hashlib.sha256(
                b"paraphrase"
            ).hexdigest(),
            "reason": "quote_not_literal",
        },
        {
            "action": "dropped_evidence_row",
            "collection": "facts",
            "row_id": "F01",
            "reason": "no_literal_source_spans_remain",
        },
    ]
    assert "paraphrase" not in json.dumps(receipt, sort_keys=True)


def test_repair_receipts_every_prune_reason_without_persisting_quotes():
    payload = {
        "headline": "shared evidence",
        "full_text": (
            "prefix exact evidence suffix shared evidence "
            "number evidence"
        ),
    }
    raw = json.dumps({
        "facts": [
            {
                "fact_id": "F01",
                "claim": "supported",
                "source_spans": [{
                    "field": "full_text",
                    "start": 0,
                    "end": 5,
                    "quote": "exact evidence",
                }],
            },
            {
                "fact_id": "F02",
                "claim": "absent",
                "source_spans": [{
                    "field": "full_text",
                    "start": 0,
                    "end": 5,
                    "quote": "not literal anywhere",
                }],
            },
            {
                "fact_id": "F03",
                "claim": "ambiguous",
                "source_spans": [{
                    "field": "summary",
                    "start": 0,
                    "end": 5,
                    "quote": "shared evidence",
                }],
            },
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
        "payload_sha256": "digest",
    })
    receipt = {}

    repaired = repair_literal_source_metadata(
        raw,
        FactIndexV4,
        payload,
        zero_padded_ids=True,
        repair_receipt=receipt,
    )

    assert repaired is not None
    assert [fact.fact_id for fact in repaired.facts] == ["F01"]
    assert repaired.numbers == []
    assert receipt["schema_version"] == (
        "scifi_codex.literal_source_repair_receipt.v1"
    )
    assert receipt["span_index_base"] == 0
    assert receipt["allowed_source_fields"] == [
        "headline", "full_text",
    ]
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
    for quote in (
        "not literal anywhere",
        "shared evidence",
        "missing number evidence",
    ):
        assert quote not in serialized


def test_schema_rejected_repair_does_not_publish_pending_receipt():
    payload = {"full_text": "literal source"}
    raw = json.dumps({
        "facts": [{
            "fact_id": "F01",
            "claim": "unsupported",
            "source_spans": [{
                "field": "full_text",
                "start": 0,
                "end": 4,
                "quote": "not present",
            }],
        }],
        "entities": [],
        "numbers": [],
        "tone": "measured",
        "payload_sha256": "digest",
    })
    receipt = {"sentinel": "unchanged"}

    assert repair_literal_source_metadata(
        raw,
        FactIndexV4,
        payload,
        zero_padded_ids=True,
        repair_receipt=receipt,
    ) is None
    assert receipt == {"sentinel": "unchanged"}


def test_repair_rehomes_exact_quote_only_when_field_label_is_wrong():
    payload = {"headline": "short title", "full_text": "full literal evidence here"}
    raw = (
        '{"facts":[{"fact_id":"F0","claim":"claim",'
        '"source_spans":[{"field":"headline","start":0,"end":6,'
        '"quote":"full literal evidence here"}]}],"entities":[],"numbers":[],'
        '"tone":"measured","payload_sha256":"digest"}'
    )
    repaired = repair_literal_source_metadata(raw, FactIndexV4, payload, zero_padded_ids=True)
    assert repaired is not None
    span = repaired.facts[0].source_spans[0]
    assert span.field == "full_text"
    assert payload[span.field][span.start:span.end] == span.quote


def test_repair_rehomes_only_within_the_supplied_source_field_allowlist():
    payload = {
        "seed_text": "Project Apollo summary evidence",
        "headline": "Project Apollo",
        "summary": "summary evidence",
        "full_text": "distinct retained full text",
    }
    raw = json.dumps({
        "facts": [
            {
                "fact_id": "F01",
                "claim": "unsupported sibling",
                "source_spans": [{
                    "field": "full_text",
                    "start": 0,
                    "end": 5,
                    "quote": "not literal anywhere",
                }],
            },
            {
                "fact_id": "F02",
                "claim": "retained sibling",
                "source_spans": [{
                    "field": "headline",
                    "start": 0,
                    "end": 14,
                    "quote": "Project Apollo",
                }],
            },
        ],
        "entities": [],
        "numbers": [],
        "tone": "measured",
        "payload_sha256": "digest",
    })

    repaired = repair_literal_source_metadata(
        raw,
        FactIndexV4,
        payload,
        zero_padded_ids=True,
        allowed_source_fields=("seed_text", "full_text"),
    )

    assert repaired is not None
    assert [fact.fact_id for fact in repaired.facts] == ["F02"]
    span = repaired.facts[0].source_spans[0]
    assert span.field == "seed_text"
    assert payload[span.field][span.start:span.end] == "Project Apollo"


def test_repair_bounds_an_exact_oversized_quote_without_changing_the_claim():
    payload = {"full_text": "literal evidence " + ("x" * 300)}
    raw = json.dumps({
        "facts": [{
            "fact_id": "F0",
            "claim": "keep this claim",
            "source_spans": [{
                "field": "full_text", "start": 0, "end": 12,
                "quote": payload["full_text"],
            }],
        }],
        "entities": [],
        "numbers": [],
        "tone": "measured",
        "payload_sha256": "digest",
    })
    repaired = repair_literal_source_metadata(
        raw, FactIndexV4, payload, zero_padded_ids=True, max_quote_chars=32,
    )
    assert repaired is not None
    assert repaired.facts[0].claim == "keep this claim"
    span = repaired.facts[0].source_spans[0]
    assert span.start == 0
    assert span.end == 32
    assert span.quote == payload["full_text"][:32]
    assert payload[span.field][span.start:span.end] == span.quote


def test_repair_refuses_an_oversized_quote_that_is_not_literal_source_text():
    payload = {"full_text": "literal evidence " + ("x" * 300)}
    raw = json.dumps({
        "facts": [{
            "fact_id": "F0",
            "claim": "claim",
            "source_spans": [{
                "field": "full_text", "start": 0, "end": 12,
                "quote": payload["full_text"] + " invented",
            }],
        }],
        "entities": [],
        "numbers": [],
        "tone": "measured",
        "payload_sha256": "digest",
    })
    assert repair_literal_source_metadata(
        raw, FactIndexV4, payload, zero_padded_ids=True, max_quote_chars=32,
    ) is None


def test_all_lane_schema_seams_name_exact_top_level_keys():
    assert "facts" in codex_schema_instruction(FactIndexV4)
    assert "scenes[*].shots[*].scene_id" in codex_schema_instruction(RadioScoreV4)


def test_p0_contract_states_literal_slice_identity():
    instruction = p0_contract_instruction(has_numeric_tokens=True)
    assert "payload[field][start:end] == quote" in instruction
    assert "Do not paraphrase" in instruction


def test_p0_repair_context_enforces_hard_byte_limit():
    try:
        compact_p0_repair_context(
            failed_artifact="x" * 200,
            rejection="bad span",
            source_evidence={"full_text": "source"},
            source_digest="digest",
            allowed_source_fields=["full_text"],
            max_bytes=32,
        )
    except ValueError as exc:
        assert "over the hard limit" in str(exc)
    else:
        raise AssertionError("oversized P0 repair context must fail loudly")


def test_schema_instruction_contains_every_required_path_for_nested_radio_score():
    instruction = codex_schema_instruction(RadioScoreV4)
    required_paths = schema_required_paths(RadioScoreV4)
    assert required_paths
    assert all(path in instruction for path in required_paths)


def test_script_artifact_instruction_contains_every_required_path():
    instruction = codex_schema_instruction(ScriptArtifactV4)
    required_paths = schema_required_paths(ScriptArtifactV4)
    assert required_paths
    assert all(path in instruction for path in required_paths)


def test_json_parser_does_not_salvage_nested_child_from_broken_outer_object():
    broken = '{"facts":[{"fact_id":"F0","claim":"child"}'
    try:
        parse_first_json_object(broken)
    except ValueError as exc:
        assert "no decodable top-level JSON object" in str(exc)
    else:
        raise AssertionError("broken outer JSON must not return its nested child")


def test_json_parser_does_not_salvage_nested_child_from_broken_fenced_outer_object():
    broken = "Here is the artifact.\n```json\n{\"facts\":[{\"fact_id\":\"F0\",\"claim\":\"child\"}\n```"
    try:
        parse_first_json_object(broken)
    except ValueError as exc:
        assert "no decodable top-level JSON object" in str(exc)
    else:
        raise AssertionError("broken fenced JSON must not return its nested child")
