from nodes._otr_scifi_codex import FactIndexV4
from nodes._otr_scifi_codex import RadioScoreV4, _schema_instruction as codex_schema_instruction
from nodes._otr_scifi_gemini import _schema_instruction as gemini_schema_instruction
from nodes._otr_scifi_sonnet import FragmentDossierV4, _schema_instruction as sonnet_schema_instruction
from nodes._otr_json import parse_first_json_object
from nodes._otr_scifi_source_repair import repair_literal_source_metadata
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
    raw = (
        '{"facts":['
        '{"fact_id":"F0","claim":"unsupported", "source_spans":[{"field":"full_text","start":0,"end":5,"quote":"paraphrase"}]},'
        '{"fact_id":"F1","claim":"supported", "source_spans":[{"field":"full_text","start":0,"end":7,"quote":"literal"}]}],'
        '"entities":[],"numbers":[],"tone":"measured","payload_sha256":"digest"}'
    )
    repaired = repair_literal_source_metadata(raw, FactIndexV4, payload, zero_padded_ids=True)
    assert repaired is not None
    assert [fact.fact_id for fact in repaired.facts] == ["F02"]


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


def test_all_lane_schema_seams_name_exact_top_level_keys():
    assert "facts" in codex_schema_instruction(FactIndexV4)
    assert "facts" in gemini_schema_instruction(FactIndexV4)
    assert "verified_facts" in sonnet_schema_instruction(FragmentDossierV4)
    assert "scenes[*].shots[*].scene_id" in codex_schema_instruction(RadioScoreV4)


def test_schema_instruction_contains_every_required_path_for_nested_radio_score():
    instruction = codex_schema_instruction(RadioScoreV4)
    required_paths = schema_required_paths(RadioScoreV4)
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
