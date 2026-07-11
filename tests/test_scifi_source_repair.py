from nodes._otr_scifi_codex import FactIndexV4
from nodes._otr_scifi_source_repair import repair_literal_source_metadata


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
