from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from nodes import _otr_scifi_codex as lane
from nodes._otr_scifi_p0_contract import p0_source_chunks


def _payload(body: str, *, seed_text: str | None = None) -> dict[str, str]:
    return {
        "headline": "Complete source",
        "summary": "A factual summary.",
        "full_text": body,
        "source": "Fixture Wire",
        "date": "2026-07-30",
        "link": "https://example.invalid/source",
        "seed_text": seed_text or "Complete source A factual summary.",
    }


def _span(field: str, source: str, start: int, length: int = 5):
    quote = source[start:start + length]
    return lane.SourceSpanV4(
        field=field, start=start, end=start + len(quote), quote=quote,
    )


def _index(
    *,
    fact_id: str,
    claim: str,
    fact_span: lane.SourceSpanV4,
    digest: str,
    entity_name: str | None = None,
    number: str | None = None,
    tone: str = "measured",
) -> lane.FactIndexV4:
    entities = []
    if entity_name is not None:
        entities.append(lane.EntityV4(
            entity_id="E01",
            name=entity_name,
            source_spans=[fact_span.model_copy(deep=True)],
        ))
    numbers = []
    if number is not None:
        numbers.append(lane.NumberV4(
            number_id="N01",
            verbatim=number,
            fact_id=fact_id,
            source_span=fact_span.model_copy(deep=True),
        ))
    return lane.FactIndexV4(
        facts=[lane.FactV4(
            fact_id=fact_id,
            claim=claim,
            source_spans=[fact_span],
            numeric_tokens=[number] if number else [],
        )],
        entities=entities,
        numbers=numbers,
        tone=tone,
        payload_sha256=digest,
    )


def test_rss_projection_retains_complete_body_before_derived_seed_alias():
    body = "complete body text"
    payload = _payload(body, seed_text=f"headline {body} summary")

    rss, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "rss_fetch", "target_words": 45},
    )
    pinned, _ = lane.validate_payload_envelope(
        payload, {"seed_source": "custom_premise", "target_words": 45},
    )

    rss_projection, _ = lane._p0_evidence_projection(rss)
    pinned_projection, _ = lane._p0_evidence_projection(pinned)
    assert next(iter(rss_projection)) == "full_text"
    assert "full_text" in rss_projection
    assert next(iter(pinned_projection)) == "seed_text"
    assert "full_text" not in pinned_projection


def test_rss_admission_keeps_body_above_old_48k_but_pinned_remains_bounded():
    body = "x" * 60_000
    envelope, _ = lane.validate_payload_envelope(
        _payload(body), {"seed_source": "rss_fetch", "target_words": 45},
    )
    assert envelope.payload.full_text == body

    with pytest.raises(lane.CodexPayloadOversizeError, match="48,000"):
        lane.validate_payload_envelope(
            _payload(body),
            {"seed_source": "custom_premise", "target_words": 45},
        )


def test_rss_body_above_secure_character_bound_refuses_instead_of_slicing():
    body = "x" * ((2 * 1024 * 1024) + 1)
    with pytest.raises(lane.CodexPayloadOversizeError, match="2 MiB"):
        lane.validate_payload_envelope(
            _payload(body), {"seed_source": "rss_fetch", "target_words": 45},
        )


def test_overlapping_windows_cover_every_character_and_every_max_quote():
    body = "".join(chr(65 + (index % 26)) for index in range(1600))
    payload = {"headline": "frame", "full_text": body}
    budget = 510
    overlap = lane.MAX_QUOTE_CHARS - 1

    windows = p0_source_chunks(
        payload, budget_chars=budget, overlap_chars=overlap,
    )
    allowance = budget - len(payload["headline"])
    intervals = []
    for start, window in windows:
        text = window["full_text"]
        assert text == body[start:start + len(text)]
        assert len(text) <= allowance
        intervals.append((start, start + len(text)))
    assert intervals[0][0] == 0
    assert intervals[-1][1] == len(body)
    for prior, current in zip(intervals, intervals[1:]):
        assert current[0] == prior[1] - overlap
        assert current[0] > prior[0]
    for start in range(0, len(body) - lane.MAX_QUOTE_CHARS + 1):
        end = start + lane.MAX_QUOTE_CHARS
        assert any(left <= start and end <= right for left, right in intervals)


def test_sentence_rewind_never_stalls_overlap_progress():
    body = ("a" * 260) + ". " + ("b" * 600)
    windows = p0_source_chunks(
        {"full_text": body},
        budget_chars=500,
        overlap_chars=300,
    )
    first_start, first = windows[0]
    assert first_start == 0
    assert len(first["full_text"]) == 500
    assert windows[1][0] == 200


@pytest.mark.parametrize("overlap", (-1, 500))
def test_invalid_overlap_is_rejected(overlap):
    with pytest.raises(ValueError, match="overlap_chars"):
        p0_source_chunks(
            {"full_text": "x" * 1000},
            budget_chars=500,
            overlap_chars=overlap,
        )


def test_rebase_offsets_only_full_text_spans_and_preserves_duplicate_occurrence():
    full_text = "alpha DUPLICATE omega " + ("x" * 20) + " DUPLICATE tail"
    second = full_text.rfind("DUPLICATE")
    local = "DUPLICATE tail"
    digest = "d" * 64
    local_index = lane.FactIndexV4(
        facts=[
            lane.FactV4(
                fact_id="F01",
                claim="Later occurrence",
                source_spans=[_span("full_text", local, 0, 9)],
            ),
            lane.FactV4(
                fact_id="F02",
                claim="Headline evidence",
                source_spans=[_span("headline", "Complete source", 0, 8)],
            ),
        ],
        entities=[lane.EntityV4(
            entity_id="E01",
            name="Duplicate",
            source_spans=[_span("full_text", local, 0, 9)],
        )],
        numbers=[lane.NumberV4(
            number_id="N01",
            verbatim="DUPLICATE",
            fact_id="F01",
            source_span=_span("full_text", local, 0, 9),
        )],
        tone="measured",
        payload_sha256="local",
    )
    before = copy.deepcopy(local_index)

    rebased = lane._rebase_p0_index(
        local_index, full_text_offset=second, a0_digest=digest,
    )

    assert local_index == before
    assert rebased.facts[0].source_spans[0].start == second
    assert rebased.entities[0].source_spans[0].start == second
    assert rebased.numbers[0].source_span.start == second
    assert rebased.facts[1].source_spans[0].start == 0
    assert lane._validate_fact_index(
        rebased,
        {"full_text": full_text, "headline": "Complete source"},
        allowed_source_fields=frozenset({"full_text", "headline"}),
        expected_payload_sha256=digest,
        relocate_mismatched_spans=False,
    ) is None


def test_merge_namespaces_local_ids_transfers_numbers_and_remaps_contiguously():
    body = "AAAAA middle BBBBB tail CCCCC"
    digest = "a" * 64
    first = _index(
        fact_id="F01",
        claim="Shared   claim",
        fact_span=_span("full_text", body, 0),
        digest=digest,
        entity_name="Shared Entity",
    )
    duplicate_with_number = _index(
        fact_id="F01",
        claim="shared claim",
        fact_span=_span("full_text", body, 0),
        digest=digest,
        entity_name="shared entity",
        number="AAAAA",
    )
    different_same_local_id = _index(
        fact_id="F01",
        claim="Different claim",
        fact_span=_span("full_text", body, 13),
        digest=digest,
        number="BBBBB",
        tone="later",
    )

    merged = lane._merge_p0_indices(
        [first, duplicate_with_number, different_same_local_id],
        a0_payload={"full_text": body},
        allowed_source_fields=frozenset({"full_text"}),
        a0_digest=digest,
    )

    assert [fact.fact_id for fact in merged.facts] == ["F01", "F02"]
    assert [number.number_id for number in merged.numbers] == ["N01", "N02"]
    assert [number.fact_id for number in merged.numbers] == ["F01", "F02"]
    assert len(merged.entities) == 1
    assert merged.tone == "measured"
    assert merged.payload_sha256 == digest
    assert lane._validate_fact_index(
        merged,
        {"full_text": body},
        allowed_source_fields=frozenset({"full_text"}),
        expected_payload_sha256=digest,
        relocate_mismatched_spans=False,
    ) is None


def test_even_sampling_includes_tail_and_is_deterministic():
    assert lane._evenly_spaced_indices(8, 6) == [0, 1, 2, 4, 5, 7]
    assert lane._evenly_spaced_indices(6, 4) == [0, 1, 3, 5]


def test_production_runner_windows_rebases_and_hands_merged_tail_to_p1(
        monkeypatch, tmp_path: Path):
    body = "".join(f"[{index:04d}] factual detail. " for index in range(90))
    payload = _payload(body)
    real_chunks = lane.p0_source_chunks
    chunk_calls = []
    p0_calls = []
    p1_inputs = {}

    def recording_chunks(source, **kwargs):
        chunk_calls.append(kwargs)
        return real_chunks(source, **kwargs)

    class StopAfterP0(RuntimeError):
        pass

    def fake_invoke(**kwargs):
        if kwargs["pass_id"] == "P0":
            p0_calls.append(kwargs["artifact_inputs"])
            evidence = kwargs["artifact_inputs"]["payload"]["payload"]
            source = evidence["full_text"]
            quote = source[-20:]
            start = len(source) - len(quote)
            result = _index(
                fact_id="F01",
                claim=f"window fact {len(p0_calls)}",
                fact_span=_span("full_text", source, start, len(quote)),
                digest=kwargs["artifact_inputs"]["payload"]["source_digest"],
            )
            assert kwargs["post_validator"](result) is None
            assert kwargs["retry_until_valid"] is True
            return result
        if kwargs["pass_id"] == "P1":
            p1_inputs.update(kwargs["artifact_inputs"])
            raise StopAfterP0
        raise AssertionError(f"unexpected pass {kwargs['pass_id']}")

    monkeypatch.setattr(lane, "p0_source_char_budget", lambda **_kwargs: 600)
    monkeypatch.setattr(lane, "p0_source_chunks", recording_chunks)
    monkeypatch.setattr(lane, "invoke_codex_structured", fake_invoke)

    with pytest.raises(StopAfterP0):
        lane.run_scifi_codex_episode(
            payload=payload,
            pack=SimpleNamespace(),
            resolved={
                "seed_source": "rss_fetch",
                "target_words": 45,
                "technical_model": "technical",
                "creative_writing_model": "creative",
            },
            led=SimpleNamespace(),
            meta={},
            creative_fn=lambda *_args, **_kwargs: "",
            technical_fn=lambda *_args, **_kwargs: "",
            slot_scheduler=None,
            source_bank_row=None,
            episode_root=tmp_path,
            episode_id="window-fixture",
        )

    assert len(p0_calls) > 1
    assert chunk_calls == [{
        "budget_chars": 600,
        "overlap_chars": lane.MAX_QUOTE_CHARS - 1,
    }]
    merged = lane.FactIndexV4.model_validate(p1_inputs["fact_index"])
    assert merged.payload_sha256 == lane._digest(
        lane.validate_payload_envelope(
            payload, {"seed_source": "rss_fetch", "target_words": 45},
        )[0].payload.model_dump(mode="json")
    )
    assert any(
        span.end > len(body) - 100
        for fact in merged.facts
        for span in fact.source_spans
        if span.field == "full_text"
    )
