"""tests/test_otr_story_room_extract.py -- Sprint 10B Wave 2 Agent G.

Unit tests for transcript-to-structured extraction. Pure tests: no
torch, no GPU, no network. The constrained-decode generate_fn is
stubbed with hand-built JSON payloads so the extraction body is
exercised against deterministic transcripts.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from nodes import _otr_story_room_extract as ext_mod


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _good_extraction_payload() -> dict:
    """Hand-built schema-valid extraction for the happy-path tests."""
    return {
        "cast": [
            {
                "name": "REN BLACK",
                "gender": "male",
                "pronouns": "he/him",
                "voice_id": "v2/en_speaker_6",
                "persona": (
                    "Station operator on Maui; ex-acoustic engineer. "
                    "Wary, deliberate, tends to clip his sentences."
                ),
                "arc_role": "reluctant insider",
            },
            {
                "name": "DR MAEVE COLE",
                "gender": "female",
                "pronouns": "she/her",
                "voice_id": "v2/en_speaker_9",
                "persona": (
                    "Lead researcher on the recording project. "
                    "Persuasive, evidence-first, quietly tired."
                ),
                "arc_role": "pressure source",
            },
        ],
        "beats": [
            {
                "beat_id": "b001",
                "speaker": "ANNOUNCER",
                "intent": "Open the episode with a one-line frame.",
                "length_target_words": 25,
                "emotional_register": "level",
            },
            {
                "beat_id": "b002",
                "speaker": "DR MAEVE COLE",
                "intent": "State the recording match concretely.",
                "length_target_words": 60,
                "emotional_register": "measured urgency",
            },
            {
                "beat_id": "b003",
                "speaker": "REN BLACK",
                "intent": "Refuse and explain why.",
                "length_target_words": 50,
                "emotional_register": "guarded",
            },
            {
                "beat_id": "b004",
                "speaker": "ANNOUNCER",
                "intent": "Close.",
                "length_target_words": 20,
                "emotional_register": "level",
            },
        ],
        "dialogue": [
            {
                "beat_id": "b001",
                "speaker": "ANNOUNCER",
                "text": (
                    "A whaling-era recording, a living pod, and a "
                    "phone call neither one of them wants to make."
                ),
            },
            {
                "beat_id": "b002",
                "speaker": "DR MAEVE COLE",
                "text": (
                    "The 1947 cylinder matched a humpback off Maui "
                    "this week. Three independent passes, ninety-one "
                    "percent confidence."
                ),
            },
            {
                "beat_id": "b003",
                "speaker": "REN BLACK",
                "text": (
                    "If we publish coordinates the boats come. I am "
                    "not signing that paper."
                ),
            },
            {
                "beat_id": "b004",
                "speaker": "ANNOUNCER",
                "text": (
                    "A song from another century, and a choice. "
                    "Good night."
                ),
            },
        ],
        "audio_cues": [
            {
                "beat_id": "b001",
                "kind": "music_open",
                "cue": "low cello bed, slow pulse",
            },
            {
                "beat_id": "b004",
                "kind": "music_close",
                "cue": "single sustained note, decay to silence",
            },
        ],
        "running_facts": [
            "the 1947 cylinder is from a museum archive",
            "Suri's pod is off Maui's west coast",
        ],
        "arc": {
            "setup": (
                "Two researchers have a recording match nobody else "
                "knows about yet."
            ),
            "complication": (
                "Publication will draw boats that could harass the "
                "pod."
            ),
            "resolution": (
                "They publish sound-only, no coordinates, and accept "
                "the slower vindication."
            ),
        },
        "premise": (
            "A 1947 humpback aria recording has been matched to a "
            "living pod off Maui."
        ),
    }


def _transcript_payload(final_draft: str = "REN BLACK: Stub draft.") -> dict:
    """Hand-built StoryRoomTranscript-shaped dict for the prompt builder."""
    return {
        "director_brief": {
            "news_premise": "1947 humpback aria matched to living pod",
            "dramatic_question": "do they publish coordinates",
            "opposed_desires": "MAEVE wants to publish; REN wants the pod left alone",
            "arc_shape": "argument, refusal, partial publication",
            "audience_feel": "quiet weight",
            "forbidden_drift": ["alarm-bell theatrics"],
            "raw_brief": "Two researchers, one recording, one choice.",
        },
        "turns": [],
        "editor_verdicts": [],
        "final_draft": final_draft,
        "terminated_clean": True,
        "total_turns": 5,
        "elapsed_seconds": 12.0,
    }


def make_generate_stub(payloads, *, raise_on=None):
    """Build a generate_fn that returns each JSON payload in order.

    payloads: list[dict | str] -- dicts get json.dumps'd; strings are
              returned verbatim (so a test can feed malformed output).
    raise_on: optional set[int] of 1-indexed call indices that raise.
    """
    calls = {"n": 0}
    raise_on = raise_on or set()

    def gen(messages, *, temperature, max_new_tokens):
        calls["n"] += 1
        if calls["n"] in raise_on:
            raise RuntimeError(f"stub raise at call {calls['n']}")
        payload = payloads[(calls["n"] - 1) % len(payloads)]
        if isinstance(payload, str):
            return payload
        return json.dumps(payload)

    gen.calls = calls
    return gen


# ---------------------------------------------------------------------------
# Schema + dataclass shape pins (Section 3.4)
# ---------------------------------------------------------------------------


def test_storyroom_extraction_dataclass_fields():
    """StoryRoomExtraction must expose the Section 3.4 fields."""
    e = ext_mod.StoryRoomExtraction(
        cast=[], beats=[], dialogue=[],
    )
    assert e.cast == []
    assert e.beats == []
    assert e.dialogue == []
    assert e.audio_cues == []
    assert e.running_facts == []
    assert e.arc is None
    assert e.premise == ""


def test_storyroom_extraction_to_dict_round_trip():
    """to_dict round-trips through json.dumps without raising."""
    payload = _good_extraction_payload()
    schema = ext_mod.StoryRoomExtractionSchema.model_validate(payload)
    extraction = ext_mod._schema_to_extraction(schema)
    d = extraction.to_dict()
    s = json.dumps(d)
    again = json.loads(s)
    assert again["premise"].startswith("A 1947 humpback")
    assert len(again["cast"]) == 2
    assert len(again["beats"]) == 4
    assert again["dialogue"][0]["speaker"] == "ANNOUNCER"
    assert again["audio_cues"][0]["kind"] == "music_open"


def test_extraction_schema_requires_min_rows():
    """Schema bounds reject empty lists where the design forbids them."""
    bad = _good_extraction_payload()
    bad["dialogue"] = []
    with pytest.raises(Exception):
        ext_mod.StoryRoomExtractionSchema.model_validate(bad)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_extract_from_transcript_happy_path():
    """One clean call -> populated StoryRoomExtraction."""
    gen = make_generate_stub([_good_extraction_payload()])
    extraction = ext_mod.extract_from_transcript(
        _transcript_payload(),
        generate_fn=gen,
        cast_names=["REN BLACK", "DR MAEVE COLE"],
        news_seed="1947 humpback aria recording",
    )
    assert len(extraction.cast) == 2
    assert len(extraction.beats) == 4
    assert len(extraction.dialogue) == 4
    assert extraction.dialogue[0]["beat_id"] == "b001"
    assert extraction.dialogue[0]["speaker"] == "ANNOUNCER"
    assert "1947" in extraction.premise
    assert gen.calls["n"] == 1


def test_extract_dialogue_keys_match_section_3_4():
    """dialogue rows MUST carry {beat_id, speaker, text}."""
    gen = make_generate_stub([_good_extraction_payload()])
    extraction = ext_mod.extract_from_transcript(
        _transcript_payload(), generate_fn=gen,
    )
    for row in extraction.dialogue:
        assert set(row.keys()) == {"beat_id", "speaker", "text"}, (
            f"Section 3.4 contract drift: {row.keys()}"
        )


def test_extract_audio_cue_keys():
    """audio_cues rows MUST carry {beat_id, kind, cue}."""
    gen = make_generate_stub([_good_extraction_payload()])
    extraction = ext_mod.extract_from_transcript(
        _transcript_payload(), generate_fn=gen,
    )
    for row in extraction.audio_cues:
        assert set(row.keys()) == {"beat_id", "kind", "cue"}


# ---------------------------------------------------------------------------
# Retry / failure paths
# ---------------------------------------------------------------------------


def test_extract_retries_on_invalid_json():
    """First attempt returns garbage; second succeeds -> 2 calls."""
    gen = make_generate_stub([
        "not json at all",
        _good_extraction_payload(),
    ])
    extraction = ext_mod.extract_from_transcript(
        _transcript_payload(), generate_fn=gen,
    )
    assert extraction is not None
    assert gen.calls["n"] == 2


def test_extract_retries_on_schema_violation():
    """First attempt fails schema; second succeeds."""
    bad = _good_extraction_payload()
    bad["dialogue"] = []  # violates min_length on dialogue
    gen = make_generate_stub([bad, _good_extraction_payload()])
    extraction = ext_mod.extract_from_transcript(
        _transcript_payload(), generate_fn=gen,
    )
    assert extraction is not None
    assert gen.calls["n"] == 2


def test_extract_raises_when_budget_exhausted():
    """Every attempt fails -> ExtractionCallFailedError."""
    gen = make_generate_stub(["garbage"] * 3)
    with pytest.raises(ext_mod.ExtractionCallFailedError) as exc_info:
        ext_mod.extract_from_transcript(
            _transcript_payload(),
            generate_fn=gen,
            max_attempts=2,
        )
    assert exc_info.value.attempts == 2
    assert exc_info.value.last_error is not None


def test_extract_raises_when_generate_fn_throws():
    """Every attempt raises -> ExtractionCallFailedError surfaces last."""
    gen = make_generate_stub(
        [_good_extraction_payload()], raise_on={1, 2},
    )
    with pytest.raises(ext_mod.ExtractionCallFailedError) as exc_info:
        ext_mod.extract_from_transcript(
            _transcript_payload(),
            generate_fn=gen,
            max_attempts=2,
        )
    assert isinstance(exc_info.value.last_error, RuntimeError)


# ---------------------------------------------------------------------------
# Prompt builder spot checks
# ---------------------------------------------------------------------------


def test_prompt_builder_embeds_draft_and_brief():
    """Prompt body must surface the final_draft + brief fields verbatim."""
    payload = _transcript_payload(
        final_draft="REN BLACK: This is the visible draft.",
    )
    messages = ext_mod.build_extract_prompt(
        payload,
        cast_names=["REN BLACK", "DR MAEVE COLE"],
        news_seed="some news seed",
    )
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    user = messages[1]["content"]
    assert "This is the visible draft" in user
    assert "1947 humpback aria matched" in user
    assert "REN BLACK" in user
    assert "DR MAEVE COLE" in user
    assert "some news seed" in user
    assert "WRITER'S FINAL DRAFT" in user


def test_prompt_builder_handles_empty_brief():
    """Missing brief in the transcript falls back to placeholder text."""
    messages = ext_mod.build_extract_prompt(
        {"final_draft": "stub"},
        cast_names=None,
        news_seed="",
    )
    user = messages[1]["content"]
    assert "no Director brief supplied" in user
    assert "no Director-locked cast names supplied" in user
    assert "no news seed supplied" in user


# ---------------------------------------------------------------------------
# Source pins
# ---------------------------------------------------------------------------


def test_extract_module_has_slot_tag():
    """The LLM call site must carry the rule-6 technical-slot tag."""
    src = Path(ext_mod.__file__).read_text(encoding="utf-8")
    assert "# LLM slot: technical" in src
    # And the schema must be Pydantic-bound (so the constrained
    # decoder can attach JsonSchemaParser).
    assert "class StoryRoomExtractionSchema(BaseModel)" in src


def test_extract_imports_stage1_types():
    """Section 3.4 contract: extraction reuses Stage1 types."""
    src = Path(ext_mod.__file__).read_text(encoding="utf-8")
    assert (
        "from ._otr_stage1_plan import Stage1Arc, Stage1Beat, "
        "Stage1CastMember"
    ) in src
