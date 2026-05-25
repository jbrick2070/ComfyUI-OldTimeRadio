"""tests/test_otr_continuity.py -- Sprint 5A continuity ledger tests.

The module under test (`nodes/_otr_continuity.py`) is pure -- no I/O,
no GPU, no real model. Every test here runs against a fake `generate_fn`
slot fn (a plain callable matching the project slot-fn contract
`(messages, *, temperature, max_new_tokens) -> str` that returns canned
JSON text) and small in-test outline / cast stubs shaped like the real
`_otr_outline.Outline` / production-ledger cast rows.

Coverage map:

  build_continuity_ledger:
    1. a well-formed JSON response parses into a ContinuityState with
       the expected location / props / facts.
    2. a malformed / garbage response returns ContinuityState.neutral()
       and never raises.
    3. a generate_fn that itself raises is swallowed -> neutral state,
       no exception escapes.
    4. an outline with no beats returns neutral WITHOUT calling the
       slot fn at all.

  render_continuity_slice:
    5. produces a non-empty block carrying the location, the facts the
       speaker KNOWS, and the "must NOT reference" hidden facts.
    6. returns "" for a neutral state.
    7. filters facts by established_beat vs beat_index -- a fact
       established at a later beat is absent from an earlier-beat slice.
    8. matches speakers loosely / case-insensitively.

The fake-generate-fn pattern and the `from nodes import ...` import
convention mirror `tests/test_structured_call.py`.
"""

from __future__ import annotations

import pytest

from nodes import _otr_continuity as cont


# ---------------------------------------------------------------------------
# Fixtures -- outline / cast stubs + canned-response fake slot fns
# ---------------------------------------------------------------------------


def _outline_stub() -> dict:
    """A minimal outline dict shaped like _otr_outline.Outline.

    The continuity module duck-types `outline.beats` and each beat's
    `speaker` / `speaker_role` / `intent` / `mood`, so a plain dict
    with a `"beats"` key exercises the same code path as the real
    pydantic Outline.
    """
    return {
        "title": "The Quiet Frequency",
        "premise": "A radio operator decodes a signal that should not exist.",
        "beats": [
            {"speaker": "MARA", "speaker_role": "character",
             "intent": "Mara tunes the dial and hears the anomaly.",
             "mood": "curious"},
            {"speaker": "DELL", "speaker_role": "character",
             "intent": "Dell dismisses it as interference.",
             "mood": "skeptical"},
            {"speaker": "MARA", "speaker_role": "character",
             "intent": "Mara realizes the signal names her.",
             "mood": "alarmed"},
        ],
    }


def _cast_stub() -> list[dict]:
    """A locked-cast stub shaped like the production-ledger cast rows."""
    return [
        {"name": "MARA", "char_id": "c01",
         "character_description": "A night-shift radio operator."},
        {"name": "DELL", "char_id": "c02",
         "character_description": "A weary station supervisor."},
    ]


def _well_formed_response() -> str:
    """A response string that parses + validates as a ContinuityState."""
    return (
        '{'
        '"location": "A coastal radio relay station, after midnight", '
        '"active_props": ["the receiver dial", "a logbook"], '
        '"facts": ['
        '  {"fact": "The anomalous signal contains Mara\'s name.", '
        '   "known_by": ["MARA"], "hidden_from": ["DELL"], '
        '   "established_beat": 2}'
        ']}'
    )


def _garbage_response() -> str:
    """A response string that is not a decodable JSON object."""
    return "this is not JSON at all {{{ unterminated"


def _make_fake_generate_fn(*, responses: list[str]):
    """Return a fake slot fn matching the project slot-fn contract.

    Signature `(messages, *, temperature, max_new_tokens) -> str`.
    Records every call and pops queued responses; returns "" when the
    queue is drained, which fails JSON parsing -- so an over-run retry
    ladder still fails cleanly rather than raising IndexError.

    The closure name is deliberately descriptive; the project forbids
    the word "dummy" in any stub / fixture name.
    """
    calls: list[dict] = []

    def generate_fn(messages, *, temperature, max_new_tokens):
        calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        })
        if not responses:
            return ""
        return responses.pop(0)

    generate_fn.calls = calls  # type: ignore[attr-defined]
    return generate_fn


def _make_raising_generate_fn():
    """Return a fake slot fn that raises -- simulates a VRAM / loader fault."""
    calls: list[dict] = []

    def generate_fn(messages, *, temperature, max_new_tokens):
        calls.append({"temperature": temperature})
        raise RuntimeError("simulated loader failure inside generate_fn")

    generate_fn.calls = calls  # type: ignore[attr-defined]
    return generate_fn


# ---------------------------------------------------------------------------
# 1. build_continuity_ledger parses a well-formed response
# ---------------------------------------------------------------------------


def test_build_continuity_ledger_parses_well_formed_response():
    generate_fn = _make_fake_generate_fn(responses=[_well_formed_response()])
    state = cont.build_continuity_ledger(
        generate_fn, _outline_stub(), _cast_stub(),
    )
    assert isinstance(state, cont.ContinuityState)
    assert not state.is_neutral()
    assert state.location == "A coastal radio relay station, after midnight"
    assert state.active_props == ["the receiver dial", "a logbook"]
    assert len(state.facts) == 1
    fact = state.facts[0]
    assert isinstance(fact, cont.ContinuityFact)
    assert "Mara" in fact.fact
    assert fact.known_by == ["MARA"]
    assert fact.hidden_from == ["DELL"]
    assert fact.established_beat == 2
    # Attempt 1 succeeded -> exactly one slot call.
    assert len(generate_fn.calls) == 1


def test_build_continuity_ledger_passes_technical_repo_id_without_raising():
    # technical_repo_id is log-attribution only; a value must not change
    # the parsed result or trip any code path.
    generate_fn = _make_fake_generate_fn(responses=[_well_formed_response()])
    state = cont.build_continuity_ledger(
        generate_fn, _outline_stub(), _cast_stub(),
        technical_repo_id="some-org/some-technical-model",
    )
    assert isinstance(state, cont.ContinuityState)
    assert state.location.startswith("A coastal radio relay station")


# ---------------------------------------------------------------------------
# 2. A malformed / garbage response returns neutral and never raises
# ---------------------------------------------------------------------------


def test_build_continuity_ledger_garbage_response_returns_neutral():
    # Every attempt in the 3-rung ladder returns garbage -> the ladder
    # exhausts -> StructuredCallFailedError -> the builder swallows it
    # and returns ContinuityState.neutral(). No exception escapes.
    generate_fn = _make_fake_generate_fn(
        responses=[_garbage_response(), _garbage_response(), _garbage_response()],
    )
    state = cont.build_continuity_ledger(
        generate_fn, _outline_stub(), _cast_stub(),
    )
    assert isinstance(state, cont.ContinuityState)
    assert state.is_neutral()
    assert state == cont.ContinuityState.neutral()
    # The ladder ran all three rungs before giving up.
    assert len(generate_fn.calls) == 3


# ---------------------------------------------------------------------------
# 3. A generate_fn that raises is swallowed -> neutral, never raises
# ---------------------------------------------------------------------------


def test_build_continuity_ledger_raising_generate_fn_returns_neutral():
    generate_fn = _make_raising_generate_fn()
    # Must not raise -- Prime Directive 1 (audio output never breaks).
    state = cont.build_continuity_ledger(
        generate_fn, _outline_stub(), _cast_stub(),
    )
    assert isinstance(state, cont.ContinuityState)
    assert state.is_neutral()
    # structured_call does not catch slot-fn exceptions, so the raise
    # surfaces on the first attempt and the builder's broad except arm
    # catches it -- exactly one call before the failure is swallowed.
    assert len(generate_fn.calls) == 1


# ---------------------------------------------------------------------------
# 4. An outline with no beats returns neutral with no LLM call
# ---------------------------------------------------------------------------


def test_build_continuity_ledger_empty_outline_skips_llm_call():
    generate_fn = _make_fake_generate_fn(responses=[_well_formed_response()])
    state = cont.build_continuity_ledger(
        generate_fn, {"beats": []}, _cast_stub(),
    )
    assert isinstance(state, cont.ContinuityState)
    assert state.is_neutral()
    # No beats -> nothing to extract -> the slot fn is never called.
    assert len(generate_fn.calls) == 0


# ---------------------------------------------------------------------------
# 5. render_continuity_slice produces a populated hard-constraint block
# ---------------------------------------------------------------------------


def _populated_state() -> cont.ContinuityState:
    """A non-neutral ContinuityState used by the render tests."""
    return cont.ContinuityState(
        location="A coastal radio relay station, after midnight",
        active_props=["the receiver dial", "a logbook"],
        facts=[
            cont.ContinuityFact(
                fact="The anomalous signal contains Mara's name.",
                known_by=["MARA"],
                hidden_from=["DELL"],
                established_beat=2,
            ),
            cont.ContinuityFact(
                fact="The station has been understaffed for a week.",
                known_by=["MARA", "DELL"],
                hidden_from=[],
                established_beat=0,
            ),
        ],
    )


def test_render_continuity_slice_non_empty_block_for_speaker():
    state = _populated_state()
    # MARA at beat 2: both facts are established (beats 0 and 2 <= 2)
    # and MARA is known_by on both.
    block = render = cont.render_continuity_slice(state, "MARA", 2)
    assert isinstance(block, str)
    assert block != ""
    assert "CONTINUITY CONSTRAINTS" in block
    # Location surfaces.
    assert "A coastal radio relay station, after midnight" in block
    # Both facts MARA knows are present.
    assert "The anomalous signal contains Mara's name." in block
    assert "The station has been understaffed for a week." in block
    # The KNOWS header names the speaker.
    assert "MARA KNOWS" in block
    assert render is block  # sanity: single return value


def test_render_continuity_slice_emits_must_not_reference_for_hidden_speaker():
    state = _populated_state()
    # DELL at beat 2: DELL is hidden_from on the signal fact and
    # known_by on the understaffed fact.
    block = cont.render_continuity_slice(state, "DELL", 2)
    assert block != ""
    # The hidden fact is rendered under an explicit must-NOT-reference
    # instruction for DELL.
    assert "must NOT reference" in block
    assert "The anomalous signal contains Mara's name." in block
    # DELL still legitimately knows the understaffed fact.
    assert "The station has been understaffed for a week." in block
    assert "DELL KNOWS" in block


# ---------------------------------------------------------------------------
# 6. render_continuity_slice returns "" for a neutral state
# ---------------------------------------------------------------------------


def test_render_continuity_slice_neutral_state_returns_empty_string():
    neutral = cont.ContinuityState.neutral()
    assert neutral.is_neutral()
    assert cont.render_continuity_slice(neutral, "MARA", 0) == ""
    assert cont.render_continuity_slice(neutral, "DELL", 99) == ""


# ---------------------------------------------------------------------------
# 7. render_continuity_slice filters facts by established_beat vs beat_index
# ---------------------------------------------------------------------------


def test_render_continuity_slice_filters_known_facts_by_established_beat():
    state = _populated_state()
    # MARA at beat 0: the signal fact is established_beat=2, so it must
    # NOT appear in MARA's KNOWS list at beat 0. The understaffed fact
    # (established_beat=0) still does.
    block = cont.render_continuity_slice(state, "MARA", 0)
    assert block != ""
    assert "The station has been understaffed for a week." in block
    # The not-yet-established fact is filtered out of the known list.
    assert "The anomalous signal contains Mara's name." not in block


def test_render_continuity_slice_hidden_fact_shown_regardless_of_beat():
    state = _populated_state()
    # DELL at beat 0: the signal fact is hidden_from DELL. A hidden fact
    # is off-limits whether or not it has been established for others,
    # so it must still appear in DELL's must-NOT-reference list at the
    # earliest beat.
    block = cont.render_continuity_slice(state, "DELL", 0)
    assert block != ""
    assert "must NOT reference" in block
    assert "The anomalous signal contains Mara's name." in block


# ---------------------------------------------------------------------------
# 8. render_continuity_slice matches speakers loosely / case-insensitively
# ---------------------------------------------------------------------------


def test_render_continuity_slice_matches_speaker_case_insensitively():
    state = _populated_state()
    # Lower-case speaker still matches the upper-case fact name lists.
    block = cont.render_continuity_slice(state, "mara", 2)
    assert block != ""
    assert "The anomalous signal contains Mara's name." in block


def test_render_continuity_slice_matches_speaker_with_title_prefix():
    # A "DR. MARA"-style speaker loosely matches the bare "MARA" name
    # via word-level containment.
    state = cont.ContinuityState(
        location="A lab",
        active_props=[],
        facts=[
            cont.ContinuityFact(
                fact="The experiment has already failed once.",
                known_by=["MARA"],
                hidden_from=[],
                established_beat=0,
            ),
        ],
    )
    block = cont.render_continuity_slice(state, "DR. MARA", 1)
    assert block != ""
    assert "The experiment has already failed once." in block


def test_render_continuity_slice_unrelated_speaker_gets_facts_dropped():
    # A speaker who is neither known_by nor hidden_from any fact still
    # gets the location / props block (those are episode-wide), but no
    # fact lines.
    state = _populated_state()
    block = cont.render_continuity_slice(state, "STRANGER", 2)
    # Location + props are episode-wide, so the block is non-empty.
    assert block != ""
    assert "A coastal radio relay station, after midnight" in block
    # No fact this stranger knows or is hidden from -> no fact lines.
    assert "KNOWS" not in block
    assert "must NOT reference" not in block
