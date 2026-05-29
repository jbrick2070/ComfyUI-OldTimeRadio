r"""Build 4 (draft) tests -- compose_exchange + repair-by-exchange-group.

Covers:
  * parse_exchange enforces one block per slot id (count + id match):
    happy path, missing slot, extra slot, duplicate slot, empty block,
    internal pauses allowed.
  * repair-by-group triggers ONCE and only once when a slot fails Tier-A.
  * fail-loud after one failed repair (status="fail", caller falls back).
  * the concrete-grounding instruction is present in the built prompt.
  * the soft forbidden-word nudge is present but NOT enforced.

No torch, no live model: generate_fn and tier_a_check are fakes.

This draft module lives outside the auto-imported node package, so we
load it by file path. Run from anywhere:

    pytest docs/sprint_drafts/build4/test_compose_exchange.py -v
"""

from __future__ import annotations

import pytest

# Promoted (Build 4, 2026-05-28): the module now lives in the nodes/
# package, so import it the normal way (the staged file-path loader block
# was removed on promotion per build4/WIRING_SPEC.md).
from nodes import _otr_compose_exchange as ce


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------


def _group(n=3):
    """A 2-3 slot beat group."""
    slots = [
        ce.VoicedSlot(dialogue_slot_id="d001", speaker="MARLOW",
                      intent="resist signing the contract", length_target_words=20),
        ce.VoicedSlot(dialogue_slot_id="d002", speaker="REESE",
                      intent="apply pressure without naming the stakes",
                      length_target_words=18),
        ce.VoicedSlot(dialogue_slot_id="d003", speaker="MARLOW",
                      intent="reveal pressure through an object", length_target_words=22),
    ]
    return slots[:n]


def _contracts():
    return {
        "d001": ce.SlotContract(
            dialogue_slot_id="d001", speaker="MARLOW",
            line_job="refuse without explaining why",
            hidden_pressure="he already signed something he regrets",
            concrete_detail_required=["the fountain pen", "the ledger"],
            state_before="guarded", state_after="cornered", must_turn=False,
        ),
        "d002": ce.SlotContract(
            dialogue_slot_id="d002", speaker="REESE",
            line_job="press the question sideways",
            hidden_pressure="needs the signature by midnight",
            concrete_detail_required=["the ledger"],
            state_before="calm", state_after="impatient", must_turn=True,
        ),
        "d003": ce.SlotContract(
            dialogue_slot_id="d003", speaker="MARLOW",
            line_job="deflect through action",
            hidden_pressure="buying time",
            concrete_detail_required=["the fountain pen"],
            state_before="cornered", state_after="defiant", must_turn=False,
        ),
    }


def _cast():
    class _C:
        def __init__(self, name, persona):
            self.name = name
            self.persona = persona
    return [
        _C("MARLOW", "Tired investigator. Clipped sentences. Trusts no one."),
        _C("REESE", "Smooth fixer. Never raises his voice. Always closing."),
    ]


def _raw_for(slots):
    """Well-formed raw output covering exactly the given slots."""
    lines = {
        "d001": "d001|MARLOW: I don't put my name on things I can't read in the dark.",
        "d002": "d002|REESE: Funny -- the dark is exactly when most men sign.",
        "d003": "d003|MARLOW: He turns the pen over once... then sets it down, cap on.",
    }
    return "\n".join(lines[s.dialogue_slot_id] for s in slots)


def _always_ok(parsed, slots):
    return ce.TierAResult(ok=True)


def _always_fail(parsed, slots):
    return ce.TierAResult(
        ok=False,
        reasons=["slot d002 below word floor"],
        failing_slot_ids=["d002"],
    )


class _CountingGen:
    """Fake generate_fn that records every call and returns a scripted
    sequence of raw outputs (one per call)."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def __call__(self, messages, *, temperature, max_new_tokens):
        self.calls.append({
            "messages": messages,
            "temperature": temperature,
            "max_new_tokens": max_new_tokens,
        })
        if self._outputs:
            return self._outputs.pop(0)
        return ""


# ---------------------------------------------------------------------------
# parse_exchange -- one block per slot
# ---------------------------------------------------------------------------


def test_parse_happy_path_three_slots():
    slots = _group(3)
    raw = _raw_for(slots)
    parsed = ce.parse_exchange(raw, [s.dialogue_slot_id for s in slots])
    assert list(parsed.keys()) == ["d001", "d002", "d003"]
    assert parsed["d001"].startswith("I don't put my name")


def test_parse_preserves_expected_order_even_if_raw_reordered():
    slots = _group(3)
    raw = "\n".join([
        "d003|MARLOW: He sets the pen down.",
        "d001|MARLOW: I don't sign in the dark.",
        "d002|REESE: The dark is when men sign.",
    ])
    parsed = ce.parse_exchange(raw, ["d001", "d002", "d003"])
    assert list(parsed.keys()) == ["d001", "d002", "d003"]


def test_parse_allows_internal_pauses_in_a_block():
    parsed = ce.parse_exchange(
        "d001|MARLOW: I... I don't sign in the dark -- not anymore.",
        ["d001"],
    )
    assert "..." in parsed["d001"]


def test_parse_rejects_missing_slot():
    with pytest.raises(ce.ExchangeParseError) as exc:
        ce.parse_exchange("d001|MARLOW: line one.", ["d001", "d002"])
    assert "missing" in str(exc.value)


def test_parse_rejects_extra_slot():
    raw = "\n".join([
        "d001|MARLOW: one.",
        "d002|REESE: two.",
        "d099|GHOST: uninvited.",
    ])
    with pytest.raises(ce.ExchangeParseError) as exc:
        ce.parse_exchange(raw, ["d001", "d002"])
    assert "extra" in str(exc.value)


def test_parse_rejects_duplicate_slot():
    raw = "\n".join([
        "d001|MARLOW: one.",
        "d001|MARLOW: again.",
    ])
    with pytest.raises(ce.ExchangeParseError) as exc:
        ce.parse_exchange(raw, ["d001"])
    assert "more than once" in str(exc.value)


def test_parse_rejects_empty_block():
    # A line with the prefix but no text must not satisfy the slot.
    with pytest.raises(ce.ExchangeParseError):
        ce.parse_exchange("d001|MARLOW:    ", ["d001"])


def test_parse_rejects_duplicate_expected_ids():
    with pytest.raises(ce.ExchangeParseError):
        ce.parse_exchange("d001|MARLOW: x.", ["d001", "d001"])


# ---------------------------------------------------------------------------
# compose_exchange -- success on first attempt
# ---------------------------------------------------------------------------


def test_compose_ok_first_attempt_no_repair():
    slots = _group(3)
    gen = _CountingGen([_raw_for(slots)])
    res = ce.compose_exchange(
        slots, _contracts(), [], _cast(),
        generate_fn=gen, tier_a_check=_always_ok,
    )
    assert res.status == "ok"
    assert res.repaired is False
    assert res.attempts == 1
    assert len(gen.calls) == 1
    assert set(res.lines) == {"d001", "d002", "d003"}


# ---------------------------------------------------------------------------
# Repair-by-group: triggers ONCE and only once
# ---------------------------------------------------------------------------


def test_repair_triggers_once_then_succeeds():
    slots = _group(3)
    gen = _CountingGen([_raw_for(slots), _raw_for(slots)])

    # Fail the first verdict, pass the second. Stateful fake.
    state = {"calls": 0}

    def tier_a(parsed, group):
        state["calls"] += 1
        if state["calls"] == 1:
            return ce.TierAResult(ok=False, reasons=["d002 below word floor"],
                                  failing_slot_ids=["d002"])
        return ce.TierAResult(ok=True)

    res = ce.compose_exchange(
        slots, _contracts(), [], _cast(),
        generate_fn=gen, tier_a_check=tier_a,
    )
    assert res.status == "ok_repaired"
    assert res.repaired is True
    # Exactly two generate calls: one original + one repair. No churn.
    assert res.attempts == 2
    assert len(gen.calls) == 2


def test_repair_pass_prompt_includes_failure_reasons():
    slots = _group(2)
    gen = _CountingGen([_raw_for(slots), _raw_for(slots)])

    state = {"calls": 0}

    def tier_a(parsed, group):
        state["calls"] += 1
        if state["calls"] == 1:
            return ce.TierAResult(ok=False, reasons=["UNIQUE_FAILURE_TOKEN_42"])
        return ce.TierAResult(ok=True)

    ce.compose_exchange(
        slots, _contracts(), [], _cast(),
        generate_fn=gen, tier_a_check=tier_a,
    )
    # The second (repair) prompt must carry the failure reason; the first
    # must not.
    first_user = gen.calls[0]["messages"][-1]["content"]
    repair_user = gen.calls[1]["messages"][-1]["content"]
    assert "UNIQUE_FAILURE_TOKEN_42" not in first_user
    assert "UNIQUE_FAILURE_TOKEN_42" in repair_user


# ---------------------------------------------------------------------------
# Fail-loud after one failed repair
# ---------------------------------------------------------------------------


def test_fail_loud_after_one_failed_repair():
    slots = _group(3)
    gen = _CountingGen([_raw_for(slots), _raw_for(slots)])
    res = ce.compose_exchange(
        slots, _contracts(), [], _cast(),
        generate_fn=gen, tier_a_check=_always_fail,
    )
    assert res.status == "fail"
    assert res.repaired is True
    # Original + exactly one repair, then stop. No third attempt.
    assert res.attempts == 2
    assert len(gen.calls) == 2
    assert res.fail_reason and "after one repair" in res.fail_reason


def test_fail_loud_on_unparseable_then_unparseable():
    slots = _group(2)
    gen = _CountingGen(["garbage with no slot prefixes", "still garbage"])
    res = ce.compose_exchange(
        slots, _contracts(), [], _cast(),
        generate_fn=gen, tier_a_check=_always_ok,
    )
    assert res.status == "fail"
    assert res.attempts == 2
    assert len(gen.calls) == 2


def test_parse_failure_then_repair_succeeds():
    slots = _group(2)
    # First raw is unparseable (missing d002), repair is clean.
    bad = "d001|MARLOW: only one slot here."
    good = _raw_for(slots)
    gen = _CountingGen([bad, good])
    res = ce.compose_exchange(
        slots, _contracts(), [], _cast(),
        generate_fn=gen, tier_a_check=_always_ok,
    )
    assert res.status == "ok_repaired"
    assert res.attempts == 2


# ---------------------------------------------------------------------------
# Prompt content: concrete grounding + soft nudge
# ---------------------------------------------------------------------------


def test_prompt_has_one_grounding_per_exchange_instruction():
    slots = _group(3)
    messages = ce.build_exchange_prompt(slots, _contracts(), [], _cast())
    blob = "\n".join(m["content"] for m in messages)
    # "one concrete detail ... not one per line" -- per-exchange, not per-line.
    assert "ONE concrete detail" in blob
    assert "not one per line" in blob
    # Grounding candidates from the Build 3 contracts surface in the prompt.
    assert "the fountain pen" in blob or "the ledger" in blob


def test_prompt_lists_forbidden_words_as_soft_nudge_only():
    slots = _group(2)
    messages = ce.build_exchange_prompt(slots, _contracts(), [], _cast())
    blob = "\n".join(m["content"] for m in messages)
    # Words present as guidance...
    assert "intriguing" in blob
    assert "anomaly" in blob
    # ...explicitly framed as not a hard rule.
    assert "not a hard rule" in blob.lower()


def test_prompt_preserves_slot_ids_and_speakers():
    slots = _group(3)
    messages = ce.build_exchange_prompt(slots, _contracts(), [], _cast())
    blob = "\n".join(m["content"] for m in messages)
    for sid in ("d001", "d002", "d003"):
        assert sid in blob
    assert "MARLOW" in blob and "REESE" in blob


def test_prompt_includes_prior_committed_lines():
    slots = _group(2)
    prior = ["d000|ANNOUNCER: The clock reads eleven.",
             "d001|REESE: You're late, Marlow."]
    messages = ce.build_exchange_prompt(slots, _contracts(), prior, _cast())
    blob = "\n".join(m["content"] for m in messages)
    assert "You're late, Marlow." in blob


def test_must_turn_instruction_present_for_turning_slot():
    slots = _group(2)  # d002 has must_turn=True in _contracts()
    messages = ce.build_exchange_prompt(slots, _contracts(), [], _cast())
    blob = "\n".join(m["content"] for m in messages)
    assert "TURN the scene" in blob


# ---------------------------------------------------------------------------
# Edge: empty group
# ---------------------------------------------------------------------------


def test_empty_group_fails_without_generate_call():
    gen = _CountingGen([])
    res = ce.compose_exchange(
        [], _contracts(), [], _cast(),
        generate_fn=gen, tier_a_check=_always_ok,
    )
    assert res.status == "fail"
    assert len(gen.calls) == 0


# ---------------------------------------------------------------------------
# Build 4 wiring seams: group_voiced_beats + make_tier_a_adapter
# ---------------------------------------------------------------------------


def _vs(sid, speaker):
    return ce.VoicedSlot(dialogue_slot_id=sid, speaker=speaker)


def test_group_voiced_beats_run_of_four_is_two_two():
    slots = [_vs(f"d00{i}", "REN") for i in range(1, 5)]
    assert [len(g) for g in ce.group_voiced_beats(slots)] == [2, 2]


def test_group_voiced_beats_run_of_five_is_three_two():
    slots = [_vs(f"d00{i}", "REN") for i in range(1, 6)]
    assert [len(g) for g in ce.group_voiced_beats(slots)] == [3, 2]


def test_group_voiced_beats_run_of_seven_has_no_trailing_singleton():
    slots = [_vs(f"d00{i}", "REN") for i in range(1, 8)]
    sizes = [len(g) for g in ce.group_voiced_beats(slots)]
    assert sizes == [3, 2, 2]
    assert 1 not in sizes


def test_group_voiced_beats_announcer_is_singleton_and_breaks_group():
    slots = [
        _vs("d001", "ANNOUNCER"),
        _vs("d002", "REN"),
        _vs("d003", "MAEVE"),
        _vs("d004", "ANNOUNCER"),
    ]
    groups = ce.group_voiced_beats(slots)
    assert [[s.dialogue_slot_id for s in g] for g in groups] == [
        ["d001"], ["d002", "d003"], ["d004"],
    ]


def test_group_voiced_beats_trailing_single_voiced_is_singleton():
    slots = [_vs("d001", "ANNOUNCER"), _vs("d002", "REN")]
    assert [len(g) for g in ce.group_voiced_beats(slots)] == [1, 1]


def test_group_voiced_beats_empty_speaker_is_singleton():
    slots = [_vs("d001", ""), _vs("d002", "REN"), _vs("d003", "MAEVE")]
    groups = ce.group_voiced_beats(slots)
    assert [s.dialogue_slot_id for s in groups[0]] == ["d001"]
    assert [s.dialogue_slot_id for s in groups[1]] == ["d002", "d003"]


def test_tier_a_adapter_passes_clean_exchange_against_live_build2():
    # Real Build 2 functions -- proves the Build 4 -> Build 2 seam.
    from nodes._otr_craft_floor import evaluate_tier_a, normalize_slot_line

    adapter = ce.make_tier_a_adapter(evaluate_tier_a, normalize_slot_line)
    slots = [_vs("d001", "MARLOW"), _vs("d002", "REN BLACK")]
    parsed = {
        "d001": "I will not sign that paper tonight.",
        "d002": "Then we both lose everything we built.",
    }
    res = adapter(parsed, slots)
    assert res.ok is True
    assert res.reasons == []


def test_tier_a_adapter_flags_below_word_floor_against_live_build2():
    from nodes._otr_craft_floor import evaluate_tier_a, normalize_slot_line

    adapter = ce.make_tier_a_adapter(evaluate_tier_a, normalize_slot_line)
    slots = [_vs("d001", "MARLOW"), _vs("d002", "REN BLACK")]
    parsed = {
        "d001": "No.",  # one word -- below the floor of 4
        "d002": "Then we both lose everything we built.",
    }
    res = adapter(parsed, slots)
    assert res.ok is False
    assert "d001" in res.failing_slot_ids
