"""Sprint 10A step 6 regression -- Stage 2 multi-turn roleplay.

Covers both modules:
  * nodes/_otr_stage2_prompt.py
      - build_stage2_prompt_chain produces the 4-turn message list
        with the expected shape, the assistant priming turns, and
        the Turn 4 spec.
      - accept_turn1_ack / accept_turn23_ack accept the documented
        variants and reject empty / unrelated replies.

  * nodes/_otr_stage2_call.py
      - compose_line: best-of-N over a mocked generate_fn, picks
        the highest-scoring candidate, returns Stage2LineRecord.
      - score_candidate: errors penalize harder than warns; length-
        closeness bonus kicks in at on-target lines.
      - Mode switch: roleplay_multiturn vs roleplay_single_turn
        both run end-to-end against a mock.
      - Reserved speaker (ANNOUNCER) returns a fallback record.
      - All-empty / all-raising candidates produce a fallback
        record without crashing the caller.

Tests do NOT load a real LLM; generate_fn is mocked with canned
strings. The actual model exercises happen in soak after step 6-C
wires this into the writer.
"""

from __future__ import annotations

import pytest

from nodes._otr_stage1_plan import (
    Stage1Arc,
    Stage1Beat,
    Stage1CastMember,
    Stage1Plan,
)
from nodes._otr_stage2_prompt import (
    TURN1_EXPECTED,
    TURN23_EXPECTED,
    accept_turn1_ack,
    accept_turn23_ack,
    assemble_full_chat,
    build_stage2_prompt_chain,
)
from nodes._otr_stage2_call import (
    DEFAULT_BEST_OF_N,
    Stage2CandidateRecord,
    Stage2LineRecord,
    build_single_turn_prompt,
    compose_line,
    score_candidate,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _plan() -> Stage1Plan:
    return Stage1Plan(
        premise="Two technicians at a Mars relay catch a signal that should not exist.",
        arc=Stage1Arc(
            setup="Routine night shift, calm comms traffic.",
            complication="An off-band carrier locks in and refuses to drop.",
            resolution="They identify the source and decide whether to acknowledge it.",
        ),
        cast=[
            Stage1CastMember(
                name="DALE PORTER",
                gender="male",
                pronouns="he/him",
                voice_id="v2/en_speaker_5",
                persona="Experienced relay technician. Measured cadences; trusts procedure.",
                arc_role="anchor of competence",
            ),
            Stage1CastMember(
                name="MEG SAARI",
                gender="female",
                pronouns="she/her",
                voice_id="v2/en_speaker_2",
                persona="Junior tech, six months in, quick to question.",
                arc_role="pressure source",
            ),
        ],
        beats=[
            Stage1Beat(
                beat_id="b001", speaker="ANNOUNCER",
                intent="Open the episode.", length_target_words=18,
                emotional_register="neutral", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b002", speaker="DALE PORTER",
                intent="Notice the off-band carrier and call it out.",
                length_target_words=24,
                emotional_register="alert calm", callback_to=None,
            ),
            Stage1Beat(
                beat_id="b003", speaker="MEG SAARI",
                intent="Ask Dale what the carrier means.",
                length_target_words=22,
                emotional_register="alert curious", callback_to="b002",
            ),
        ],
        running_facts=["The relay is on Mars.", "Dale is senior to Meg."],
    )


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class TestPromptChainShape:
    def test_priming_messages_length_seven(self):
        """BUG-LOCAL-280 (2026-05-27): chain length bumped 6 -> 7 with
        the addition of the leading user turn that asks for the Ready
        ack. Mistral-Nemo's chat template requires user/assistant
        alternation after the optional system message; the pre-fix
        system -> assistant directly raised TemplateError on every
        generate. The leading user turn restores alternation."""
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        # system + 3 user + 3 assistant primes = 7 messages
        # post-BUG-LOCAL-280.
        assert len(chain.priming_messages) == 7

    def test_priming_alternates_correctly(self):
        """BUG-LOCAL-280: alternation is now
        system -> user -> assistant -> user -> assistant -> user ->
        assistant, which Mistral-Nemo's chat template renders cleanly.
        """
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        expected_roles = [
            "system",
            "user", "assistant",
            "user", "assistant",
            "user", "assistant",
        ]
        got_roles = [m["role"] for m in chain.priming_messages]
        assert got_roles == expected_roles

    def test_turn1_system_carries_speaker_persona(self):
        """BUG-LOCAL-280: the system prompt no longer carries the
        'Reply Ready' instruction (that moved to the leading user
        turn at index 1). System content stays focused on the
        role-bind + persona."""
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        turn1_system = chain.priming_messages[0]
        assert turn1_system["role"] == "system"
        assert "DALE PORTER" in turn1_system["content"]
        assert "measured cadences" in turn1_system["content"].lower()

    def test_turn1_user_asks_for_ready_ack(self):
        """BUG-LOCAL-280 new: the leading user turn (index 1) carries
        the Ready ack instruction that used to be in the system
        prompt. This restores user/assistant alternation."""
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        turn1_user = chain.priming_messages[1]
        assert turn1_user["role"] == "user"
        assert TURN1_EXPECTED in turn1_user["content"].lower()

    def test_turn2_user_carries_arc_and_premise(self):
        """BUG-LOCAL-280: turn2 user moved from index 2 -> index 3
        after the leading user-ack insertion at index 1."""
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        turn2 = chain.priming_messages[3]
        assert turn2["role"] == "user"
        assert "Mars relay" in turn2["content"]
        # Arc setup phrase
        assert "Routine night shift" in turn2["content"]

    def test_turn3_user_carries_facts_and_pronouns(self):
        """BUG-LOCAL-280: turn3 user moved from index 4 -> index 5
        after the leading user-ack insertion at index 1."""
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        turn3 = chain.priming_messages[5]
        assert turn3["role"] == "user"
        assert "Mars" in turn3["content"]   # from running_facts
        assert "he/him" in turn3["content"]   # speaker pronouns
        assert "anchor of competence" in turn3["content"]

    def test_turn4_user_carries_beat_intent_length_register(self):
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        turn4 = chain.turn4_user
        assert turn4["role"] == "user"
        assert "off-band carrier" in turn4["content"]
        assert "24" in turn4["content"]   # length target
        assert "alert calm" in turn4["content"]
        # Must instruct no stage directions
        assert "no stage directions" in turn4["content"].lower()

    def test_turn4_includes_previous_line_when_supplied(self):
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
            previous_speaker="ANNOUNCER",
            previous_line="Signal Lost, episode three.",
        )
        assert "ANNOUNCER" in chain.turn4_user["content"]
        assert "Signal Lost, episode three" in chain.turn4_user["content"]

    def test_turn4_omits_previous_clause_when_no_prior_line(self):
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        assert "Previous line" not in chain.turn4_user["content"]

    def test_assemble_full_chat_appends_turn4(self):
        """BUG-LOCAL-280: full chat length bumped 7 -> 8 with the
        leading user-ack turn added to the priming chain."""
        plan = _plan()
        chain = build_stage2_prompt_chain(
            speaker=plan.cast[0],
            beat=plan.beats[1],
            arc=plan.arc,
            premise=plan.premise,
            running_facts=plan.running_facts,
        )
        full = assemble_full_chat(chain)
        assert len(full) == 8  # 7 priming + Turn 4 (post-BUG-LOCAL-280)
        assert full[-1] == chain.turn4_user


# ---------------------------------------------------------------------------
# Acknowledgment-chain validators
# ---------------------------------------------------------------------------


class TestAckValidators:
    def test_turn1_ready_literal(self):
        assert accept_turn1_ack("Ready") is True
        assert accept_turn1_ack("ready.") is True
        assert accept_turn1_ack("I am ready") is True

    def test_turn1_close_variants_accepted(self):
        assert accept_turn1_ack("Got it") is True
        assert accept_turn1_ack("Okay") is True
        assert accept_turn1_ack("Understood.") is True

    def test_turn1_empty_rejected(self):
        assert accept_turn1_ack("") is False
        assert accept_turn1_ack("   ") is False
        assert accept_turn1_ack(None) is False  # type: ignore[arg-type]

    def test_turn1_unrelated_rejected(self):
        # "Hello world" matches none of the patterns.
        assert accept_turn1_ack("Hello world") is False

    def test_turn23_confirmed_literal(self):
        assert accept_turn23_ack("Confirmed.") is True
        assert accept_turn23_ack("confirm") is True

    def test_turn23_close_variants_accepted(self):
        assert accept_turn23_ack("Noted.") is True
        assert accept_turn23_ack("Agreed.") is True
        assert accept_turn23_ack("Got it.") is True

    def test_turn23_empty_rejected(self):
        assert accept_turn23_ack("") is False
        assert accept_turn23_ack(None) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# score_candidate
# ---------------------------------------------------------------------------


class TestScoreCandidate:
    def test_clean_line_scores_high(self):
        plan = _plan()
        beat = plan.beats[1]  # length target 24
        text = " ".join(["word"] * 24)   # exact length
        score, val = score_candidate(plan, beat, text)
        # Base 100 - 0 errors - 0 warns + length bonus -> ~105.
        # But word-soup is off-beat; that's a WARN (cost 1.0).
        # So expected ~104.
        assert score > 90.0
        assert val.ok or all(f.severity == "warn" for f in val.findings)

    def test_error_penalty_dominates(self):
        plan = _plan()
        beat = plan.beats[1]
        # Way too short -> length_drift ERROR + off-beat WARN.
        score, val = score_candidate(plan, beat, "yo")
        assert score < 95.0
        assert any(f.severity == "error" for f in val.findings)

    def test_length_bonus_falls_off(self):
        plan = _plan()
        beat = plan.beats[1]
        # Target 24, length 50 -- length_drift ERROR + bonus 0.
        far_score, _ = score_candidate(plan, beat, " ".join(["w"] * 50))
        # Target 24, length 24 -- no length error, max bonus.
        on_score, _ = score_candidate(plan, beat, " ".join(["w"] * 24))
        assert on_score > far_score


# ---------------------------------------------------------------------------
# compose_line best-of-N orchestration
# ---------------------------------------------------------------------------


class TestComposeLineBestOfN:
    def test_picks_best_scoring_candidate(self):
        plan = _plan()
        beat = plan.beats[1]
        # 4 candidates: 3 broken, 1 clean.
        canned = [
            "",                                                 # empty -> -inf
            "Forfeit to the void *sighs*",                       # banned + leak
            "She braced against the bulkhead.",                  # pronoun mismatch
            "The carrier is off-band locked into channel four at oh-three-twenty eastern band edge.",  # clean-ish
        ]
        calls = {"n": 0}

        def fake_generate(messages, *, temperature, max_new_tokens):
            i = calls["n"]
            calls["n"] += 1
            return canned[i]

        record = compose_line(
            plan=plan,
            beat=beat,
            generate_fn=fake_generate,
            mode="roleplay_multiturn",
            n=4,
        )
        assert record.chosen_index >= 0
        assert "carrier" in record.chosen_text
        assert len(record.candidates) == 4

    def test_all_empty_returns_fallback_record(self):
        plan = _plan()
        beat = plan.beats[1]

        def fake_generate(messages, *, temperature, max_new_tokens):
            return ""

        record = compose_line(
            plan=plan, beat=beat, generate_fn=fake_generate, n=3,
        )
        assert record.chosen_index == -1
        assert record.chosen_text == ""
        assert record.fallback_reason is not None
        assert "empty" in record.fallback_reason.lower() or "raised" in record.fallback_reason.lower()

    def test_all_raising_returns_fallback(self):
        plan = _plan()
        beat = plan.beats[1]

        def fake_generate(messages, *, temperature, max_new_tokens):
            raise RuntimeError("boom")

        record = compose_line(
            plan=plan, beat=beat, generate_fn=fake_generate, n=2,
        )
        assert record.chosen_index == -1
        assert record.chosen_text == ""
        # 2 candidate records were still appended (each with -inf score).
        assert len(record.candidates) == 2

    def test_reserved_speaker_returns_fallback_record(self):
        plan = _plan()
        announcer_beat = plan.beats[0]   # speaker='ANNOUNCER'

        def fake_generate(messages, *, temperature, max_new_tokens):
            return "irrelevant"

        record = compose_line(
            plan=plan, beat=announcer_beat, generate_fn=fake_generate, n=2,
        )
        assert record.chosen_index == -1
        assert record.fallback_reason is not None
        assert "reserved" in record.fallback_reason.lower()
        # No candidates ran (we bailed before the N-loop).
        assert len(record.candidates) == 0

    def test_invalid_mode_raises(self):
        plan = _plan()
        beat = plan.beats[1]

        def fake_generate(messages, *, temperature, max_new_tokens):
            return "clean"

        with pytest.raises(ValueError, match="mode"):
            compose_line(
                plan=plan, beat=beat, generate_fn=fake_generate,
                mode="bogus_mode",   # type: ignore[arg-type]
                n=1,
            )


# ---------------------------------------------------------------------------
# Single-turn mode
# ---------------------------------------------------------------------------


class TestSingleTurnMode:
    def test_single_turn_prompt_is_one_user_message(self):
        plan = _plan()
        beat = plan.beats[1]
        messages = build_single_turn_prompt(
            plan=plan,
            beat=beat,
            speaker_name="DALE PORTER",
        )
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        # Content carries the same fundamentals as Turn 4 of A2:
        # premise + arc + facts + beat + length + register.
        content = messages[0]["content"]
        assert "Mars relay" in content
        assert "off-band carrier" in content
        assert "he/him" in content
        assert "24" in content
        assert "alert calm" in content

    def test_single_turn_rejects_reserved_speaker(self):
        plan = _plan()
        beat = plan.beats[0]   # ANNOUNCER
        with pytest.raises(ValueError, match="reserved"):
            build_single_turn_prompt(
                plan=plan,
                beat=beat,
                speaker_name=beat.speaker,
            )

    def test_compose_line_in_single_turn_mode_works(self):
        plan = _plan()
        beat = plan.beats[1]

        def fake_generate(messages, *, temperature, max_new_tokens):
            return "The carrier locked in at oh-three-twenty -- off-band, channel four, eastern edge of the comms band."

        record = compose_line(
            plan=plan,
            beat=beat,
            generate_fn=fake_generate,
            mode="roleplay_single_turn",
            n=2,
        )
        assert record.mode == "roleplay_single_turn"
        assert record.chosen_index >= 0
        assert "carrier" in record.chosen_text


# ---------------------------------------------------------------------------
# Stage2 records shape
# ---------------------------------------------------------------------------


class TestRecordShape:
    def test_default_best_of_n_is_4(self):
        assert DEFAULT_BEST_OF_N == 4

    def test_candidate_record_default_counts(self):
        from nodes._otr_stage3_validators import ValidationResult
        rec = Stage2CandidateRecord(
            index=0, text="hi", score=10.0,
            elapsed_seconds=0.1, validation=ValidationResult(),
        )
        assert rec.error_count == 0
        assert rec.warn_count == 0

    def test_line_record_carries_metadata(self):
        rec = Stage2LineRecord(
            beat_id="b002", speaker="DALE PORTER",
            mode="roleplay_multiturn",
        )
        assert rec.beat_id == "b002"
        assert rec.candidates == []
        assert rec.chosen_index == -1
