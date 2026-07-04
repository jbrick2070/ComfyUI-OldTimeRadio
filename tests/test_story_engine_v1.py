"""tests/test_story_engine_v1.py -- story-engine v1 sprint (F1..F8).

Per-feature unit coverage for the content-only story-quality fixes. Pure
Python, no GPU, no LLM (mock generate_fns only). One file, grouped by feature.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    LineRequest,
    _build_user_prompt,
    _gender_to_pronouns,
    build_voice_card,
    compose_announcer_outro,
    compose_line_draft,
)
from nodes._otr_casting import CastingResponse, DescriptionResponse  # noqa: E402
from nodes._otr_line_hygiene import detect_narration_self_address  # noqa: E402
from nodes._otr_dramatic_state_llm import (  # noqa: E402
    ARC_SHAPES,
    DramaticStateLLM,
    _make_post_validator,
    _pick_templates,
    derive_news_dramatic_state,
    pick_arc_shape,
)
from nodes._otr_dramatic_state import pick_costly_choice_slot  # noqa: E402
from nodes._otr_slot_drama_contract import (  # noqa: E402
    SlotDramaContract,
    derive_contract_skeleton,
    validate_episode_contracts,
)


def _req(**over):
    base = dict(
        speaker="ALICE",
        intent="reveal the signal",
        mood="tense",
        target_words=15,
        canon_header="TITLE: x\nSETTING: y\nTIME: z\nPREMISE: w",
        last_lines=[("BOB", "What did you find?")],
    )
    base.update(over)
    return LineRequest(**base)


# ===========================================================================
# F1 -- length tail no longer hard-caps every line at 20-30 words
# ===========================================================================

class TestF1LengthTail:

    def test_tail_drops_literal_word_cap(self):
        prompt = _build_user_prompt(_req(target_words=120))
        assert "20-30 words" not in prompt
        assert "about 20-30" not in prompt

    def test_tail_keeps_spoken_cadence_rider(self):
        prompt = _build_user_prompt(_req())
        assert "spoken-length -- one breath" in prompt
        assert "Ground this line in the news facts" in prompt

    def test_word_count_target_still_present(self):
        # the per-line target is still communicated via WRITE LINE
        prompt = _build_user_prompt(_req(target_words=42))
        assert "Word count target: 42." in prompt

    def test_token_budget_scales_to_beat_target(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # small beat target -> attempt-1 budget scales (15*4=60 < cap 200)
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=15))
        assert captured["mnt"] == 60

    def test_token_budget_capped_for_long_beat(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # large beat target -> capped at 200 (864*4 would be 3456)
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=864))
        assert captured["mnt"] == 200

    def test_token_budget_zero_target_uses_full_cap(self):
        captured = {}

        def mock_fn(messages, *, temperature, max_new_tokens, **kw):
            captured.setdefault("mnt", max_new_tokens)
            return "ALICE: I found a signal buried in the noise."

        # zero/falsy target -> full cap, never the 40 floor starving it
        compose_line_draft(creative_fn=mock_fn, req=_req(target_words=0))
        assert captured["mnt"] == 200


# ===========================================================================
# F6 -- split rider: indirect-performance unconditional, situation-change gated
# ===========================================================================

class TestF6SplitRider:

    def test_indirect_rider_unconditional_on_plain_beat(self):
        # no dramatic fields at all -> the indirect rider is STILL present
        prompt = _build_user_prompt(_req())
        assert "Do not summarize the objective" in prompt
        assert "Perform the objective indirectly" in prompt

    def test_situation_change_absent_on_plain_beat(self):
        prompt = _build_user_prompt(_req())
        assert "The situation must be different after this line." not in prompt

    def test_situation_change_present_on_turn_beat(self):
        prompt = _build_user_prompt(_req(beat_turn="she names the omission aloud"))
        assert "Perform the objective indirectly" in prompt
        assert "The situation must be different after this line." in prompt

    def test_rider_lands_before_speak_now(self):
        prompt = _build_user_prompt(_req(beat_turn="the lie collapses"))
        assert prompt.index("Perform the objective indirectly") < prompt.index("Speak now.")


# ===========================================================================
# F2 -- costly-choice binding: must_turn only on a character slot, audit-safe
# ===========================================================================

_DS = {
    "dramatic_question": "Will the vial be opened before the audit closes?",
    "character_a_wants": "broadcast the strain to the whole network",
    "character_b_wants": "keep the vial sealed and unrecorded",
    "ending_change": "the vial is opened and the record cannot be undone",
}


def _skeleton(slot_id, speaker, slot_index, costly):
    ds = dict(_DS)
    ds["costly_choice_beat"] = costly
    return derive_contract_skeleton(
        slot_row={"dialogue_slot_id": slot_id, "speaker": speaker},
        slot_index=slot_index,
        dramatic_state=ds,
        active_props=[],
        key_terms=["the vial", "the audit"],
    )


def _full_contract(slot_id, speaker, slot_index, costly):
    sk = _skeleton(slot_id, speaker, slot_index, costly)
    sk["line_job"] = "press the point without naming it"
    sk["hidden_pressure"] = "the audit clock is running out"
    return SlotDramaContract.model_validate(sk)


class TestF2CostlyBinding:

    def test_pick_costly_from_character_only_returns_character_slot(self):
        char_slots = ["d002", "d004", "d006"]
        assert pick_costly_choice_slot(char_slots) in char_slots

    def test_must_turn_lands_on_costly_character_slot(self):
        # costly = d004 (a character slot) -> only d004 gets must_turn
        s2 = _skeleton("d002", "EDNA", 0, "d004")
        s4 = _skeleton("d004", "PETER", 1, "d004")
        s5 = _skeleton("d005", "ANNOUNCER", 2, "d004")
        assert s2["must_turn"] is False
        assert s4["must_turn"] is True
        assert s5["must_turn"] is False

    def test_cleared_costly_yields_no_turn_and_invalid_audit(self):
        # all-announcer / empty-cast path: costly cleared -> no must_turn
        c1 = _full_contract("d001", "ANNOUNCER", 0, "")
        c2 = _full_contract("d002", "ANNOUNCER", 1, "")
        assert c1.must_turn is False and c2.must_turn is False
        ok, reasons = validate_episode_contracts(
            [c1, c2], [], ["the vial", "the audit"])
        assert ok is False
        assert any("no slot carries" in r for r in reasons)

    def test_exactly_one_character_turn_passes_audit(self):
        c_open = _full_contract("d001", "ANNOUNCER", 0, "d004")
        c_mid = _full_contract("d002", "EDNA", 1, "d004")
        c_turn = _full_contract("d004", "PETER", 2, "d004")
        ok, reasons = validate_episode_contracts(
            [c_open, c_mid, c_turn], [], ["the vial", "the audit"])
        assert ok is True, reasons
        turn_slots = [c.dialogue_slot_id for c in (c_open, c_mid, c_turn) if c.must_turn]
        assert turn_slots == ["d004"]

    def test_two_turns_fail_audit(self):
        c_a = _full_contract("d002", "EDNA", 1, "d002")
        c_b = _full_contract("d004", "PETER", 2, "d004")
        ok, reasons = validate_episode_contracts(
            [c_a, c_b], [], ["the vial", "the audit"])
        assert ok is False
        assert any("more than one" in r for r in reasons)


# ===========================================================================
# F3 -- ending-aware outro: resolved ending must not hedge
# ===========================================================================

_RESOLVED = "the vial is opened and the record cannot be undone"
_UNRESOLVED = "the truth remains unknown"


def _outro(creative_fn, ending_change=""):
    return compose_announcer_outro(
        creative_fn=creative_fn,
        script_brief="A sealed vial holds the oldest strain.",
        news_close_brief="The vial is opened and recorded.",
        intro_text="Tonight, a sealed vial.",
        creative_repo_id="test",
        ending_change=ending_change,
    )


class TestF3EndingAwareOutro:

    def test_resolved_hedge_then_clean_recomposes(self):
        calls = {"n": 0}

        def mock(messages, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return "Whether she was right remains to be seen, friends."
            return "And so the vial was opened, the record now stands. Good night."

        res = _outro(mock, ending_change=_RESOLVED)
        assert res.compose_flags == ("announcer_outro_resolved_recomposed",)
        assert "remains to be seen" not in res.text.lower()

    def test_resolved_persistent_hedge_keeps_ai_line(self):
        # NO-FALLBACK (2026-07-03): after a recompose the close STILL hedges, but it
        # is a VALID AI-written line (a quality miss, not an LLM failure), so it is
        # KEPT -- no canned resolved-template swap. Flagged for observability.
        def mock(messages, **kw):
            return "Whether she was right remains to be seen tonight, friends."

        res = _outro(mock, ending_change=_RESOLVED)
        assert res.compose_flags == ("announcer_outro", "outro_residual_hedge")
        assert "remains to be seen" in res.text.lower()  # the AI line is kept as-is

    def test_unresolved_hedge_is_allowed(self):
        def mock(messages, **kw):
            return "Whether she was right remains to be seen, friends."

        res = _outro(mock, ending_change=_UNRESOLVED)
        assert res.compose_flags == ("announcer_outro",)
        assert "remains to be seen" in res.text.lower()

    def test_missing_ending_change_no_crash(self):
        def mock(messages, **kw):
            return "And the signal fades into the night. Good night, listeners."

        res = _outro(mock)  # ending_change="" default
        assert res.compose_flags == ("announcer_outro",)
        assert "signal fades" in res.text.lower()

    def test_resolved_clean_first_pass_no_recompose(self):
        def mock(messages, **kw):
            return "And so the vial was opened, the record now stands. Good night."

        res = _outro(mock, ending_change=_RESOLVED)
        assert res.compose_flags == ("announcer_outro",)


# ===========================================================================
# F4 -- speaker gender/pronouns reach the line composer
# ===========================================================================

class TestF4GenderPronouns:

    @pytest.mark.parametrize("gender,subj,obj", [
        ("male", "he", "him"),
        ("female", "she", "her"),
        ("man", "he", "him"),
        ("woman", "she", "her"),
        ("nonbinary", "they", "them"),
        ("agender", "they", "them"),
    ])
    def test_gender_to_pronouns(self, gender, subj, obj):
        pr = _gender_to_pronouns(gender)
        assert pr[0] == subj and pr[1] == obj

    def test_empty_gender_no_pronouns(self):
        assert _gender_to_pronouns("") is None
        assert _gender_to_pronouns(None) is None

    def test_female_directive_in_prompt(self):
        prompt = _build_user_prompt(_req(speaker="MAEVE", speaker_gender="female"))
        assert "MAEVE is female; use she/her pronouns for MAEVE" in prompt

    def test_male_directive_in_prompt(self):
        prompt = _build_user_prompt(_req(speaker="PETER", speaker_gender="male"))
        assert "use he/him pronouns for PETER" in prompt

    def test_no_gender_no_directive(self):
        # name-independent: no PRONOUNS directive when gender is unset
        prompt = _build_user_prompt(_req(speaker="ALICE"))
        assert "pronouns for" not in prompt


# ===========================================================================
# F5 -- speech register (speech_signature) reaches the cast card
# ===========================================================================

class TestF5SpeechSignature:

    def test_card_renders_signature_when_present(self):
        card = build_voice_card({
            "name": "EDNA", "gender": "female",
            "character_description": "weary archivist",
            "speech_signature": "clipped, formal",
        })
        assert "speaks: clipped, formal" in card

    def test_card_backfills_empty_signature(self):
        # key present but empty -> deterministic backfill
        card = build_voice_card({
            "name": "EDNA", "gender": "female", "speech_signature": "",
        })
        assert "speaks: plain spoken" in card

    def test_legacy_card_without_key_unchanged(self):
        # no speech_signature key at all -> byte-identical legacy card
        card = build_voice_card({"name": "EDNA", "gender": "female"})
        assert card == "EDNA (female)"
        assert "speaks:" not in card

    def test_casting_schemas_accept_optional_signature(self):
        # default "" when omitted (legacy/weak model)
        d = DescriptionResponse(character_description="a weary archivist here")
        assert d.speech_signature == ""
        c = CastingResponse(
            character_description="a weary archivist here",
            gender="female", speech_signature="blunt, profane",
        )
        assert c.speech_signature == "blunt, profane"

    def test_all_voice_cards_join_includes_signature(self):
        rows = [
            {"name": "EDNA", "gender": "female", "speech_signature": "clipped"},
            {"name": "PETER", "gender": "male", "speech_signature": "rambling"},
        ]
        joined = "\n".join(build_voice_card(r) for r in rows)
        assert "speaks: clipped" in joined and "speaks: rambling" in joined


# ===========================================================================
# F7 -- narration / self-address hygiene
# ===========================================================================

class TestF7NarrationHygiene:

    def test_first_person_allowed(self):
        assert detect_narration_self_address("I cannot open it.", "EDNA") is False

    def test_legit_third_person_reference_allowed(self):
        # reference to OTHERS, no narration verb on a self lead
        assert detect_narration_self_address("They know the code.", "EDNA") is False
        assert detect_narration_self_address("He is lying about the vial.", "EDNA") is False

    def test_speaker_name_substring_no_trigger(self):
        # "Ednathon" must not match the speaker name "EDNA"
        assert detect_narration_self_address("Ednathon guards the door.", "EDNA") is False

    def test_true_self_narration_triggers(self):
        assert detect_narration_self_address("She paces the dark lab slowly.", "EDNA") is True
        assert detect_narration_self_address("Edna stares at the sealed vial.", "EDNA") is True

    def test_empty_no_trigger(self):
        assert detect_narration_self_address("", "EDNA") is False
        assert detect_narration_self_address(None, "EDNA") is False

    def test_output_constraint_in_prompt(self):
        prompt = _build_user_prompt(_req())
        assert "never narrate your own actions in the third person" in prompt
        assert "never say your own name" in prompt


# ===========================================================================
# F8 -- arc-shape variety
# ===========================================================================

class TestF8ArcShape:

    def test_pick_arc_shape_deterministic_and_valid(self):
        assert pick_arc_shape("seed-123") in ARC_SHAPES
        assert pick_arc_shape("seed-123") == pick_arc_shape("seed-123")

    def test_pick_arc_shape_empty_is_default(self):
        assert pick_arc_shape("") == ARC_SHAPES[0]
        assert pick_arc_shape(None) == ARC_SHAPES[0]

    def test_distribution_not_single_valued(self):
        shapes = {pick_arc_shape(f"news-{i}") for i in range(60)}
        assert len(shapes) > 1

    def test_shape_templates_differ_from_default(self):
        dread = _pick_templates("the vial", "slow_dread")
        default = _pick_templates("the vial", "")
        assert dread != default
        blob = " ".join(dread).lower()
        assert "the vial" in blob

    def test_post_validator_branch_messages(self):
        pv_c = _make_post_validator(["x"], confrontation=True)
        pv_n = _make_post_validator(["x"], confrontation=False)
        identical = DramaticStateLLM(
            character_a_wants="same want x", character_b_wants="same want x",
            dramatic_question="what happens to x next?",
            ending_change="x changes for good")
        assert "opposed" in pv_c(identical)
        assert "identical" in pv_n(identical)

    def test_post_validator_nonconfrontation_accepts_distinct(self):
        pv_n = _make_post_validator(["x"], confrontation=False)
        distinct = DramaticStateLLM(
            character_a_wants="seek the truth about x",
            character_b_wants="protect those x would expose",
            dramatic_question="will the truth of x ever emerge?",
            ending_change="x stays partly unknown")
        assert pv_n(distinct) is None

    def test_derive_llm_failure_fails_loud_after_stamping_shape(self):
        # NO-FALLBACK (2026-07-03): a failing dramatic-state LLM call RAISES; it no
        # longer degrades to the news-templated fallback. arc_shape is still stamped
        # into meta (before the LLM call) as a side effect.
        import pytest

        meta = {"news": {"key_terms": ["the vial"],
                         "script_brief": "a sealed vial holds a strain"}}

        def boom(**kw):
            raise RuntimeError("no llm in test")

        with pytest.raises(RuntimeError, match="no-fallback"):
            derive_news_dramatic_state(
                meta=meta,
                cast_rows=[{"name": "EDNA"}, {"name": "PETER"}],
                voice_slot_ids=["d001", "d002", "d003", "d004"],
                slot_fn=None, structured_call_fn=boom, arc_shape="slow_dread",
            )
        assert meta["arc_shape"] == "slow_dread"
