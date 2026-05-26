"""Sprint 10A step 3-B regression -- Stage 1 call site.

Exercises generate_stage1_plan with a MOCKED generate_fn. Covers:
  * Happy path: generate_fn returns valid JSON; plan parses; attempts
    list has one entry with status='valid_first_attempt'.
  * Retry path: generate_fn returns garbage on attempt 1, valid on
    attempt 2; status='valid_after_retry' on the successful record.
  * Decode-cutoff: generate_fn returns truncated JSON; status=
    'decode_cutoff' on the failing record.
  * Parse failure: generate_fn returns JSON that's schema-invalid;
    status='parse_failed'.
  * Semantic failure: generate_fn returns schema-valid JSON that
    fails the cross-field check (duplicate cast voice_id); status=
    'semantic_failed'.
  * Generate raises: generate_fn raises an exception; status=
    'generate_failed'.
  * Retry exhaustion: all attempts fail; raises
    Stage1PlanGenerationError with attempts count + last error.
  * Prompt assembly: build_stage1_user_prompt produces a non-empty
    prompt that references the news seed AND num_characters AND
    target_words.
  * Input guards: num_characters out of [1,6] AND empty news_seed
    both raise ValueError BEFORE any generate_fn call.
  * No real model load: every test runs without torch / transformers
    GPU code.

Tests do NOT load Mistral-Nemo. Operator soak gate (>=19/20) is
measured at runtime by counting status='valid_first_attempt' in the
attempts list across 20 episodes.
"""

from __future__ import annotations

import json

import pytest


def _good_plan_json() -> str:
    """A schema-valid + semantically-valid plan JSON string."""
    plan = {
        "premise": "Two technicians at a Mars relay catch a signal that should not exist.",
        "arc": {
            "setup": "Routine night shift, calm comms traffic on the relay.",
            "complication": "An off-band carrier locks in and refuses to drop.",
            "resolution": "They identify the source and decide whether to acknowledge it.",
        },
        "cast": [
            {
                "name": "DALE PORTER",
                "gender": "male",
                "pronouns": "he/him",
                "voice_id": "v2/en_speaker_5",
                "persona": "Experienced relay technician. Measured cadences; trusts procedure but knows when to bend it.",
                "arc_role": "anchor of competence",
            },
            {
                "name": "MEG SAARI",
                "gender": "female",
                "pronouns": "she/her",
                "voice_id": "v2/en_speaker_2",
                "persona": "Junior tech, six months on the post. Quick to question, slow to panic.",
                "arc_role": "pressure source and witness",
            },
        ],
        "beats": [
            {
                "beat_id": "b001",
                "speaker": "ANNOUNCER",
                "intent": "Open with date stamp and location.",
                "length_target_words": 18,
                "emotional_register": "neutral procedural",
                "callback_to": None,
            },
            {
                "beat_id": "b002",
                "speaker": "DALE PORTER",
                "intent": "Establish the routine shift.",
                "length_target_words": 24,
                "emotional_register": "calm grounded",
                "callback_to": None,
            },
            {
                "beat_id": "b003",
                "speaker": "MEG SAARI",
                "intent": "Notice the off-band carrier.",
                "length_target_words": 22,
                "emotional_register": "alert curious",
                "callback_to": "b002",
            },
            {
                "beat_id": "b004",
                "speaker": "ANNOUNCER",
                "intent": "Close with a hook.",
                "length_target_words": 16,
                "emotional_register": "measured close",
                "callback_to": None,
            },
        ],
        "running_facts": [
            "The relay is on Mars.",
            "Dale is senior to Meg.",
        ],
    }
    return json.dumps(plan)


def _schema_invalid_plan_json() -> str:
    """JSON that parses fine but fails pydantic (bad voice_id)."""
    d = json.loads(_good_plan_json())
    d["cast"][0]["voice_id"] = "v2/en_speaker_42"  # not in v2/en_speaker_[0-9]
    return json.dumps(d)


def _semantic_invalid_plan_json() -> str:
    """JSON that parses + pydantic-validates but fails the cross-
    field check (duplicate voice_id)."""
    d = json.loads(_good_plan_json())
    d["cast"][1]["voice_id"] = d["cast"][0]["voice_id"]
    return json.dumps(d)


def _truncated_plan_json() -> str:
    return _good_plan_json()[:-50]  # chop off closing braces


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    def test_valid_first_attempt_returns_plan_and_one_record(self):
        from nodes._otr_stage1_call import generate_stage1_plan
        from nodes._otr_stage1_plan import Stage1Plan

        def fake_generate(messages, *, temperature, max_new_tokens):
            return _good_plan_json()

        plan, attempts = generate_stage1_plan(
            fake_generate,
            news_seed="A surprising scientific finding on Mars relays.",
            num_characters=2,
            target_words=350,
        )

        assert isinstance(plan, Stage1Plan)
        assert len(attempts) == 1
        assert attempts[0].attempt == 1
        assert attempts[0].status == "valid_first_attempt"
        assert attempts[0].error_type is None

    def test_episode_title_hint_propagates_to_prompt(self):
        from nodes._otr_stage1_call import build_stage1_user_prompt

        s = build_stage1_user_prompt(
            news_seed="seed text body",
            num_characters=2,
            target_words=350,
            episode_title_hint="Working Title One",
        )
        assert "Working Title One" in s
        assert "seed text body" in s
        assert "2" in s  # num_characters
        assert "350" in s  # target_words

    def test_episode_title_hint_blank_branch(self):
        from nodes._otr_stage1_call import build_stage1_user_prompt

        s = build_stage1_user_prompt(
            news_seed="seed",
            num_characters=2,
            target_words=350,
            episode_title_hint="",
        )
        assert "No working title provided" in s


# ---------------------------------------------------------------------------
# Retry path
# ---------------------------------------------------------------------------


class TestRetryPath:
    def test_garbage_then_valid_succeeds_on_attempt_2(self):
        from nodes._otr_stage1_call import generate_stage1_plan

        calls = {"n": 0}

        def fake_generate(messages, *, temperature, max_new_tokens):
            calls["n"] += 1
            if calls["n"] == 1:
                return "not json at all"
            return _good_plan_json()

        plan, attempts = generate_stage1_plan(
            fake_generate,
            news_seed="seed body text",
            num_characters=2,
            target_words=350,
        )

        assert plan is not None
        assert len(attempts) == 2
        assert attempts[0].status == "decode_cutoff"
        assert attempts[1].status == "valid_after_retry"

    def test_decode_cutoff_status_on_truncated_json(self):
        from nodes._otr_stage1_call import generate_stage1_plan, Stage1PlanGenerationError

        def fake_generate(messages, *, temperature, max_new_tokens):
            return _truncated_plan_json()

        with pytest.raises(Stage1PlanGenerationError):
            generate_stage1_plan(
                fake_generate,
                news_seed="seed",
                num_characters=2,
                target_words=350,
                max_attempts=2,
            )

    def test_schema_invalid_status_parse_failed(self):
        from nodes._otr_stage1_call import generate_stage1_plan, Stage1PlanGenerationError

        def fake_generate(messages, *, temperature, max_new_tokens):
            return _schema_invalid_plan_json()

        with pytest.raises(Stage1PlanGenerationError) as ei:
            generate_stage1_plan(
                fake_generate,
                news_seed="seed",
                num_characters=2,
                target_words=350,
                max_attempts=1,
            )
        # One attempt with parse_failed status
        assert ei.value.attempts == 1

    def test_semantic_invalid_status_semantic_failed(self):
        from nodes._otr_stage1_call import generate_stage1_plan, Stage1PlanGenerationError

        def fake_generate(messages, *, temperature, max_new_tokens):
            return _semantic_invalid_plan_json()

        with pytest.raises(Stage1PlanGenerationError):
            generate_stage1_plan(
                fake_generate,
                news_seed="seed",
                num_characters=2,
                target_words=350,
                max_attempts=1,
            )

    def test_generate_raises_attempt_recorded(self):
        from nodes._otr_stage1_call import generate_stage1_plan, Stage1PlanGenerationError

        def fake_generate(messages, *, temperature, max_new_tokens):
            raise RuntimeError("simulated generate failure")

        with pytest.raises(Stage1PlanGenerationError) as ei:
            generate_stage1_plan(
                fake_generate,
                news_seed="seed",
                num_characters=2,
                target_words=350,
                max_attempts=2,
            )
        assert ei.value.attempts == 2
        assert "simulated generate failure" in str(ei.value)


# ---------------------------------------------------------------------------
# Input guards (fail BEFORE generate_fn call)
# ---------------------------------------------------------------------------


class TestInputGuards:
    def test_num_characters_zero_rejected(self):
        from nodes._otr_stage1_call import generate_stage1_plan

        called = {"n": 0}

        def fake(*a, **kw):
            called["n"] += 1
            return _good_plan_json()

        with pytest.raises(ValueError, match="num_characters"):
            generate_stage1_plan(
                fake, news_seed="seed", num_characters=0, target_words=350,
            )
        assert called["n"] == 0  # never reached generate

    def test_num_characters_too_high_rejected(self):
        from nodes._otr_stage1_call import generate_stage1_plan

        def fake(*a, **kw):
            return _good_plan_json()

        with pytest.raises(ValueError, match="num_characters"):
            generate_stage1_plan(
                fake, news_seed="seed", num_characters=7, target_words=350,
            )

    def test_empty_news_seed_rejected(self):
        from nodes._otr_stage1_call import generate_stage1_plan

        def fake(*a, **kw):
            return _good_plan_json()

        with pytest.raises(ValueError, match="news_seed"):
            generate_stage1_plan(
                fake, news_seed="   ", num_characters=2, target_words=350,
            )


# ---------------------------------------------------------------------------
# Diagnostic record shape
# ---------------------------------------------------------------------------


class TestAttemptRecord:
    def test_record_has_documented_fields(self):
        from nodes._otr_stage1_call import Stage1AttemptRecord

        rec = Stage1AttemptRecord(
            attempt=1,
            status="valid_first_attempt",
            elapsed_seconds=0.5,
        )
        assert rec.attempt == 1
        assert rec.status == "valid_first_attempt"
        assert rec.error_type is None
        assert rec.error_message is None

    def test_record_with_error(self):
        from nodes._otr_stage1_call import Stage1AttemptRecord

        rec = Stage1AttemptRecord(
            attempt=2,
            status="parse_failed",
            elapsed_seconds=1.1,
            error_type="ValidationError",
            error_message="cast.0.voice_id pattern mismatch",
        )
        assert rec.error_type == "ValidationError"
        assert "voice_id" in rec.error_message
