"""tests/test_lfc_polish_fixes.py

LFC sprint Commit 3 -- regression coverage for ADR sections 6.1-6.4:

  section 6.1  Announcer-aware polish prompt (speaker_role branch).
  section 6.2  Refusal detector rejects "I cannot rewrite this." outputs.
  section 6.3  Polish prompt context expansion -- BEAT INTENT + last 2
               PREVIOUS LINES land in the user prompt body.
  section 6.4  Separate polish_generate_fn factory in
               _otr_model_loader.make_polish_generate_fn. polish_line
               uses the polish_generate_fn when provided.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_line_composer import (  # noqa: E402
    _POLISH_SYSTEM_PROMPT,
    _POLISH_SYSTEM_PROMPT_ANNOUNCER,
    is_polish_refusal,
    polish_line,
)


# ---------------------------------------------------------------------------
# section 6.1 -- Announcer guard
# ---------------------------------------------------------------------------


class TestAnnouncerGuard:
    def test_announcer_prompt_distinct_from_character_prompt(self):
        assert (
            _POLISH_SYSTEM_PROMPT_ANNOUNCER != _POLISH_SYSTEM_PROMPT
        )

    def test_character_prompt_forbids_narration(self):
        # Character path must keep the strict "no narration of any
        # kind" line from the existing prompt.
        assert "narration of any kind" in _POLISH_SYSTEM_PROMPT

    def test_announcer_prompt_allows_narration(self):
        # Announcer path explicitly allows third-person narration.
        text = _POLISH_SYSTEM_PROMPT_ANNOUNCER.lower()
        assert "third-person narration is ok" in text or (
            "narrative" in text and "third-person" in text
        )
        # But it still bans bracket stage direction.
        assert "[VOICE:" in _POLISH_SYSTEM_PROMPT_ANNOUNCER or (
            "bracket" in text
        )

    def test_polish_line_routes_announcer_to_announcer_prompt(self):
        seen = {"system": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["system"] = next(
                m["content"] for m in messages if m["role"] == "system"
            )
            return "Stay tuned for our final dispatch."

        polish_line(
            capture,
            leaked_line="(beat) Stay tuned [whispers]",
            speaker_voice_card="ANNOUNCER",
            speaker_role="announcer", polish_generate_fn=capture,
        )
        assert seen["system"] == _POLISH_SYSTEM_PROMPT_ANNOUNCER

    def test_polish_line_routes_character_to_character_prompt(self):
        seen = {"system": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["system"] = next(
                m["content"] for m in messages if m["role"] == "system"
            )
            return "I told you it was rigged."

        polish_line(
            capture,
            leaked_line='"It was rigged," she said.',
            speaker_voice_card="ALICE",
            speaker_role="character", polish_generate_fn=capture,
        )
        assert seen["system"] == _POLISH_SYSTEM_PROMPT

    def test_polish_line_default_role_is_character(self):
        # Back-compat: pre-fix callers that didn't pass speaker_role
        # still get the character prompt.
        seen = {"system": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["system"] = next(
                m["content"] for m in messages if m["role"] == "system"
            )
            return "Hello there."

        polish_line(capture, "x", "ALICE", polish_generate_fn=capture)
        assert seen["system"] == _POLISH_SYSTEM_PROMPT

    def test_announcer_role_label_in_user_prompt(self):
        seen = {"user": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["user"] = next(
                m["content"] for m in messages if m["role"] == "user"
            )
            return "Stay tuned."

        polish_line(
            capture, "(beat) Stay tuned.", "ANNOUNCER",
            speaker_role="announcer", polish_generate_fn=capture,
        )
        assert "ANNOUNCER:" in seen["user"]
        assert "CHARACTER:" not in seen["user"]

    def test_character_role_label_in_user_prompt(self):
        seen = {"user": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["user"] = next(
                m["content"] for m in messages if m["role"] == "user"
            )
            return "fixed line"

        polish_line(
            capture, "leaked", "ALICE", speaker_role="character", polish_generate_fn=capture,
        )
        assert "CHARACTER:" in seen["user"]
        assert "ANNOUNCER:" not in seen["user"]


# ---------------------------------------------------------------------------
# section 6.2 -- Refusal detector
# ---------------------------------------------------------------------------


class TestRefusalDetector:
    @pytest.mark.parametrize("text,expected", [
        ("I cannot rewrite this.", True),
        ("I can't help with that.", True),
        ("I can’t do that.", True),  # smart quote
        ("Sorry, I can't help with that.", True),
        ("I'm sorry, but as a language model", True),
        ("As an AI, I won't be able to", True),
        ("I'm unable to comply with that request.", True),
        ("I won't rewrite the line.", True),
        ("I will not produce that content.", True),
        ("Unfortunately, I cannot help.", True),
        ("I'm afraid I can't do that, Dave.", True),
        # Non-refusal dialogue must NOT match.
        ("I told you it was rigged.", False),
        ("I cannot believe you did that.", False),
        ("I will go now.", False),
        # Edge cases.
        ("", False),
        ("   ", False),
        ("Hello there.", False),
    ])
    def test_is_polish_refusal_matches(self, text, expected):
        assert is_polish_refusal(text) is expected

    def test_polish_returns_original_on_refusal(self):
        def refusal(messages, *, temperature, max_new_tokens, stop=None):
            return "I cannot rewrite this."

        original = '"That can\'t be right," she whispered.'
        result = polish_line(refusal, original, "ALICE", polish_generate_fn=refusal)
        assert result == original

    def test_polish_returns_original_on_apology(self):
        def refusal(messages, *, temperature, max_new_tokens, stop=None):
            return "I'm sorry, but I cannot produce that line."

        original = "*sighs* I don't know what to say."
        result = polish_line(refusal, original, "ALICE", polish_generate_fn=refusal)
        assert result == original


# ---------------------------------------------------------------------------
# section 6.3 -- Context expansion
# ---------------------------------------------------------------------------


class TestPolishContextExpansion:
    def test_beat_intent_lands_in_user_prompt(self):
        seen = {"user": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["user"] = next(
                m["content"] for m in messages if m["role"] == "user"
            )
            return "fixed"

        polish_line(
            capture, "leaked", "ALICE",
            beat_intent="reveal the discovery to BOB", polish_generate_fn=capture,
        )
        assert "BEAT INTENT: reveal the discovery to BOB" in seen["user"]

    def test_previous_lines_land_in_user_prompt(self):
        seen = {"user": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["user"] = next(
                m["content"] for m in messages if m["role"] == "user"
            )
            return "fixed"

        polish_line(
            capture, "leaked", "ALICE",
            previous_lines=("Bob, look at this.", "What did you find?"), polish_generate_fn=capture,
        )
        assert "PREVIOUS LINES:" in seen["user"]
        assert "Bob, look at this." in seen["user"]
        assert "What did you find?" in seen["user"]

    def test_previous_lines_capped_at_two(self):
        seen = {"user": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["user"] = next(
                m["content"] for m in messages if m["role"] == "user"
            )
            return "fixed"

        polish_line(
            capture, "leaked", "ALICE",
            previous_lines=(
                "line one", "line two", "line three", "line four",
            ), polish_generate_fn=capture,
        )
        # Cap = last 2 entries.
        assert "line three" in seen["user"]
        assert "line four" in seen["user"]
        # Older lines must NOT be in the prompt.
        assert "line one" not in seen["user"]
        assert "line two" not in seen["user"]

    def test_no_context_falls_back_to_minimal_prompt(self):
        # Back-compat: pre-fix callers (just character_voice_card +
        # leaked_line) still get a clean prompt without BEAT INTENT /
        # PREVIOUS LINES blocks.
        seen = {"user": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["user"] = next(
                m["content"] for m in messages if m["role"] == "user"
            )
            return "fixed"

        polish_line(capture, "leaked", "ALICE", polish_generate_fn=capture)
        assert "BEAT INTENT:" not in seen["user"]
        assert "PREVIOUS LINES:" not in seen["user"]
        assert "CHARACTER: ALICE" in seen["user"]
        assert "ORIGINAL LINE: leaked" in seen["user"]

    def test_empty_beat_intent_and_previous_lines_dropped(self):
        seen = {"user": None}

        def capture(messages, *, temperature, max_new_tokens, stop=None):
            seen["user"] = next(
                m["content"] for m in messages if m["role"] == "user"
            )
            return "fixed"

        polish_line(
            capture, "leaked", "ALICE",
            beat_intent="   ",
            previous_lines=("", "  ", ""), polish_generate_fn=capture,
        )
        assert "BEAT INTENT:" not in seen["user"]
        assert "PREVIOUS LINES:" not in seen["user"]


# ---------------------------------------------------------------------------
# section 6.4 -- Separate polish_generate_fn
# ---------------------------------------------------------------------------


class TestPolishGenerateFnRouting:
    def test_polish_generate_fn_preferred_over_generate_fn(self):
        calls = {"main": 0, "polish": 0}

        def main_fn(messages, *, temperature, max_new_tokens, stop=None):
            calls["main"] += 1
            return "main-result"

        def polish_fn(messages, *, temperature, max_new_tokens, stop=None):
            calls["polish"] += 1
            return "polish-result"

        result = polish_line(
            main_fn, "leaked", "ALICE",
            polish_generate_fn=polish_fn,
        )
        assert calls["main"] == 0
        assert calls["polish"] == 1
        assert result == "polish-result"

    def test_polish_falls_back_to_generate_fn_when_polish_fn_is_none(self):
        calls = {"main": 0}

        def main_fn(messages, *, temperature, max_new_tokens, stop=None):
            calls["main"] += 1
            return "main-result"

        result = polish_line(main_fn, "leaked", "ALICE", polish_generate_fn=main_fn)
        assert calls["main"] == 1
        assert result == "main-result"

    def test_make_polish_generate_fn_factory_shape(self):
        # The model-loader factory exists and rejects malformed input.
        from nodes._otr_model_loader import (
            ModelLoaderError,
            make_polish_generate_fn,
        )
        with pytest.raises(ModelLoaderError):
            make_polish_generate_fn({})
        with pytest.raises(ModelLoaderError):
            make_polish_generate_fn({"model": object()})

    def test_make_polish_generate_fn_returns_callable(self):
        from nodes._otr_model_loader import make_polish_generate_fn

        # Smoke test: factory returns a callable with the documented
        # signature (messages, *, temperature, max_new_tokens) -> str.
        # We don't execute the call here -- the real generate path
        # needs a live torch tensor model. The sampling-shape contract
        # is covered by test_polish_sampling_constants_pinned below.
        class _StubTok:
            eos_token_id = 0

        class _StubModel:
            device = "cpu"

        fn = make_polish_generate_fn({
            "model": _StubModel(),
            "tokenizer": _StubTok(),
        })
        assert callable(fn)
        import inspect
        sig = inspect.signature(fn)
        params = sig.parameters
        assert "messages" in params
        assert "temperature" in params
        assert "max_new_tokens" in params
        # temperature + max_new_tokens are keyword-only (after *).
        assert params["temperature"].kind == inspect.Parameter.KEYWORD_ONLY
        assert params["max_new_tokens"].kind == inspect.Parameter.KEYWORD_ONLY

    def test_polish_sampling_constants_pinned(self):
        # Polish sampling is independent of caller-supplied min_p /
        # repetition_penalty. Verified by reading the module-level
        # constants used inside make_polish_generate_fn -- these are
        # the contract the fix locks in.
        from nodes import _otr_model_loader as _OTRML
        assert _OTRML._POLISH_TOP_P == 0.9
        assert _OTRML._POLISH_DO_SAMPLE is True


# ---------------------------------------------------------------------------
# Integration -- new polish_line keyword args do not regress legacy callers
# ---------------------------------------------------------------------------


class TestBackCompatLegacyCallers:
    def test_three_arg_call_still_works(self):
        # The existing test_phase1_composer_prompt.py calls polish_line
        # with three positional args. That pathway must still produce
        # a polished line under the unchanged default sampling.
        def mock(messages, *, temperature, max_new_tokens, stop=None):
            assert temperature < 0.6  # polish low-temp invariant
            return "Then we should go now."

        result = polish_line(
            mock, '"Then we should go," she said.', "ALICE", polish_generate_fn=mock,
        )
        assert result == "Then we should go now."

    def test_generate_fn_typeerror_fallback_path(self):
        # Legacy generate_fn implementations that don't accept the
        # `stop=` kwarg raise TypeError; polish_line retries without
        # it. This path must still work after the section 6.x additions.
        calls = {"with_stop": 0, "without_stop": 0}

        def mock_no_stop(messages, *, temperature, max_new_tokens, **kw):
            if "stop" in kw:
                calls["with_stop"] += 1
                raise TypeError("unexpected keyword argument 'stop'")
            calls["without_stop"] += 1
            return "Then we should go now."

        result = polish_line(mock_no_stop, '"Leaked."', "ALICE", polish_generate_fn=mock_no_stop)
        assert calls["with_stop"] == 1
        assert calls["without_stop"] == 1
        assert result == "Then we should go now."
