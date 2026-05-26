"""Sprint 10A step 3-D regression -- Stage 1 prompt carries explicit bounds.

The Stage 1 system prompt must instruct the LLM about constraints that
lm-format-enforcer's JsonSchemaParser does NOT enforce at the
token-sampling layer. The shadow-pass soak runs on 2026-05-26 surfaced
two failure families that prompt-layer instruction can address:

  * length_target_words = 0 (lmfe doesn't enforce numeric min/max --
    surfaced 2/2 attempts on 2 runs; first-attempt valid plan rate
    at risk without 3-D).
  * Name-gender inversion (cast audit caught a 'Dr. Amelia Hart'
    with male-coded voice; legacy casting caught NELSON CORBEN=female
    in the same run). Schema can't encode 'this name expects this
    gender'; the prompt has to teach the convention.

These tests pin the bullets are present. They do NOT exercise the
LLM (no transformers load) -- they only assert the prompt string
carries the required language. The operator soak gate measures
whether the LLM actually obeys.
"""

from __future__ import annotations

from nodes._otr_stage1_call import _STAGE1_SYSTEM_PROMPT


class TestPromptCarriesNumericBound:
    def test_length_target_words_explicit_range(self):
        """length_target_words bullet must spell out the exact
        5-200 integer range so the LLM has no excuse for emitting
        0 / negative / out-of-range values."""
        prompt = _STAGE1_SYSTEM_PROMPT
        assert "length_target_words" in prompt
        assert "5 and 200" in prompt or "5 to 200" in prompt
        assert "integer" in prompt.lower()
        # The 'never emit 0' instruction is what specifically targets
        # the 2026-05-26 failure mode.
        assert ("Never emit 0" in prompt
                or "never emit 0" in prompt
                or "Never emit zero" in prompt
                or "never emit zero" in prompt), (
            "Stage 1 prompt must explicitly forbid emitting "
            "length_target_words=0 (the failure mode surfaced on "
            "pending_20260526_114111 and pending_20260526_120436)"
        )

    def test_typical_range_hint_present(self):
        """A typical-range hint (e.g. '15 to 45 words') anchors the
        LLM to a sensible distribution so it doesn't reach for the
        extreme values just because they're legal."""
        prompt = _STAGE1_SYSTEM_PROMPT
        # Some flavor of "a typical beat is X to Y words" must be
        # present. Pin via a low-specificity regex so the language
        # can be tweaked without breaking this test.
        import re
        assert re.search(
            r"typical beat is.*\d+.*\d+.*word",
            prompt,
            re.IGNORECASE,
        ), (
            "Stage 1 prompt should give the LLM a typical-range "
            "anchor for length_target_words so it doesn't reach for "
            "the schema bounds by default."
        )


class TestPromptCarriesNameGenderHint:
    def test_male_coded_names_listed(self):
        """Operator-flagged male-coded names that the LLM has been
        inverting must appear in the prompt as explicit male examples."""
        prompt = _STAGE1_SYSTEM_PROMPT
        for name in ("ROBINSON", "COLE", "NELSON", "OSCAR", "REGINALD"):
            assert name in prompt, (
                f"Stage 1 prompt must list {name!r} as a male-coded "
                f"name. This name was flagged by the operator in a "
                f"real soak run with the LLM inverting its gender."
            )

    def test_female_coded_names_listed(self):
        """Female-coded names that the LLM has inverted in soak."""
        prompt = _STAGE1_SYSTEM_PROMPT
        for name in ("MIRA",):
            assert name in prompt, (
                f"Stage 1 prompt must list {name!r} as a female-coded "
                f"name."
            )

    def test_voice_gender_mapping_explicit(self):
        """The Bark v2 pool's male/female split should be enumerated
        so the LLM doesn't pick v2/en_speaker_0 (male) for a female
        character (the 'Dr. Amelia Hart' failure on 2026-05-26)."""
        prompt = _STAGE1_SYSTEM_PROMPT
        # Look for some flavor of "voice_id 0, 1, 3, 5, 6, 8 are male"
        # without pinning exact phrasing.
        assert "male-coded" in prompt or "male coded" in prompt
        assert "female-coded" in prompt or "female coded" in prompt
        # And at least one specific voice_id from each gender group
        # must appear.
        for vid in ("0", "1", "3", "5"):
            assert vid in prompt
        for vid in ("2", "4", "7"):
            assert vid in prompt


class TestPromptStillCarriesUnchangedRules:
    """Belt and braces: the 3-D edits must not have removed any of
    the existing hard rules."""

    def test_existing_rules_intact(self):
        prompt = _STAGE1_SYSTEM_PROMPT
        assert "v2/en_speaker_0 through v2/en_speaker_9" in prompt
        assert "bNNN" in prompt
        assert "ANNOUNCER" in prompt
        assert "MUSIC" in prompt
        assert "callback_to" in prompt
        assert "SFW" in prompt
        assert "Output JSON ONLY" in prompt
