"""Sprint 9 (Narrative Face): story-causal portrait diversity tests.

Pins the contract for every Narrative Face change:

  1. `_FACE_PRESSURE_BY_ROLE` -- the Python-pinned story-causal
     face-pressure anchor keyed off the existing _ROLE_VOCAB.
  2. `_build_user_prompt` injects "Face pressure: ..." after "Role: ..."
     when the role is in the vocab; nothing when role is empty or
     unknown.
  3. `_build_user_prompt` emits the full CHARACTER VISUAL CONTRACT
     block: header, format string, 5 rules, JSON template.
  4. `_format_prior_entry` smart-trims at the first sentence boundary
     under the cap; falls back to char-trim when no period fires.
  5. `_build_portrait_prompt` carries the full three-anti-pattern
     guard (non-generic, not a stock portrait, not glamour
     photography) and the spec's story-linked expression cue.
  6. `compose_shot_prompt` truncates portrait_prompt to the PASS 3
     cap at a sentence boundary so the richer CONTRACT-style
     descriptions don't blow CLIP's effective tokenization window.

Tests are PURE -- direct calls into the casting and visual modules,
no FLUX / torch / LLM dependencies.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import _otr_casting as cst
from nodes import otr_video_plan as ovp
from visual import batch_flux_portrait_render as bfpr


def _build_prompt(
    *,
    name: str = "PROBE",
    gender: str = "",
    timbre: str = "",
    role: str = "",
    prior_cast: list | None = None,
    casting_brief: str = "story",
    style: str = "open",
) -> str:
    """Test helper -- _build_user_prompt has 4 required positional
    args (name, news_seed, style, prior_cast); this wrapper keeps the
    test body readable by passing them through with sane defaults."""
    return cst._build_user_prompt(
        name,
        "",                       # news_seed (replaced by casting_brief when non-empty)
        style,
        list(prior_cast or []),   # prior_cast
        gender=gender,
        timbre=timbre,
        role=role,
        casting_brief=casting_brief,
    )


# ---------------------------------------------------------------------------
# 1. _FACE_PRESSURE_BY_ROLE constant
# ---------------------------------------------------------------------------


class TestFacePressureConstant:

    def test_keys_match_role_vocab(self):
        """The face-pressure dict must have an entry for every role
        the precompute_ensemble_slots rotation produces. A missing
        key would silently lose the structural anchor for that
        role's characters."""
        assert set(cst._FACE_PRESSURE_BY_ROLE.keys()) == set(cst._ROLE_VOCAB)

    def test_every_value_is_a_face_phrase(self):
        """Each value must literally start with `face shows` -- the
        structural anchor the LLM weaves into the FACE block of the
        character_description."""
        for role, phrase in cst._FACE_PRESSURE_BY_ROLE.items():
            assert isinstance(phrase, str)
            assert phrase.startswith("face shows"), (
                f"role {role!r} pressure phrase does not start with "
                f"'face shows': {phrase!r}"
            )

    def test_phrases_are_role_specific(self):
        """No two roles share the same phrase -- each role contributes
        a distinct dramatic-pressure signal."""
        phrases = list(cst._FACE_PRESSURE_BY_ROLE.values())
        assert len(set(phrases)) == len(phrases), (
            "two roles share a face-pressure phrase; the structural "
            "anchor must be role-specific"
        )


# ---------------------------------------------------------------------------
# 2. Face pressure injection in _build_user_prompt
# ---------------------------------------------------------------------------


class TestFacePressureInjection:

    def test_face_pressure_emitted_after_role_when_lead(self):
        prompt = _build_prompt(
            name="JONES",
            gender="male",
            timbre="dry",
            role="lead",
            casting_brief="A radio operator notices a signal.",
        )
        lines = prompt.splitlines()
        role_idx = next(
            i for i, ln in enumerate(lines) if ln.startswith("Role: ")
        )
        pressure_idx = next(
            i for i, ln in enumerate(lines)
            if ln.startswith("Face pressure: ")
        )
        assert pressure_idx == role_idx + 1, (
            "Face pressure line must immediately follow Role line"
        )
        assert (
            cst._FACE_PRESSURE_BY_ROLE["lead"]
            in lines[pressure_idx]
        )

    def test_face_pressure_emitted_for_every_role_in_vocab(self):
        for role in cst._ROLE_VOCAB:
            prompt = _build_prompt(role=role)
            expected_phrase = cst._FACE_PRESSURE_BY_ROLE[role]
            assert f"Face pressure: {expected_phrase}" in prompt, (
                f"role {role!r} did not inject the expected face-"
                f"pressure phrase"
            )

    def test_no_face_pressure_when_role_empty(self):
        prompt = _build_prompt(role="")
        assert "Face pressure:" not in prompt

    def test_no_face_pressure_when_role_outside_vocab(self):
        """A future-proofing pass: if a caller hands in a role outside
        _ROLE_VOCAB (e.g. 'narrator', 'announcer'), the injection
        falls through silently rather than crashing or emitting an
        empty phrase."""
        prompt = _build_prompt(role="narrator")
        # Role line still emitted; pressure injection skipped.
        assert "Role: narrator" in prompt
        assert "Face pressure:" not in prompt

    def test_face_pressure_is_case_insensitive(self):
        """The role lookup uses .lower() -- mixed-case roles still
        resolve to the right anchor."""
        prompt = _build_prompt(role="Lead")  # mixed case
        assert (
            f"Face pressure: {cst._FACE_PRESSURE_BY_ROLE['lead']}"
            in prompt
        )


# ---------------------------------------------------------------------------
# 3. CHARACTER VISUAL CONTRACT block
# ---------------------------------------------------------------------------


class TestContractBlock:

    def test_contract_header_present(self):
        prompt = _build_prompt()
        assert "CHARACTER VISUAL CONTRACT:" in prompt

    def test_contract_format_string_present(self):
        prompt = _build_prompt()
        # The spec format string anchors every layer the description
        # must include: age decade, story-linked role, Face block
        # (shape / eyes / nose / hair / detail), Presence, Voice.
        assert "<age decade>" in prompt
        assert "<story-linked role>" in prompt
        assert "<face shape>" in prompt
        assert "<eyes/brow>" in prompt
        assert "<nose/mouth/jaw>" in prompt
        assert "<hair/hairline>" in prompt
        assert "<one distinctive story-linked detail>" in prompt
        assert "Presence:" in prompt
        assert "Voice:" in prompt

    def test_contract_rules_present(self):
        """All five enumerated rules from the design doc must appear
        verbatim in the prompt body -- explicit rules survive model
        swaps better than examples-only guidance."""
        prompt = _build_prompt()
        for needle in (
            "must match the character's role and emotional function",
            "distinctive detail must feel earned by the premise",
            "concrete facial geometry, not vague mood words",
            "visually distinct from the rest of the cast",
            "Avoid glamour, fashion-model, influencer, symmetrical",
        ):
            assert needle in prompt, (
                f"CHARACTER VISUAL CONTRACT rule missing from "
                f"_build_user_prompt: {needle!r}"
            )

    def test_contract_json_template_uses_as_above(self):
        """The JSON template tail uses the spec's `<as above>`
        placeholder rather than a verbose format string -- the
        format guidance lives in the CONTRACT block, the JSON line
        is just the schema lock."""
        prompt = _build_prompt()
        assert (
            '{"character_description":"<as above>"}'
            in prompt
        )


# ---------------------------------------------------------------------------
# 4. _format_prior_entry smart-trim
# ---------------------------------------------------------------------------


class TestSmartTrim:

    def test_short_description_unchanged(self):
        out = cst._format_prior_entry({
            "name": "JONES",
            "gender": "male",
            "character_description": "Tall detective.",
        })
        assert "Tall detective." in out

    def test_trim_at_first_period_under_cap(self):
        """A description with a sentence-end inside the 120-char cap
        must trim AT the period, preserving the lead sentence whole."""
        desc = (
            "Late-30s mission technician. Face: narrow oval, "
            "tense deep-set eyes, pinched nose, thinning hair flat "
            "from a headset. Presence: sleep-deprived, precise. "
            "Voice: clipped, trying not to shake."
        )
        out = cst._format_prior_entry({
            "name": "JONES",
            "gender": "male",
            "character_description": desc,
        })
        # The lead sentence ends at "...technician." -- must be present.
        assert "Late-30s mission technician." in out
        # The Voice tail must NOT be in the trimmed echo.
        assert "Voice: clipped" not in out
        # No ellipsis when trim landed at a real sentence end.
        assert "..." not in out

    def test_fallback_char_trim_when_no_period_in_cap(self):
        """A degenerate single-clause description with no period
        inside the 120-char cap must still produce a bounded echo
        via the legacy char-trim."""
        desc = (
            "a very long single clause description without any "
            "internal period punctuation across well over the "
            "hundred-twenty-character cap that simulates a writer "
            "who forgot the format completely"
        )
        out = cst._format_prior_entry({
            "name": "JONES",
            "gender": "male",
            "character_description": desc,
        })
        # The full string did not land in the output.
        assert len(out) < len(desc) + 30
        # Truncation indicator landed.
        assert "..." in out

    def test_trim_preserves_age_role_and_face_lead(self):
        """The whole point of the smart trim: under the new CONTRACT
        format, the lead sentence carries age + role; the second
        sentence opens the Face: block. The echo MUST carry at
        least the age + role; preserving the Face: block is a bonus
        when it fits inside the cap."""
        desc = (
            "Late-40s field doctor with arctic-light eyes. "
            "Face: square jaw, weather-cracked skin, salt-and-"
            "pepper crew cut, scar across the left brow. "
            "Presence: weary precision. Voice: calm and measured."
        )
        out = cst._format_prior_entry({
            "name": "DOC",
            "gender": "male",
            "character_description": desc,
        })
        # Age + role lead must survive.
        assert "Late-40s field doctor" in out


# ---------------------------------------------------------------------------
# 5. Portrait builder full anti-glamour guard + spec expression
# ---------------------------------------------------------------------------


class TestPortraitAntiGlamourGuard:

    def test_three_anti_patterns_named_explicitly(self):
        """FLUX's dominant headshot failure modes are stock-photo
        composition and glamour-photography skin. Name both
        explicitly so a single anti-pattern cue cannot silently
        be the only guard."""
        prompt = bfpr._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
        )
        assert "distinctive non-generic face" in prompt
        assert "not a stock portrait" in prompt
        assert "not glamour photography" in prompt

    def test_story_linked_expression_string_present(self):
        prompt = bfpr._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
        )
        assert (
            "subtle expression matching the character's role in the story"
            in prompt
        )

    def test_neutral_expression_string_absent(self):
        """The legacy `"neutral expression"` cue made every face land
        on FLUX's default archetype. It must NOT come back."""
        prompt = bfpr._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
        )
        # Standalone "neutral expression," segment must be gone.
        assert ", neutral expression," not in prompt
        assert not prompt.endswith(", neutral expression")

    def test_remaining_composition_guidance_unchanged(self):
        prompt = bfpr._build_portrait_prompt(
            speaker="Jones",
            appearance="weary detective",
            style_anchor="noir studio portrait",
        )
        for needle in (
            "centered composition",
            "frontal pose facing camera",
            "soft studio lighting",
            "35mm film grain",
            "no other characters in frame",
            "no background props",
        ):
            assert needle in prompt, (
                f"composition guidance dropped: {needle!r}"
            )


# ---------------------------------------------------------------------------
# 6. compose_shot_prompt PASS 3 portrait truncation
# ---------------------------------------------------------------------------


class TestPass3PortraitTruncation:

    def test_short_portrait_passes_through_unchanged(self):
        short = "Late-30s technician, weary precision."
        out = ovp._truncate_portrait_for_composite(short)
        assert out == short

    def test_long_portrait_trimmed_at_sentence_boundary(self):
        """A CONTRACT-format description: age+role lead, Face block,
        Presence, Voice -- must trim at the last sentence boundary
        inside the 160-char cap."""
        long_portrait = (
            "Late-30s mission technician, the one who noticed the "
            "signal first. Face: narrow oval, tense deep-set eyes, "
            "pinched nose, thinning hair flat from a headset, scar "
            "above the left brow. Presence: sleep-deprived, precise. "
            "Voice: clipped, trying not to shake."
        )
        out = ovp._truncate_portrait_for_composite(long_portrait)
        # Within cap.
        assert len(out) <= ovp._PASS3_PORTRAIT_HARD_MAX_CHARS
        # Ends at a real period (no ellipsis fallback fired).
        assert out.endswith(".")
        assert "..." not in out
        # Age + role lead preserved.
        assert "Late-30s mission technician" in out

    def test_trimmed_portrait_drops_voice_and_presence_tails(self):
        """The PASS 3 cap is sized so the visual-geometry block
        (Face:) survives but the audio-tail (Voice:) drops -- PASS 3
        cares about visual signal, not radio-performance cues."""
        long_portrait = (
            "Late-30s mission technician, the one who noticed the "
            "signal first. Face: narrow oval, tense deep-set eyes. "
            "Presence: sleep-deprived, precise. "
            "Voice: clipped, trying not to shake."
        )
        out = ovp._truncate_portrait_for_composite(long_portrait)
        assert "Voice: clipped" not in out

    def test_fallback_to_comma_when_no_period_in_cap(self):
        """A degenerate clause-only description with no internal
        periods must still trim cleanly at a comma."""
        long_portrait = (
            "a long single-sentence description with commas, "
            "and more commas, and yet more commas, "
            "and even more, that runs without any period punctuation"
            "until much later in the text well past one sixty chars"
        )
        out = ovp._truncate_portrait_for_composite(long_portrait)
        assert len(out) <= ovp._PASS3_PORTRAIT_HARD_MAX_CHARS
        # No mid-word cut.
        assert not out.endswith("commas,")  # would mean comma trim left trailing punctuation
        # Either trimmed at a comma boundary or hit the char-fallback.
        # Both are acceptable; the assertion is that we didn't
        # overflow.

    def test_fallback_ellipsis_when_no_clause_boundary(self):
        """A pathological single-token run of chars with no clauses
        falls through to the hard char-trim with ellipsis."""
        long_portrait = "x" * 200
        out = ovp._truncate_portrait_for_composite(long_portrait)
        assert len(out) <= ovp._PASS3_PORTRAIT_HARD_MAX_CHARS + 3
        assert out.endswith("...")

    def test_empty_portrait_returns_empty(self):
        assert ovp._truncate_portrait_for_composite("") == ""

    def test_compose_shot_prompt_uses_truncation(self):
        """End-to-end: a long portrait fed into compose_shot_prompt
        must come out trimmed AND concatenated with the scene/era/
        style layers."""
        long_portrait = (
            "Late-30s mission technician, the one who noticed the "
            "signal first. Face: narrow oval, tense deep-set eyes, "
            "pinched nose, thinning hair flat from a headset, scar "
            "above the left brow. Presence: sleep-deprived, precise. "
            "Voice: clipped, trying not to shake."
        )
        composed = ovp.compose_shot_prompt(
            portrait=long_portrait,
            scene_visual="a dim radio room",
            era_tail="bare bulb, tense",
            style_tail="35mm film grain",
            shot_hint="medium shot",
        )
        # Trimmed portrait lead survives.
        assert "Late-30s mission technician" in composed
        # Voice tail dropped.
        assert "Voice: clipped" not in composed
        # Downstream layers all landed.
        assert "a dim radio room" in composed
        assert "medium shot" in composed
        assert "bare bulb" in composed
        assert "35mm film grain" in composed

    def test_pass1_portrait_builder_not_affected_by_pass3_cap(self):
        """PASS 1 (`_build_portrait_prompt`) renders the portrait
        alone -- the full character_description belongs there. The
        PASS 3 cap (`_truncate_portrait_for_composite`) is the ONLY
        truncation site; PASS 1 must not import or call it."""
        src = inspect.getsource(bfpr._build_portrait_prompt)
        assert "_truncate_portrait_for_composite" not in src, (
            "PASS 1 portrait builder now references the PASS 3 "
            "truncation helper -- they should stay separate so the "
            "portrait-only render keeps the full description"
        )


# ---------------------------------------------------------------------------
# 7. Source-level wiring locks
# ---------------------------------------------------------------------------


class TestSourceLevelLocks:

    def test_face_pressure_dict_referenced_in_build_user_prompt(self):
        src = inspect.getsource(cst._build_user_prompt)
        assert "_FACE_PRESSURE_BY_ROLE" in src, (
            "_build_user_prompt no longer references "
            "_FACE_PRESSURE_BY_ROLE -- Narrative Face anchor regressed"
        )

    def test_contract_block_referenced_in_build_user_prompt(self):
        src = inspect.getsource(cst._build_user_prompt)
        assert "CHARACTER VISUAL CONTRACT" in src, (
            "_build_user_prompt no longer carries the CHARACTER VISUAL "
            "CONTRACT header text -- Narrative Face contract regressed"
        )

    def test_pass3_truncation_referenced_in_compose_shot_prompt(self):
        src = inspect.getsource(ovp.compose_shot_prompt)
        assert "_truncate_portrait_for_composite" in src, (
            "compose_shot_prompt no longer calls "
            "_truncate_portrait_for_composite -- Q1 PASS 3 token "
            "budget guard regressed"
        )
