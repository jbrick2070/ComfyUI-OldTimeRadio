"""Sprint 8.6: HuMo lip-sync brief-rewire tests.

`_build_pos_prompt` now reads `meta.atmosphere_line` via the
canonical `_read_brief_field` reader and inserts it BEFORE the v1
lighting/atmosphere terms in the HuMo positive prompt. Order:

  speaker_desc -> atmosphere_line -> lighting -> _DEFAULT_POS_SUFFIX

The v1 lighting/atmosphere path is preserved unchanged. v1-era
ledgers and the v2 failure sentinel return empty atmosphere_line
and the function falls through to the legacy lighting-only or
suffix-only paths.
"""
from __future__ import annotations

import inspect

import pytest

from nodes import batch_humo_render as bhr


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _meta_v2_ok(
    *,
    atmosphere_line: str = "smoke and sweat under a swinging bulb.",
    lighting_terms: list | None = None,
    atmosphere_terms: list | None = None,
) -> dict:
    """v2 _success_delta-shaped meta with both signals present."""
    return {
        "story_brief":        "a dim radio room",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting":    [],
            "lighting":   lighting_terms or ["bare bulb", "harsh shadow"],
            "atmosphere": atmosphere_terms or ["tense"],
        },
        "atmosphere_line": atmosphere_line,
    }


def _meta_v1_legacy() -> dict:
    """Pre-Sprint-8.1 meta -- no v2 keys."""
    return {
        "story_brief":        "a dim radio room",
        "story_brief_status": "ok",
        "story_brief_terms": {
            "setting":    [],
            "lighting":   ["bare bulb"],
            "atmosphere": ["tense"],
        },
    }


def _meta_v2_failure_sentinel() -> dict:
    return {
        "story_brief":        "",
        "story_brief_status": "failed",
        "story_brief_terms": {
            "setting":    [],
            "lighting":   [],
            "atmosphere": [],
        },
        "atmosphere_line": "",
    }


def _cast() -> list[dict]:
    return [
        {
            "name": "JONES",
            "character_description":
                "tall detective with weathered face, distinctive scar",
        },
    ]


def _line() -> dict:
    return {"speaker_role": "character", "char_id": "JONES", "text": "hello"}


# ---------------------------------------------------------------------------
# 1. atmosphere_line lands in the prompt when supplied
# ---------------------------------------------------------------------------


class TestAtmosphereLineLands:

    def test_atmosphere_line_appears(self):
        meta = _meta_v2_ok(atmosphere_line="ATMOSPHERE_CANARY_PHRASE.")
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        assert "ATMOSPHERE_CANARY_PHRASE." in prompt

    def test_atmosphere_line_alongside_v1_lighting(self):
        meta = _meta_v2_ok(
            atmosphere_line="V2_LINE",
            lighting_terms=["V1_LIGHT"],
            atmosphere_terms=["V1_ATMO"],
        )
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        assert "V2_LINE" in prompt
        assert "V1_LIGHT" in prompt
        assert "V1_ATMO" in prompt


# ---------------------------------------------------------------------------
# 2. Ordering: speaker_desc -> atmosphere_line -> lighting -> suffix
# ---------------------------------------------------------------------------


class TestOrdering:

    def test_v2_line_precedes_v1_lighting(self):
        meta = _meta_v2_ok(
            atmosphere_line="V2_CANARY",
            lighting_terms=["V1_CANARY"],
        )
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        i_v2 = prompt.index("V2_CANARY")
        i_v1 = prompt.index("V1_CANARY")
        assert i_v2 < i_v1, (
            "atmosphere_line must precede v1 lighting terms"
        )

    def test_speaker_desc_still_leads(self):
        meta = _meta_v2_ok(atmosphere_line="ATMO")
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        first_segment = prompt.split(",")[0].strip()
        assert "tall detective" in first_segment

    def test_suffix_still_trails(self):
        meta = _meta_v2_ok(atmosphere_line="ATMO")
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        assert prompt.endswith(bhr._DEFAULT_POS_SUFFIX)


# ---------------------------------------------------------------------------
# 3. Fallback paths
# ---------------------------------------------------------------------------


class TestFallbackPaths:

    def test_v1_era_ledger_byte_identical_to_pre_8_6(self):
        """No atmosphere_line key. Prompt must match the pre-Sprint-8.6
        legacy composition (speaker_desc, lighting, suffix)."""
        meta = _meta_v1_legacy()
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        # speaker_desc + v1 lighting + suffix; no v2 atmosphere_line.
        assert "tall detective" in prompt
        assert "bare bulb" in prompt
        assert prompt.endswith(bhr._DEFAULT_POS_SUFFIX)

    def test_v2_failure_sentinel_byte_identical_to_pure_legacy(self):
        """v2 sentinel sets atmosphere_line=''. Prompt must match the
        all-signals-empty path."""
        meta = _meta_v2_failure_sentinel()
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        # Only speaker_desc + suffix; no lighting (helper returns '');
        # no atmosphere_line.
        expected = (
            "tall detective with weathered face, distinctive scar"
            f", {bhr._DEFAULT_POS_SUFFIX}"
        )
        assert prompt == expected

    def test_non_string_atmosphere_line_falls_through(self):
        meta = _meta_v1_legacy()
        meta["atmosphere_line"] = ["not", "a", "string"]  # type: ignore[assignment]
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        # Non-string atmosphere_line treated as empty.
        assert "not" not in prompt[:50]  # not in early prompt
        # v1 path still works.
        assert "bare bulb" in prompt

    def test_whitespace_atmosphere_line_treated_as_empty(self):
        meta = _meta_v1_legacy()
        meta["atmosphere_line"] = "   \t  "
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        # Match the no-atmosphere_line path.
        meta_no_atmo = _meta_v1_legacy()
        prompt_no_atmo = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta_no_atmo,
        )
        assert prompt == prompt_no_atmo

    def test_none_meta_still_renders(self):
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), None,
        )
        # speaker_desc + suffix only.
        assert "tall detective" in prompt
        assert prompt.endswith(bhr._DEFAULT_POS_SUFFIX)


# ---------------------------------------------------------------------------
# 4. v2 atmosphere_line without v1 lighting -- new viable path
# ---------------------------------------------------------------------------


class TestV2OnlyPath:

    def test_atmosphere_line_alone_renders(self):
        """v2 producer ran but the v1 helper returned empty (lighting +
        atmosphere term arrays were empty). atmosphere_line still
        contributes -- new viable path."""
        meta = _meta_v2_ok(
            atmosphere_line="lone v2 signal",
            lighting_terms=[],
            atmosphere_terms=[],
        )
        prompt = bhr._build_pos_prompt(
            "JONES", _line(), _cast(), meta,
        )
        assert "lone v2 signal" in prompt
        # Suffix still rides.
        assert prompt.endswith(bhr._DEFAULT_POS_SUFFIX)


# ---------------------------------------------------------------------------
# 5. Source-level wiring lock
# ---------------------------------------------------------------------------


class TestReaderHelperUsed:

    def test_function_source_uses_brief_reader(self):
        src = inspect.getsource(bhr._build_pos_prompt)
        assert "_read_brief_field" in src, (
            "_build_pos_prompt no longer uses _read_brief_field -- "
            "Sprint 8.6 wiring regressed"
        )

    def test_function_reads_atmosphere_line_key(self):
        src = inspect.getsource(bhr._build_pos_prompt)
        assert '"atmosphere_line"' in src, (
            "_build_pos_prompt no longer reads meta.atmosphere_line "
            "-- Sprint 8.6 wiring regressed"
        )

    def test_function_still_uses_v1_lighting_helper(self):
        """The v1 lighting helper must STAY in place so dual-signal
        rendering works -- the rewire is additive, not a replacement."""
        src = inspect.getsource(bhr._build_pos_prompt)
        assert "get_story_brief_lighting" in src, (
            "_build_pos_prompt dropped the v1 get_story_brief_lighting "
            "call -- Sprint 8.6 was meant to be additive, not "
            "replacement"
        )
