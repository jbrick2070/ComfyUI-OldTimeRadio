"""Sprint 8.1: brief reader (_otr_brief_reader._read_brief_field) tests.

Locks the contract every v2 consumer relies on:

  1. Flat-key read returns the v2 top-level value (A1 happy path).
  2. Dotted-path read traverses the existing v1 nested object
     `story_brief_terms.atmosphere` (forward-compat per decision B).
  3. Caller-supplied default returns when the segment is absent.
  4. Caller-supplied default returns when an intermediate value is
     not a dict (e.g. typo points into a list).
  5. Caller-supplied default returns when meta is empty / wrong-type
     / None.
  6. Parent dict containing a `meta` sub-key normalizes correctly.
  7. Brief-shaped meta dict (carrying story_brief_status directly)
     reads as-is without wrapping.
  8. Terminal value of `None` reads as `default` (a producer
     regression that explicitly stamps None reads as no-signal, not
     literal None).
  9. Empty / whitespace / empty-segment dotted_path raises
     ValueError -- typo catcher.
"""
from __future__ import annotations

import pytest

from nodes import _otr_brief_reader as br


# ---------------------------------------------------------------------------
# Fixtures: meta deltas the producer would actually emit
# ---------------------------------------------------------------------------


def _v2_success_meta() -> dict:
    """Minimal meta dict shaped like a Sprint 8.1 _success_delta."""
    return {
        "story_brief":               "a dim room under a bare bulb",
        "story_brief_status":        "ok",
        "story_brief_error":         None,
        "story_brief_model":         "tech/model-id",
        "story_brief_prompt_version": "v2",
        "story_brief_source":        "post_script_reflection",
        "story_brief_char_count":    28,
        "story_brief_terms": {
            "setting":    ["room", "table"],
            "lighting":   ["bare bulb"],
            "atmosphere": ["tense", "smoky"],
        },
        "music_mood_terms": ["tense", "sombre", "uneasy"],
        "visual_palette":   ["amber", "smoke-grey"],
        "key_objects":      ["table", "lamp"],
        "tempo_hint":       "slow",
        "atmosphere_line":  "smoke and sweat under a bare bulb.",
    }


def _v2_failure_sentinel_meta() -> dict:
    """Meta shaped like the v2 _failure_sentinel."""
    return {
        "story_brief":               "",
        "story_brief_status":        "failed",
        "story_brief_error":         "json_parse_failed",
        "story_brief_model":         "tech/model-id",
        "story_brief_prompt_version": "v2",
        "story_brief_source":        "post_script_reflection",
        "story_brief_char_count":    0,
        "story_brief_terms": {
            "setting":    [],
            "lighting":   [],
            "atmosphere": [],
        },
        "music_mood_terms": [],
        "visual_palette":   [],
        "key_objects":      [],
        "tempo_hint":       "",
        "atmosphere_line":  "",
    }


def _v1_legacy_meta() -> dict:
    """Pre-Sprint-8.1 meta dict -- v2 fields absent entirely. The
    reader must drop through to the caller-supplied default on every
    v2 read so v1 consumers keep working through the rewire."""
    return {
        "story_brief":               "a dim room under a bare bulb",
        "story_brief_status":        "ok",
        "story_brief_error":         None,
        "story_brief_model":         "tech/model-id",
        "story_brief_prompt_version": "v1",
        "story_brief_source":        "post_script_reflection",
        "story_brief_char_count":    28,
        "story_brief_terms": {
            "setting":    ["room"],
            "lighting":   ["bare bulb"],
            "atmosphere": ["tense"],
        },
    }


# ---------------------------------------------------------------------------
# 1. Flat-key read on the v2 happy path
# ---------------------------------------------------------------------------


class TestFlatKeyV2:

    def test_reads_list_field(self):
        meta = _v2_success_meta()
        out = br._read_brief_field(meta, "music_mood_terms", default=[])
        assert out == ["tense", "sombre", "uneasy"]

    def test_reads_string_field(self):
        meta = _v2_success_meta()
        out = br._read_brief_field(meta, "atmosphere_line", default="")
        assert out == "smoke and sweat under a bare bulb."

    def test_reads_v1_prose_field(self):
        meta = _v2_success_meta()
        out = br._read_brief_field(meta, "story_brief", default="")
        assert out == "a dim room under a bare bulb"


# ---------------------------------------------------------------------------
# 2. Dotted-path read on the v1 nested object (forward-compat per B)
# ---------------------------------------------------------------------------


class TestDottedPath:

    def test_reads_nested_v1_atmosphere(self):
        meta = _v2_success_meta()
        out = br._read_brief_field(
            meta, "story_brief_terms.atmosphere", default=[],
        )
        assert out == ["tense", "smoky"]

    def test_reads_nested_v1_setting(self):
        meta = _v2_success_meta()
        out = br._read_brief_field(
            meta, "story_brief_terms.setting", default=[],
        )
        assert out == ["room", "table"]

    def test_reads_nested_v1_lighting(self):
        meta = _v2_success_meta()
        out = br._read_brief_field(
            meta, "story_brief_terms.lighting", default=[],
        )
        assert out == ["bare bulb"]


# ---------------------------------------------------------------------------
# 3-5. Default fallback paths
# ---------------------------------------------------------------------------


class TestDefaultFallback:

    def test_default_when_v2_field_absent_on_v1_ledger(self):
        """v1-era meta has no v2 fields; every v2 read returns default."""
        meta = _v1_legacy_meta()
        assert br._read_brief_field(
            meta, "music_mood_terms", default=[],
        ) == []
        assert br._read_brief_field(
            meta, "atmosphere_line", default="",
        ) == ""
        assert br._read_brief_field(
            meta, "tempo_hint", default="default-tempo",
        ) == "default-tempo"

    def test_default_when_failure_sentinel(self):
        """v2 failure-sentinel stamps safe-empty values; reader returns
        them as-is (the value present IS '', not absent)."""
        meta = _v2_failure_sentinel_meta()
        # The field is present but empty -- the reader returns the
        # stamped value, NOT the default. Consumers treat "" / [] as
        # "fall through to v1 path" via truthiness on the return.
        assert br._read_brief_field(
            meta, "music_mood_terms", default=["sentinel-default"],
        ) == []
        assert br._read_brief_field(
            meta, "atmosphere_line", default="sentinel-default",
        ) == ""

    def test_default_when_intermediate_is_not_a_dict(self):
        """Traversing into a non-dict (e.g. typo asks for a list-shaped
        v1 field's child) returns default rather than raising."""
        meta = _v2_success_meta()
        # `setting_terms` doesn't exist as a top-level key on the v2
        # delta (the producer nests it under story_brief_terms), but
        # `story_brief` IS a top-level string -- a dotted-path read
        # that tries to traverse INTO it must return default.
        out = br._read_brief_field(
            meta, "story_brief.something", default="fallback",
        )
        assert out == "fallback"

    def test_default_when_meta_is_empty_dict(self):
        out = br._read_brief_field({}, "music_mood_terms", default=[])
        assert out == []

    def test_default_when_meta_is_not_a_dict(self):
        assert br._read_brief_field(
            None, "music_mood_terms", default=["x"],
        ) == ["x"]
        assert br._read_brief_field(
            "not a dict", "music_mood_terms", default=["x"],
        ) == ["x"]
        assert br._read_brief_field(
            123, "music_mood_terms", default=["x"],
        ) == ["x"]


# ---------------------------------------------------------------------------
# 6. Parent dict carrying a `meta` sub-key normalizes
# ---------------------------------------------------------------------------


class TestParentDictNormalization:

    def test_parent_with_meta_subkey_reads_through(self):
        """A consumer that hands in the parent ledger / script_json
        (carrying `meta` as a sub-key) reads through to the brief
        fields just like a bare meta dict."""
        parent = {
            "cast": [{"name": "Jones"}],
            "lines": [],
            "meta": _v2_success_meta(),
        }
        out = br._read_brief_field(
            parent, "music_mood_terms", default=[],
        )
        assert out == ["tense", "sombre", "uneasy"]

    def test_bare_meta_reads_as_is(self):
        meta = _v2_success_meta()
        out = br._read_brief_field(meta, "music_mood_terms", default=[])
        assert out == ["tense", "sombre", "uneasy"]


# ---------------------------------------------------------------------------
# 7. None terminal value reads as default
# ---------------------------------------------------------------------------


class TestNoneTerminal:

    def test_terminal_none_returns_default(self):
        """If the meta carries `field: None` explicitly (which the
        production producer never does, but a future regression
        might), the reader returns the caller default -- a v2
        consumer should never see a literal None where it expected
        a list or string."""
        meta = {
            "story_brief_status": "ok",
            "music_mood_terms":   None,
            "atmosphere_line":    None,
        }
        assert br._read_brief_field(
            meta, "music_mood_terms", default=["fallback"],
        ) == ["fallback"]
        assert br._read_brief_field(
            meta, "atmosphere_line", default="fallback",
        ) == "fallback"


# ---------------------------------------------------------------------------
# 8. Typo catcher -- empty / whitespace / empty-segment paths raise
# ---------------------------------------------------------------------------


class TestDottedPathTypoGuard:

    def test_empty_path_raises(self):
        with pytest.raises(ValueError):
            br._read_brief_field({}, "", default=None)

    def test_whitespace_path_raises(self):
        with pytest.raises(ValueError):
            br._read_brief_field({}, "   ", default=None)

    def test_leading_dot_raises(self):
        with pytest.raises(ValueError):
            br._read_brief_field({}, ".music_mood_terms", default=None)

    def test_trailing_dot_raises(self):
        with pytest.raises(ValueError):
            br._read_brief_field({}, "music_mood_terms.", default=None)

    def test_double_dot_raises(self):
        with pytest.raises(ValueError):
            br._read_brief_field(
                {}, "story_brief_terms..atmosphere", default=None,
            )

    def test_non_string_path_raises(self):
        with pytest.raises(ValueError):
            br._read_brief_field({}, None, default=None)  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            br._read_brief_field({}, 42, default=None)  # type: ignore[arg-type]
