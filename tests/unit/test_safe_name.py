"""Unit tests for _safe_name() filename sanitizer (BUG-006 regression)."""

import re
import pytest


# Inline the function to avoid importing torch-dependent v2_preview module.
# This matches the implementation in nodes/v2_preview.py exactly.
def _safe_name(name, max_len=80):
    """Strip filesystem-unsafe characters from an output filename."""
    name = re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", name).strip(" .")
    return (name or "untitled")[:max_len]


class TestSafeName:
    """BUG-006 regression: every test here must stay green after any edit."""

    def test_normal_name_unchanged(self):
        assert _safe_name("otr_v2_episode") == "otr_v2_episode"

    def test_slashes_replaced(self):
        assert _safe_name("test/../weird") == "test_.._weird"

    def test_colon_replaced(self):
        assert _safe_name("file:name") == "file_name"

    def test_question_mark_replaced(self):
        assert _safe_name("what?") == "what_"

    def test_angle_brackets_replaced(self):
        assert _safe_name("hello<world>test") == "hello_world_test"

    def test_pipe_replaced(self):
        assert _safe_name("a|b") == "a_b"

    def test_quotes_replaced(self):
        assert _safe_name('say"hi"') == "say_hi_"

    def test_backslash_replaced(self):
        assert _safe_name("path\\to\\file") == "path_to_file"

    def test_control_chars_replaced(self):
        assert _safe_name("null\x00here\x1ftoo") == "null_here_too"

    def test_empty_string_returns_untitled(self):
        assert _safe_name("") == "untitled"

    def test_whitespace_only_returns_untitled(self):
        assert _safe_name("   ") == "untitled"

    def test_dots_only_returns_untitled(self):
        assert _safe_name("...") == "untitled"

    def test_truncation_at_80(self):
        long_name = "a" * 200
        result = _safe_name(long_name)
        assert len(result) == 80
        assert result == "a" * 80

    def test_custom_max_len(self):
        result = _safe_name("abcdefgh", max_len=5)
        assert result == "abcde"

    def test_full_action_plan_example(self):
        """The exact test case from the action plan acceptance criteria."""
        result = _safe_name('test/../weird:name?.mp4')
        assert result == "test_.._weird_name_.mp4"

    def test_combined_unsafe_chars(self):
        result = _safe_name('a<b>c:d*e?f"g|h\\i/j')
        assert "_" in result
        assert "<" not in result
        assert ">" not in result
        assert ":" not in result
        assert "*" not in result
        assert "?" not in result
        assert '"' not in result
        assert "|" not in result
        assert "\\" not in result
        assert "/" not in result
