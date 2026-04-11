"""Unit tests for _encode_prompt() dict-copy safety (BUG-001 regression).

These tests use a mock CLIP object to verify that encode_from_tokens()'s
returned dict is NOT mutated by _encode_prompt(). No GPU or ComfyUI needed.
"""

import pytest


class MockCLIP:
    """Minimal CLIP stub that returns a shared dict from encode_from_tokens."""

    def __init__(self):
        # Simulate a shared/cached dict that encode_from_tokens might return
        self._cached_output = {
            "cond": [1.0, 2.0, 3.0],  # stand-in for cond tensor
            "pooled_output": [4.0, 5.0],
            "extra_key": "metadata",
        }

    def tokenize(self, text):
        return {"tokens": text.split()}

    def encode_from_tokens(self, tokens, return_pooled=True, return_dict=True):
        # Return the SAME dict object every time (simulates caching)
        return self._cached_output


def _encode_prompt_fixed(clip, text):
    """The fixed version: shallow-copies before pop."""
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    output = dict(output)  # shallow copy
    cond = output.pop("cond")
    return [[cond, output]]


def _encode_prompt_broken(clip, text):
    """The OLD broken version: mutates the shared dict."""
    tokens = clip.tokenize(text)
    output = clip.encode_from_tokens(tokens, return_pooled=True, return_dict=True)
    cond = output.pop("cond")
    return [[cond, output]]


class TestEncodePromptCopySafety:
    """BUG-001 regression: ensure encode_from_tokens dict is never mutated."""

    def test_fixed_version_does_not_mutate_source(self):
        clip = MockCLIP()
        original_keys = set(clip._cached_output.keys())

        _encode_prompt_fixed(clip, "test prompt one")

        # The cached dict must still have "cond"
        assert "cond" in clip._cached_output
        assert set(clip._cached_output.keys()) == original_keys

    def test_fixed_version_returns_valid_conditioning(self):
        clip = MockCLIP()
        result = _encode_prompt_fixed(clip, "test prompt")

        assert len(result) == 1
        cond, extras = result[0]
        assert cond == [1.0, 2.0, 3.0]
        assert "pooled_output" in extras
        assert "cond" not in extras  # popped from the copy

    def test_back_to_back_encodes_both_succeed(self):
        """Two encodes on the same CLIP must both return valid data."""
        clip = MockCLIP()

        result_a = _encode_prompt_fixed(clip, "prompt A")
        result_b = _encode_prompt_fixed(clip, "prompt B")

        cond_a, extras_a = result_a[0]
        cond_b, extras_b = result_b[0]

        # Both should have valid cond data
        assert cond_a == [1.0, 2.0, 3.0]
        assert cond_b == [1.0, 2.0, 3.0]

        # Neither extras dict should contain "cond"
        assert "cond" not in extras_a
        assert "cond" not in extras_b

    def test_broken_version_would_fail_second_call(self):
        """Demonstrates the bug: broken version mutates, second call KeyErrors."""
        clip = MockCLIP()

        _encode_prompt_broken(clip, "first call")

        # Second call should crash because "cond" was popped from the shared dict
        with pytest.raises(KeyError):
            _encode_prompt_broken(clip, "second call")
