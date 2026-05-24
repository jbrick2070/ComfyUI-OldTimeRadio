"""BUG-LOCAL-262 -- system-role normalization regression.

Gemma-2's chat template hard-rejects the `system` role (its Jinja
template contains a literal raise_exception("System role not
supported")). The OTR writer generate path passes a message list
containing a {"role": "system", ...} entry straight into
tokenizer.apply_chat_template, so the template throws and the run
aborts (first surfaced in the style picker's chooser pass).

The fix adds two helpers to nodes/_otr_loader_backends.py:

  tokenizer_supports_system_role(tokenizer) -> bool
      Probes the tokenizer's chat template with a 2-message list.
      Returns False if the render raises (system role unsupported).

  normalize_messages_for_tokenizer(tokenizer, messages) -> list
      No-op when the system role is supported. Otherwise folds each
      system message's content into the first following user turn
      and drops the system entries -- the standard Gemma pattern.

These tests use fake tokenizers (a strict one that rejects any
system message and a permissive one that accepts everything) so the
suite runs offline with no model download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_loader_backends as backends  # noqa: E402


# ---------------------------------------------------------------------------
# Fake tokenizers
# ---------------------------------------------------------------------------


class _SystemRejectingTokenizer:
    """Stand-in for a Gemma-2-style tokenizer: apply_chat_template
    raises on any message list that contains a system role.
    """

    def __init__(self) -> None:
        self.last_messages: list[dict] | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = list(messages)
        if any(m.get("role") == "system" for m in messages):
            raise ValueError("System role not supported")
        return "|".join(
            f"{m.get('role')}:{m.get('content')}" for m in messages
        )


class _PermissiveTokenizer:
    """Stand-in for a Mistral-style tokenizer: accepts a system role."""

    def __init__(self) -> None:
        self.last_messages: list[dict] | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = list(messages)
        return "|".join(
            f"{m.get('role')}:{m.get('content')}" for m in messages
        )


# ---------------------------------------------------------------------------
# Capability probe
# ---------------------------------------------------------------------------


def test_probe_detects_system_role_rejection() -> None:
    tok = _SystemRejectingTokenizer()
    assert backends.tokenizer_supports_system_role(tok) is False


def test_probe_detects_system_role_support() -> None:
    tok = _PermissiveTokenizer()
    assert backends.tokenizer_supports_system_role(tok) is True


# ---------------------------------------------------------------------------
# normalize_messages_for_tokenizer
# ---------------------------------------------------------------------------


def test_fold_drops_system_role_and_prepends_to_first_user() -> None:
    """A system-rejecting tokenizer must, after normalization, receive
    a list with NO system role and the system text prepended to the
    first user message. apply_chat_template must then NOT raise.
    """
    tok = _SystemRejectingTokenizer()
    messages = [
        {"role": "system", "content": "You are a strict editor."},
        {"role": "user", "content": "Pick the best descriptor."},
    ]
    normalized = backends.normalize_messages_for_tokenizer(tok, messages)

    assert all(m.get("role") != "system" for m in normalized), (
        f"normalized list still contains a system role: {normalized}"
    )
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == (
        "You are a strict editor.\n\nPick the best descriptor."
    )
    # The whole point: the normalized list must render cleanly.
    rendered = tok.apply_chat_template(
        normalized, tokenize=False, add_generation_prompt=True,
    )
    assert "You are a strict editor." in rendered
    assert "Pick the best descriptor." in rendered


def test_no_op_when_tokenizer_accepts_system_role() -> None:
    """A permissive tokenizer must get the message list back
    unchanged -- same object, no folding.
    """
    tok = _PermissiveTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    normalized = backends.normalize_messages_for_tokenizer(tok, messages)
    assert normalized is messages, (
        "permissive tokenizer should get the original list back "
        "unchanged"
    )


def test_fold_concatenates_multiple_system_messages_in_order() -> None:
    tok = _SystemRejectingTokenizer()
    messages = [
        {"role": "system", "content": "first"},
        {"role": "system", "content": "second"},
        {"role": "user", "content": "question"},
    ]
    normalized = backends.normalize_messages_for_tokenizer(tok, messages)
    assert len(normalized) == 1
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == "first\n\nsecond\n\nquestion"


def test_fold_converts_orphan_system_to_user_when_no_user_follows() -> None:
    """A system message with no following user message must be
    converted to a user message so its content is not lost.
    """
    tok = _SystemRejectingTokenizer()
    messages = [
        {"role": "system", "content": "only a system message"},
    ]
    normalized = backends.normalize_messages_for_tokenizer(tok, messages)
    assert len(normalized) == 1
    assert normalized[0]["role"] == "user"
    assert normalized[0]["content"] == "only a system message"
    # Must render without raising.
    tok.apply_chat_template(normalized, tokenize=False)


def test_fold_preserves_assistant_turns_after_first_user() -> None:
    """Folding only touches system messages and the first user turn;
    assistant turns and later user turns pass through untouched.
    """
    tok = _SystemRejectingTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    normalized = backends.normalize_messages_for_tokenizer(tok, messages)
    roles = [m["role"] for m in normalized]
    assert roles == ["user", "assistant", "user"]
    assert normalized[0]["content"] == "sys\n\nu1"
    assert normalized[1]["content"] == "a1"
    assert normalized[2]["content"] == "u2"


def test_input_list_is_not_mutated() -> None:
    tok = _SystemRejectingTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    original = [dict(m) for m in messages]
    backends.normalize_messages_for_tokenizer(tok, messages)
    assert messages == original, (
        "normalize_messages_for_tokenizer mutated its input list"
    )


# ---------------------------------------------------------------------------
# encode_messages_for_row routes through the normalizer
# ---------------------------------------------------------------------------


def test_encode_messages_for_row_folds_system_for_rejecting_tokenizer() -> None:
    """The transformers_default dispatch branch must normalize system
    messages so the dispatch path and the live writer path agree.
    """
    from dataclasses import dataclass

    @dataclass
    class _Row:
        chat_template_kind: str = "transformers_default"
        repo_id: str = "google/gemma-2-2b-it"

    tok = _SystemRejectingTokenizer()
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    # Must not raise -- the branch folds the system role before
    # calling apply_chat_template.
    backends.encode_messages_for_row(tok, messages, _Row())
    assert tok.last_messages is not None
    assert all(m.get("role") != "system" for m in tok.last_messages)


# ---------------------------------------------------------------------------
# Public surface
# ---------------------------------------------------------------------------


def test_helpers_are_exported() -> None:
    for name in (
        "tokenizer_supports_system_role",
        "normalize_messages_for_tokenizer",
    ):
        assert name in backends.__all__, (
            f"{name} missing from _otr_loader_backends.__all__"
        )


def test_gemma_2_2b_it_row_is_a_clean_technical_pick() -> None:
    """BUG-LOCAL-262: google/gemma-2-2b-it must have a catalog row
    with transformers_default chat template and an 8192 context
    window matching HARD_VRAM_CONTEXT_LIMIT so check_context_window
    does not trip.
    """
    from nodes import _otr_model_catalog as catalog

    rows = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    row = rows.get("google/gemma-2-2b-it")
    assert row is not None, (
        "google/gemma-2-2b-it missing from CURATED_LLM_MODELS"
    )
    assert row.chat_template_kind == "transformers_default"
    assert row.context_window >= catalog.HARD_VRAM_CONTEXT_LIMIT
    # The context-window precondition must not trip for this row.
    backends.check_context_window(row)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
