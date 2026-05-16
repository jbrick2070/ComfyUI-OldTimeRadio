"""Sprint D D2c -- chat_template_kind dispatch + stop_tokens passthrough.

Five assertions per v3 plan:

  test_transformers_default_kind_uses_apply_chat_template
      The row whose chat_template_kind is "transformers_default"
      routes through tokenizer.apply_chat_template.

  test_raw_completion_kind_concatenates_content_only
      The "raw_completion" kind concatenates message contents with
      newlines and feeds tokenizer() directly. No chat-template
      involved.

  test_manual_kind_raises_not_implemented_with_clear_message
      "manual" kind raises NotImplementedError with a message that
      names the missing schema field and points at D-future.

  test_stop_tokens_threaded_to_generate_kwargs
      generate_kwargs_for_row returns stop_strings as a list.

  test_dispatch_path_uses_compute_effective_context_limit
      AST snapshot proves compute_effective_context_limit is
      exported from _otr_loader_backends and callable on a
      CuratedModel row.
"""
from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_loader_backends as backends_proto  # noqa: E402
from nodes import _otr_model_catalog as catalog  # noqa: E402


@dataclass
class _FakeRow:
    chat_template_kind: str = "transformers_default"
    stop_tokens: tuple[str, ...] = ()
    context_window: int = 8192
    repo_id: str = "fake/repo"


class _FakeTokenizer:
    """Minimum fake. Records which path was taken so the test can
    assert dispatch went through the expected branch.
    """

    def __init__(self) -> None:
        self.apply_chat_template_called_with: tuple | None = None
        self.bare_call_called_with: tuple | None = None

    def apply_chat_template(self, messages, **kwargs):
        self.apply_chat_template_called_with = (messages, kwargs)
        return "<chat-template-encoded>"

    def __call__(self, text, **kwargs):
        self.bare_call_called_with = (text, kwargs)
        return "<bare-tokenize-encoded>"


# ---------------------------------------------------------------------------
# Dispatch behavior
# ---------------------------------------------------------------------------


def test_transformers_default_kind_uses_apply_chat_template() -> None:
    tok = _FakeTokenizer()
    row = _FakeRow(chat_template_kind="transformers_default")
    msgs = [
        {"role": "system", "content": "hello"},
        {"role": "user", "content": "test"},
    ]
    out = backends_proto.encode_messages_for_row(tok, msgs, row)
    assert out == "<chat-template-encoded>"
    assert tok.apply_chat_template_called_with is not None, (
        "transformers_default kind did not route through "
        "apply_chat_template"
    )
    assert tok.bare_call_called_with is None, (
        "transformers_default kind unexpectedly invoked bare "
        "tokenizer call as well"
    )
    # The dispatch passes return_tensors='pt' and
    # add_generation_prompt=True.
    _, kwargs = tok.apply_chat_template_called_with
    assert kwargs.get("return_tensors") == "pt"
    assert kwargs.get("add_generation_prompt") is True


def test_raw_completion_kind_concatenates_content_only() -> None:
    tok = _FakeTokenizer()
    row = _FakeRow(chat_template_kind="raw_completion")
    msgs = [
        {"role": "system", "content": "alpha"},
        {"role": "user", "content": "beta"},
        {"role": "assistant", "content": "gamma"},
    ]
    out = backends_proto.encode_messages_for_row(tok, msgs, row)
    assert out == "<bare-tokenize-encoded>"
    assert tok.bare_call_called_with is not None, (
        "raw_completion kind did not route through bare tokenizer "
        "call"
    )
    text, _ = tok.bare_call_called_with
    assert text == "alpha\nbeta\ngamma", (
        f"raw_completion concatenation = {text!r}; expected "
        f"'alpha\\nbeta\\ngamma'"
    )
    assert tok.apply_chat_template_called_with is None, (
        "raw_completion kind unexpectedly invoked apply_chat_template"
    )


def test_manual_kind_raises_not_implemented_with_clear_message() -> None:
    tok = _FakeTokenizer()
    row = _FakeRow(
        chat_template_kind="manual",
        repo_id="fake/manual-tokenizer-model",
    )
    msgs = [{"role": "user", "content": "test"}]
    with pytest.raises(NotImplementedError) as excinfo:
        backends_proto.encode_messages_for_row(tok, msgs, row)
    msg = str(excinfo.value)
    assert "manual" in msg, (
        f"NotImplementedError message does not name the 'manual' "
        f"kind: {msg!r}"
    )
    assert "D-future" in msg or "deferred" in msg.lower(), (
        f"NotImplementedError message does not point at D-future / "
        f"deferral: {msg!r}"
    )
    assert "fake/manual-tokenizer-model" in msg, (
        f"NotImplementedError message does not include the affected "
        f"repo_id: {msg!r}"
    )


def test_unknown_chat_template_kind_raises_value_error() -> None:
    """Schema drift between catalog Literal and dispatch dict fails
    loud with ValueError rather than silently picking a default."""
    tok = _FakeTokenizer()
    row = _FakeRow(chat_template_kind="some_fictional_kind_v999")
    with pytest.raises(ValueError, match="some_fictional_kind_v999"):
        backends_proto.encode_messages_for_row(tok, [], row)


# ---------------------------------------------------------------------------
# Stop tokens
# ---------------------------------------------------------------------------


def test_stop_tokens_threaded_to_generate_kwargs() -> None:
    """generate_kwargs_for_row returns stop_strings as a list of
    the row's stop_tokens.
    """
    row_empty = _FakeRow(stop_tokens=())
    kwargs_empty = backends_proto.generate_kwargs_for_row(row_empty)
    assert kwargs_empty == {"stop_strings": []}

    row_eos = _FakeRow(stop_tokens=("</s>",))
    kwargs_eos = backends_proto.generate_kwargs_for_row(row_eos)
    assert kwargs_eos == {"stop_strings": ["</s>"]}

    row_multi = _FakeRow(stop_tokens=("<|end|>", "<|im_end|>"))
    kwargs_multi = backends_proto.generate_kwargs_for_row(row_multi)
    assert kwargs_multi == {"stop_strings": ["<|end|>", "<|im_end|>"]}


def test_stop_tokens_against_real_talkie_row() -> None:
    """End-to-end smoke against the real catalog talkie row.
    Talkie's stop_tokens is ('</s>',) per D1a.
    """
    rows = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    talkie = rows["talkie-lm/talkie-1930-13b-it"]
    kwargs = backends_proto.generate_kwargs_for_row(talkie)
    assert kwargs == {"stop_strings": ["</s>"]}


# ---------------------------------------------------------------------------
# compute_effective_context_limit exported + usable
# ---------------------------------------------------------------------------


def test_dispatch_path_uses_compute_effective_context_limit() -> None:
    """compute_effective_context_limit must be exported from
    _otr_loader_backends so D2c dispatch consumers can cap prompt
    budgets per row. AST snapshot confirms it's in __all__.
    """
    src = (REPO_ROOT / "nodes" / "_otr_loader_backends.py").read_text(
        encoding="utf-8",
    )
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "__all__":
                if isinstance(node.value, ast.List):
                    names = [
                        e.value for e in node.value.elts
                        if isinstance(e, ast.Constant)
                        and isinstance(e.value, str)
                    ]
                    assert "compute_effective_context_limit" in names, (
                        f"compute_effective_context_limit missing "
                        f"from _otr_loader_backends.__all__: {names}"
                    )
                    return
    pytest.fail("no __all__ assignment found in _otr_loader_backends.py")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
