"""Sprint D D1c -- talkie dropdown surface + writer-widget routing.

Four structural assertions (no HF auth required, no GPU):

  test_talkie_in_creative_writing_model_dropdown_enum
      The writer's creative_writing_model dropdown enum (built by
      dropdown_choices() from CURATED_LLM_MODELS) includes talkie.
      A pre-D1a writer would not see talkie at all.

  test_writer_widget_routes_talkie_to_gptq_int4_adapter
      When the writer's creative slot is set to talkie's repo_id,
      get_backend_for_row resolves to TransformersGPTQInt4Backend.

  test_effective_context_limit_for_talkie_is_4096_not_8192
      Talkie's compute_effective_context_limit returns 4096 (smaller
      than HARD_VRAM_CONTEXT_LIMIT 8192 so the row wins). Mirror of
      the D1b helper test from the writer-widget perspective.

  test_talkie_row_chat_template_kind_is_transformers_default
      Schema-level assertion that the talkie row declares
      chat_template_kind = "transformers_default". The runtime
      tokenizer guard (test_talkie_tokenizer_has_chat_template_when_*)
      is gated under OTR_REGRESSION_RUNTIME=1 and lives in the runtime
      smoke test file.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_loader_backends as backends_proto  # noqa: E402
from nodes import _otr_model_catalog as catalog  # noqa: E402
from nodes import _otr_model_runtime as runtime  # noqa: E402

TALKIE_REPO_ID = "talkie-lm/talkie-1930-13b-it"


def _talkie_row():
    rows = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    talkie = rows.get(TALKIE_REPO_ID)
    assert talkie is not None, (
        f"talkie row {TALKIE_REPO_ID!r} missing from CURATED_LLM_MODELS"
    )
    return talkie


def test_talkie_in_creative_writing_model_dropdown_enum() -> None:
    """The dropdown_choices() enum (writer's creative_writing_model
    widget source) must include talkie's repo_id.
    """
    choices = catalog.dropdown_choices()
    # dropdown_choices may append a UI suffix like [NOT DOWNLOADED]
    # to entries not in the local HF cache. Strip suffixes for the
    # presence check.
    bare_choices = [str(c).split(" ", 1)[0].strip() for c in choices]
    assert TALKIE_REPO_ID in bare_choices, (
        f"talkie repo_id {TALKIE_REPO_ID!r} missing from dropdown "
        f"choices; choices (bare): {bare_choices}"
    )


def test_writer_widget_routes_talkie_to_gptq_int4_adapter() -> None:
    """Setting the writer's creative slot to talkie's repo_id must
    dispatch via get_backend_for_row to TransformersGPTQInt4Backend.
    """
    talkie = _talkie_row()
    adapter = runtime.get_backend_for_row(talkie)
    assert isinstance(adapter, runtime.TransformersGPTQInt4Backend), (
        f"writer widget routing for talkie resolved to "
        f"{type(adapter).__name__}; expected TransformersGPTQInt4Backend"
    )


def test_effective_context_limit_for_talkie_is_4096_not_8192() -> None:
    """compute_effective_context_limit must return 4096 for talkie.
    HARD_VRAM_CONTEXT_LIMIT is 8192 (or env-overridden); talkie's
    row.context_window is 4096; the smaller wins.
    """
    talkie = _talkie_row()
    out = backends_proto.compute_effective_context_limit(talkie)
    assert out == 4096, (
        f"compute_effective_context_limit(talkie) = {out}; expected "
        f"4096 (row.context_window={talkie.context_window} clamped "
        f"to HARD_VRAM_CONTEXT_LIMIT={catalog.HARD_VRAM_CONTEXT_LIMIT})"
    )
    # And specifically NOT 8192 -- pin the regression class where a
    # future commit defaults talkie's context_window to the system limit.
    assert out != 8192, (
        f"compute_effective_context_limit(talkie) = 8192; talkie's "
        f"context_window appears to have drifted from 4096"
    )


def test_talkie_row_chat_template_kind_is_transformers_default() -> None:
    """Schema-level assertion: talkie declares chat_template_kind =
    'transformers_default'. The runtime guard that actually loads
    the tokenizer and verifies tokenizer.chat_template is not None
    is in test_tokenizer_chat_template_guard.py and is runtime-gated.
    """
    talkie = _talkie_row()
    assert talkie.chat_template_kind == "transformers_default", (
        f"talkie chat_template_kind = {talkie.chat_template_kind!r}; "
        f"expected 'transformers_default'. If the talkie tokenizer "
        f"ships without a chat_template, flip the row's "
        f"chat_template_kind to 'manual' and re-run D2c."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
