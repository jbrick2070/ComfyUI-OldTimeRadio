"""Sprint D D1c -- runtime-gated tokenizer chat-template guard.

When the catalog row declares `chat_template_kind = transformers_default`
the underlying HuggingFace tokenizer MUST expose a `chat_template`
attribute. If not, D2c's `_encode_messages` dispatch would call
`apply_chat_template` against a tokenizer that has no template -- the
transformers library raises with a noisy stack at that point.

This guard catches the mismatch at fixture-time instead of at
generation-time. Per Sprint D G3 gating discipline the guard runs
under OTR_REGRESSION_RUNTIME=1 because it requires `AutoTokenizer
.from_pretrained(repo_id)` which pulls the tokenizer.json from the
gated talkie HuggingFace repo (`requires_auth=True`).

Sprint A or operator-discretion runs this with HF_TOKEN configured.
Structural pytest invocations skip silently.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from nodes import _otr_model_catalog as catalog  # noqa: E402

TALKIE_REPO_ID = "talkie-lm/talkie-1930-13b-it"

OTR_REGRESSION_RUNTIME = os.environ.get("OTR_REGRESSION_RUNTIME") == "1"


@pytest.mark.skipif(
    not OTR_REGRESSION_RUNTIME,
    reason="runtime gate: set OTR_REGRESSION_RUNTIME=1 to exercise",
)
def test_talkie_tokenizer_has_chat_template_when_kind_is_transformers_default() -> None:
    """Per v3 plan D1c: fail loud if the row says
    chat_template_kind=transformers_default but the tokenizer has no
    chat_template; the operator should switch the row to
    chat_template_kind=manual in that case.

    Runtime-gated because AutoTokenizer.from_pretrained pulls the
    tokenizer.json from the gated repo (requires HF_TOKEN).
    """
    from transformers import AutoTokenizer  # noqa: PLC0415

    rows = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    row = rows[TALKIE_REPO_ID]
    tok = AutoTokenizer.from_pretrained(row.repo_id)
    if row.chat_template_kind == "transformers_default":
        assert tok.chat_template is not None, (
            f"{row.repo_id} declares chat_template_kind="
            f"transformers_default but tokenizer.chat_template is None. "
            f"Switch the row to chat_template_kind=manual and define "
            f"the manual template in D2c, or pick a different talkie "
            f"variant whose tokenizer ships with a chat_template."
        )
