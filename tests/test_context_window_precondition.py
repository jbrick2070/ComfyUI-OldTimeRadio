"""Sprint D D4 -- context_window precondition gate.

Two structural assertions. No HF auth no GPU.

  test_context_window_precondition_below_hard_limit_raises
      A row whose context_window is BELOW HARD_VRAM_CONTEXT_LIMIT
      trips check_context_window with a clear error message.

  test_context_window_precondition_at_or_above_hard_limit_passes
      A row whose context_window is at or above HARD_VRAM_CONTEXT_LIMIT
      passes silently (no return value, no raise).

Plus a real-row smoke for Mistral-Nemo: at context_window=8192 it
matches the default HARD_VRAM_CONTEXT_LIMIT and does not trip.
"""
from __future__ import annotations

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
    context_window: int
    repo_id: str = "fake/repo"


def test_context_window_precondition_below_hard_limit_raises() -> None:
    """A row with context_window < HARD_VRAM_CONTEXT_LIMIT raises
    RuntimeError with a clear message naming the offending values.
    """
    hard_limit = catalog.HARD_VRAM_CONTEXT_LIMIT
    row = _FakeRow(context_window=max(1, hard_limit - 1024))
    with pytest.raises(RuntimeError) as excinfo:
        backends_proto.check_context_window(row)
    msg = str(excinfo.value)
    assert "context_window" in msg
    assert "HARD_VRAM_CONTEXT_LIMIT" in msg
    assert "G5" in msg, (
        f"precondition error message does not reference operator "
        f"gate G5: {msg!r}"
    )


def test_context_window_precondition_at_hard_limit_passes() -> None:
    """A row whose context_window equals HARD_VRAM_CONTEXT_LIMIT
    passes silently (the strict-less-than comparison is intentional;
    equal-to is acceptable).
    """
    row = _FakeRow(context_window=catalog.HARD_VRAM_CONTEXT_LIMIT)
    # Should not raise.
    out = backends_proto.check_context_window(row)
    assert out is None  # no return value


def test_context_window_precondition_above_hard_limit_passes() -> None:
    """A row whose context_window exceeds HARD_VRAM_CONTEXT_LIMIT
    passes silently. The precondition is only about catastrophic
    undersize at the model level.
    """
    row = _FakeRow(context_window=catalog.HARD_VRAM_CONTEXT_LIMIT + 100_000)
    backends_proto.check_context_window(row)  # no raise


def test_context_window_precondition_mistral_nemo_real_row_passes() -> None:
    """Mistral-Nemo at context_window=16384 (2026-07-19: raised from 8192 so
    the local scifi_codex_v4 420/720w script pass fits) is >= the default
    HARD_VRAM_CONTEXT_LIMIT=8192, so it does NOT trip the precondition.
    """
    rows_by_id = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    mistral = rows_by_id.get("mistralai/Mistral-Nemo-Instruct-2407")
    assert mistral is not None
    assert mistral.context_window == 16384
    # No raise (16384 >= HARD_VRAM_CONTEXT_LIMIT).
    backends_proto.check_context_window(mistral)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
