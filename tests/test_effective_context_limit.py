"""Sprint D D1b -- compute_effective_context_limit helper.

The row-level context_window field (D1a) plus the system-wide
HARD_VRAM_CONTEXT_LIMIT (overridable via OTR_HARD_VRAM_CONTEXT_LIMIT)
together resolve to an effective per-row cap via:

    min(HARD_VRAM_CONTEXT_LIMIT, row.context_window)

This is the helper D2c's adapter dispatch uses to cap prompt budget
per backend. Mirror of the legacy CURATED_CONTEXT_OVERRIDES /
resolve_context_cap path; reads the new row field so per-row
variance (talkie at 4096) is respected.
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
    """Minimum-fixture row carrying just the context_window field."""
    context_window: int
    repo_id: str = "fake/repo"


def test_compute_effective_context_limit_clamps_to_hard_limit_when_window_larger() -> None:
    """A row with context_window > HARD_VRAM_CONTEXT_LIMIT clamps to
    the system limit. Catches the "model advertises 128k context but
    16 GB rig can only feed 8k" case.
    """
    hard_limit = catalog.HARD_VRAM_CONTEXT_LIMIT
    row = _FakeRow(context_window=hard_limit + 50_000)
    out = backends_proto.compute_effective_context_limit(row)
    assert out == hard_limit, (
        f"compute_effective_context_limit({hard_limit + 50_000}) = "
        f"{out}; expected clamp to {hard_limit}"
    )


def test_compute_effective_context_limit_returns_row_value_when_smaller() -> None:
    """A row with context_window < HARD_VRAM_CONTEXT_LIMIT wins.
    This is talkie's case: native 4096, system limit 8192,
    effective limit 4096.
    """
    hard_limit = catalog.HARD_VRAM_CONTEXT_LIMIT
    assert hard_limit >= 4096, (
        "test assumption: HARD_VRAM_CONTEXT_LIMIT is at least 4096"
    )
    row = _FakeRow(context_window=4096)
    out = backends_proto.compute_effective_context_limit(row)
    assert out == 4096, (
        f"compute_effective_context_limit(4096) = {out}; expected 4096 "
        f"(smaller of 4096 and HARD_VRAM_CONTEXT_LIMIT={hard_limit})"
    )


def test_compute_effective_context_limit_returns_row_value_when_equal() -> None:
    """Boundary case: row == limit returns the shared value."""
    hard_limit = catalog.HARD_VRAM_CONTEXT_LIMIT
    row = _FakeRow(context_window=hard_limit)
    out = backends_proto.compute_effective_context_limit(row)
    assert out == hard_limit


def test_compute_effective_context_limit_against_talkie_real_row() -> None:
    """The actual talkie row in the catalog clamps to 4096 (smaller
    than the 8192 hard limit). End-to-end smoke against the real D1a
    surface, not a fake row.
    """
    rows_by_id = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    talkie = rows_by_id.get("talkie-lm/talkie-1930-13b-it")
    assert talkie is not None, "talkie row missing from CURATED_LLM_MODELS"
    out = backends_proto.compute_effective_context_limit(talkie)
    assert out == 4096, (
        f"talkie effective context limit = {out}; expected 4096"
    )


def test_compute_effective_context_limit_against_mistral_nemo_real_row() -> None:
    """Mistral-Nemo at 8192 stays at 8192 (no clamp; window equals
    limit). Pin the audio-C7-baseline row's effective limit.
    """
    rows_by_id = {m.repo_id: m for m in catalog.CURATED_LLM_MODELS}
    mistral = rows_by_id.get("mistralai/Mistral-Nemo-Instruct-2407")
    assert mistral is not None
    out = backends_proto.compute_effective_context_limit(mistral)
    assert out == catalog.HARD_VRAM_CONTEXT_LIMIT, (
        f"Mistral-Nemo effective context limit = {out}; expected "
        f"HARD_VRAM_CONTEXT_LIMIT={catalog.HARD_VRAM_CONTEXT_LIMIT}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
