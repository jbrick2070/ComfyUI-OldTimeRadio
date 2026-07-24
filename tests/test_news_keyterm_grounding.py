"""Optional news key-term grounding is deterministic and non-gating."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_NODES_DIR = _REPO_ROOT / "nodes"
for _p in (_REPO_ROOT, _NODES_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

news_interpreter = pytest.importorskip("news_interpreter")

_SOURCE = (
    "Astronomers using a large telescope reported an unexplained "
    "signal from a nearby star system this week. Researchers cautioned "
    "that the data must be verified before publication."
)
_BASE_BRIEF = {
    "casting_brief": (
        "A field astronomer and a skeptical colleague test every claim."
    ),
    "script_brief": (
        "A telescope picks up an unexplained signal and researchers verify it."
    ),
    "news_close_brief": (
        "Researchers reported an unexplained signal, with verification pending."
    ),
}


def _build(key_terms):
    payload = dict(_BASE_BRIEF, key_terms=list(key_terms))
    calls = {"n": 0}

    def technical_fn(messages, *, temperature=0.7, max_new_tokens=400):
        calls["n"] += 1
        return json.dumps(payload)

    briefs = news_interpreter.build_news_briefs(
        technical_fn=technical_fn,
        full_text=_SOURCE,
        headline="Signal detected",
        summary=_SOURCE[:120],
        seed=42,
    )
    return briefs, calls["n"]


def test_fabricated_terms_are_deleted_after_one_structural_call():
    briefs, calls = _build(
        ["telescope", "signal", "wormhole", "antimatter"])
    assert briefs.key_terms == ["telescope", "signal"]
    assert calls == 1


def test_one_grounded_term_is_valid():
    briefs, calls = _build(["telescope", "wormhole", "antimatter"])
    assert briefs.key_terms == ["telescope"]
    assert calls == 1


def test_zero_grounded_terms_is_valid():
    briefs, calls = _build(["wormhole", "antimatter"])
    assert briefs.key_terms == []
    assert calls == 1


def test_fully_grounded_terms_pass_unchanged():
    briefs, calls = _build(["telescope", "signal", "researchers"])
    assert briefs.key_terms == ["telescope", "signal", "researchers"]
    assert calls == 1
