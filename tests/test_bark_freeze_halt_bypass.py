"""Functional CastLock enforcement of structural/safety freeze verdicts."""

from __future__ import annotations

import json

import pytest


def _ledger(verdict=None, block_class=None):
    meta = {}
    if verdict is not None:
        meta["freeze_verdict"] = verdict
    if block_class is not None:
        meta["freeze_block_class"] = block_class
    return json.dumps(
        {
            "schema_version": "l3-2026-05-14",
            "cast": [],
            "lines": [],
            "meta": meta,
        }
    )


def _lock(ledger_json):
    from nodes.cast_lock import CastLock

    return CastLock().lock(script_json=ledger_json)


@pytest.mark.parametrize("block_class", ["structural", "safety", None])
def test_needs_full_rerun_halts(block_class, monkeypatch):
    monkeypatch.delenv("OTR_BYPASS_FREEZE_HALT", raising=False)
    with pytest.raises(ValueError, match="needs_full_rerun"):
        _lock(_ledger("needs_full_rerun", block_class))


def test_operator_diagnostic_bypass_allows_flagged_ledger(monkeypatch):
    monkeypatch.setenv("OTR_BYPASS_FREEZE_HALT", "1")
    out = _lock(_ledger("needs_full_rerun", "structural"))
    assert isinstance(out, tuple) and len(out) == 4


@pytest.mark.parametrize("verdict", [None, "frozen_clean", "frozen_with_warns"])
def test_nonterminal_verdicts_continue(verdict, monkeypatch):
    monkeypatch.delenv("OTR_BYPASS_FREEZE_HALT", raising=False)
    out = _lock(_ledger(verdict))
    assert isinstance(out, tuple) and len(out) == 4
