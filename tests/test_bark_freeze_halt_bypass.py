"""Freeze-halt gate -- re-homed to OTR_CastLock (audio clean-break 1a).

The BUG-LOCAL-276 / BUG-LOCAL-300 freeze-halt used to be enforced inside the
legacy audio nodes (BatchBark et al.). With bark flipped to a per_line registry
engine and the batch node retired, the gate moved to OTR_CastLock -- the single
ledger authority that runs FIRST in the v2 audio chain (CastLock ->
CharacterVoices -> Announcer -> Theme), so one gate covers every downstream
audio engine. The per-node ``bypass_freeze_halt`` WIDGET became the env var
``OTR_BYPASS_FREEZE_HALT`` (OTR convention: every knob is an env var; v2 nodes
forbid extra widgets, plan E.4).

These tests drive ``CastLock.lock()`` functionally (headless -- preserve_ledger
does no model load) plus a few source/env-contract pins. ASCII-only source.
"""
from __future__ import annotations

import json
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
CAST_LOCK_SRC = REPO_ROOT / "nodes" / "cast_lock.py"


def _ledger(verdict=None, block_class=None, unload_ok=None):
    meta = {}
    if verdict is not None:
        meta["freeze_verdict"] = verdict
    if block_class is not None:
        meta["freeze_block_class"] = block_class
    if unload_ok is not None:
        meta["freeze_unload_ok"] = unload_ok
    return json.dumps({
        "schema_version": "l3-2026-05-14",
        "cast": [],
        "lines": [],
        "meta": meta,
    })


def _lock(ledger_json):
    from nodes.cast_lock import CastLock

    return CastLock().lock(script_json=ledger_json)


# ---------------------------------------------------------------------------
# Functional gate behavior (drives CastLock.lock directly)
# ---------------------------------------------------------------------------
class TestFreezeHaltFunctional:
    def test_structural_needs_full_rerun_halts(self, monkeypatch):
        monkeypatch.delenv("OTR_BYPASS_FREEZE_HALT", raising=False)
        with pytest.raises(ValueError, match="needs_full_rerun"):
            _lock(_ledger("needs_full_rerun", "structural"))

    def test_missing_block_class_treated_structural_halts(self, monkeypatch):
        """Missing/unknown freeze_block_class is treated as structural (halt)."""
        monkeypatch.delenv("OTR_BYPASS_FREEZE_HALT", raising=False)
        with pytest.raises(ValueError):
            _lock(_ledger("needs_full_rerun", None))

    def test_quality_block_renders_not_halts(self, monkeypatch):
        """A subjective quality verdict on a cast-clean ledger renders."""
        monkeypatch.delenv("OTR_BYPASS_FREEZE_HALT", raising=False)
        monkeypatch.delenv("OTR_BARK_HALT_ON_QUALITY_BLOCK", raising=False)
        out = _lock(_ledger("needs_full_rerun", "quality"))
        assert isinstance(out, tuple) and len(out) == 4  # rendered, did not raise

    def test_env_bypass_renders_flagged_ledger(self, monkeypatch):
        monkeypatch.setenv("OTR_BYPASS_FREEZE_HALT", "1")
        out = _lock(_ledger("needs_full_rerun", "structural"))
        assert isinstance(out, tuple) and len(out) == 4

    def test_strict_quality_env_halts_on_quality(self, monkeypatch):
        monkeypatch.delenv("OTR_BYPASS_FREEZE_HALT", raising=False)
        monkeypatch.setenv("OTR_BARK_HALT_ON_QUALITY_BLOCK", "1")
        with pytest.raises(ValueError):
            _lock(_ledger("needs_full_rerun", "quality"))

    def test_no_verdict_renders(self, monkeypatch):
        monkeypatch.delenv("OTR_BYPASS_FREEZE_HALT", raising=False)
        out = _lock(_ledger())
        assert isinstance(out, tuple) and len(out) == 4


# ---------------------------------------------------------------------------
# Source / env-contract pins
# ---------------------------------------------------------------------------
class TestFreezeHaltContract:
    def _src(self) -> str:
        return CAST_LOCK_SRC.read_text(encoding="utf-8")

    def test_gate_lives_in_cast_lock(self):
        src = self._src()
        assert "freeze_verdict" in src
        assert "needs_full_rerun" in src

    def test_bypass_is_env_var_not_widget(self):
        """The bypass moved from a per-node widget to an env var; the widget
        name must NOT have followed the gate into the v2 surface (E.4)."""
        src = self._src()
        assert "OTR_BYPASS_FREEZE_HALT" in src
        assert "bypass_freeze_halt" not in src

    def test_quality_split_env_present(self):
        assert "OTR_BARK_HALT_ON_QUALITY_BLOCK" in self._src()

    def test_references_bug_numbers(self):
        src = self._src()
        assert "BUG-LOCAL-276" in src
        assert "BUG-LOCAL-300" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
