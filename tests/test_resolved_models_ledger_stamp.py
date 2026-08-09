"""A `~latest` alias must leave its RESOLVED model in the ledger.

WHY THIS FILE EXISTS. Queue item 1 chunk B promoted the creative default from a
pinned slug to `~anthropic/claude-opus-latest`. The whole safety argument for
shipping a pointer instead of a pin is: "replay is unaffected, because the
ledger records the concrete model that actually served the run."

That record lived in exactly ONE place -- the "RESOLVED (OPENROUTER)" section
of the credits sheet, built in video_engine.py on the VIDEO path. Proven live
on 2026-08-09: a story-only leg (workflows/otr_story_only.json, 3 nodes, no
media) ran `~anthropic/claude-opus-latest`, made real accounted remote calls,
returned RESULT SUCCESS -- and its ledger contained no provenance at all,
because no video node ever executed. Every writer-only and scoring run silently
lost the answer to "which model wrote this".

The stamp now happens in the writer, immediately before its terminal save.
"""
from __future__ import annotations

import ast
from pathlib import Path

SRC = (Path(__file__).resolve().parents[1] / "nodes"
       / "OTR_LedgerScriptWriter.py")


def _writer_source() -> str:
    return SRC.read_text(encoding="utf-8")


def test_writer_stamps_resolved_models_into_meta():
    src = _writer_source()
    assert 'meta["resolved_models"]' in src, (
        "the writer does not stamp resolved_models; a story-only or "
        "writer-only run loses the concrete model that served it"
    )


def test_the_stamp_precedes_the_terminal_save():
    """Order is the whole point -- stamping after led.save() persists nothing."""
    src = _writer_source()
    stamp = src.find('meta["resolved_models"]')
    save = src.find("saved_path = led.save()")
    assert stamp > 0 and save > 0
    assert stamp < save, (
        "resolved_models is stamped AFTER the terminal save, so it never "
        "reaches disk"
    )


def test_the_stamp_is_conditional_so_local_runs_stay_byte_identical():
    """A purely local run must add no key at all."""
    src = _writer_source()
    idx = src.find('meta["resolved_models"]')
    window = src[max(0, idx - 300):idx]
    assert "if _resolved_now:" in window, (
        "the stamp is unconditional -- a local-only run would gain an empty "
        "resolved_models key and every existing ledger shape would change"
    )


def test_provenance_can_never_fail_a_render():
    """Diagnostics must not be able to kill a writer run."""
    src = _writer_source()
    idx = src.find("resolved_models_snapshot as _resolved_snapshot")
    assert idx > 0, "snapshot import missing"
    window = src[max(0, idx - 200):idx + 300]
    assert "try:" in window and "except Exception" in window, (
        "the snapshot import/call is not guarded"
    )


def test_the_writer_module_still_parses():
    """Cheap AST gate -- this edit sits deep in a 6600-line file."""
    ast.parse(_writer_source())


def test_backend_still_exposes_the_snapshot_helper():
    """The stamp reads through the backend's public helper rather than
    re-deriving the mapping, so the credits sheet and the ledger can never
    disagree about what served the run."""
    from nodes import _otr_openrouter_backend as orb
    assert callable(getattr(orb, "resolved_models_snapshot", None))
    assert callable(getattr(orb, "reset_run_budget", None))
