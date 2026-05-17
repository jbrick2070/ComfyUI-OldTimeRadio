"""Sprint E E11 / R-11: writer stamps meta.episode_title from widget.

Closes M11. Pre-E11 the writer never stamped meta.episode_title;
SignalLost's title chain Tier 1 was dead and the chain fell to
Tier 4 (widget at SignalLost) or Tier 5 (TIMESTAMP_LASTRESORT).
Post-E11 the writer stamps the widget value at K.5.7 so Tier 1
fires correctly.

Source-only assertion (does not exercise the full run() loop
which requires a real LLM call).
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WRITER_SOURCE = REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


def test_writer_stamps_episode_title_at_k557():
    src = WRITER_SOURCE.read_text(encoding="utf-8")
    # The K.5.7 stamp must assign meta["episode_title"] = widget value.
    assert "K.5.7" in src
    assert 'meta["episode_title"]' in src or "meta['episode_title']" in src


def test_writer_stamp_handles_empty_widget_explicitly():
    """Empty widget -> stamp empty string, not omit the key. Lets
    downstream meta.get("episode_title") return "" rather than the
    surprise KeyError path."""
    src = WRITER_SOURCE.read_text(encoding="utf-8")
    # The except arm in the stamp block explicitly stamps "".
    assert 'meta["episode_title"] = ""' in src or "meta['episode_title'] = ''" in src


def test_writer_stamp_lives_inside_run_method():
    """The stamp must be on the writer's hot path, not a helper that
    can be bypassed. K.5.7 is INSIDE OTR_LedgerScriptWriter.run."""
    src = WRITER_SOURCE.read_text(encoding="utf-8")
    # Find the run() method start and check the K.5.7 stamp comes
    # before the L. Assemble return values block.
    run_idx = src.find("def run(")
    k557_idx = src.find("K.5.7")
    L_assemble_idx = src.find("# --- L. Assemble return values")
    assert run_idx >= 0, "writer must have a run() method"
    assert k557_idx >= 0
    assert L_assemble_idx >= 0
    assert run_idx < k557_idx < L_assemble_idx, (
        "K.5.7 stamp must live inside run() and BEFORE return assembly"
    )


def test_widget_value_path_uses_episode_title_param():
    """The stamp reads from the local `episode_title` parameter (the
    widget value) rather than from some other ledger field."""
    src = WRITER_SOURCE.read_text(encoding="utf-8")
    # The stamp block contains `_widget_title = (episode_title or "").strip()`
    assert "(episode_title or" in src
