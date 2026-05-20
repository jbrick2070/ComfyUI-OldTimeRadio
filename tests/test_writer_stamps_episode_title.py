"""The J.5 post-composition title pass is the single authority for
meta.episode_title. Source-only assertions (do not exercise the full
run() loop, which needs a real LLM call).

History: Sprint E "K.5.7" added a later widget-only re-stamp that ran
AFTER J.5 and clobbered J.5's LLM-generated title with the empty
widget value -- a blank episode_title field produced an empty
meta.episode_title, so the video title chain fell to the timestamp
last-resort (BUG-LOCAL-236). K.5.7 was deleted 2026-05-20; J.5's
stamp `meta["episode_title"] = final_title` is now the only one.
"""
from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
WRITER_SOURCE = REPO_ROOT / "nodes" / "OTR_LedgerScriptWriter.py"


def test_j5_title_pass_stamps_episode_title():
    """J.5 stamps meta.episode_title from the resolved final_title
    (user-typed, else LLM regen from the finished script, else
    outline.title fallback)."""
    src = WRITER_SOURCE.read_text(encoding="utf-8")
    assert "J.5. Post-composition title regen" in src
    assert 'meta["episode_title"] = final_title' in src


def test_no_widget_only_clobber_returns():
    """The retired K.5.7 widget-only re-stamp must not come back -- it
    clobbered J.5's generated title with the empty widget value."""
    src = WRITER_SOURCE.read_text(encoding="utf-8")
    assert "K.5.7:" not in src, "K.5.7 widget-only title clobber resurrected"
    assert '_widget_title = (episode_title or' not in src, (
        "widget-only episode_title re-stamp resurrected"
    )


def test_j5_stamp_lives_inside_run_before_return_assembly():
    """J.5's stamp must be on the writer's hot path, before the
    L. return-assembly block."""
    src = WRITER_SOURCE.read_text(encoding="utf-8")
    run_idx = src.find("def run(")
    stamp_idx = src.find('meta["episode_title"] = final_title')
    l_idx = src.find("# --- L. Assemble return values")
    assert run_idx >= 0, "writer must have a run() method"
    assert stamp_idx >= 0, "J.5 episode_title stamp missing"
    assert l_idx >= 0
    assert run_idx < stamp_idx < l_idx, (
        "J.5 stamp must live inside run() and before return assembly"
    )
