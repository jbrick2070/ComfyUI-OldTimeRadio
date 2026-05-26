"""tests/test_otr_render_plan.py -- Sprint 6 critic-to-render coupling.

Covers `nodes/_otr_render_plan.build_render_plan`:

  1. all defaults + clean critic -> every character line in ledger order.
  2. `render_max_n` caps the plan length and stamps applied_max_n.
  3. `dramatic_peaks_only` reorders the plan by critic.render_priority.
  4. `protagonist_only` restricts the plan to the protagonist's lines.
  5. `manual_line_ids` overrides everything (selection, flat_lines, the
     arc-verdict gate) -- the operator's hand wins.
  6. The arc-verdict gate blocks the render when cycle_count >= 2 AND
     arc_verdict in (mid_collapse, flat).
  7. The arc-verdict gate does NOT fire when cycle_count < 2 (the
     reroll loop has not yet exhausted).
  8. `flat_lines` are excluded UNLESS the line was rerolled (status
     "rerolled" in meta.reroll_history).
  9. `build_render_plan` NEVER raises on malformed input -- returns None.
 10. An empty character pool yields an empty plan (not blocked) so the
     caller stamps an empty list and HuMo skips no real character work.

Pure-Python -- no LLM, no ledger object, no fixtures beyond plain dicts.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nodes._otr_render_plan import (  # noqa: E402
    DEFAULT_RENDER_MAX_N,
    RENDER_SELECTION_CHOICES,
    build_render_plan,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _critic_report(arc_verdict="strong", render_priority=None,
                   flat_lines=None, reroll_targets=None):
    return {
        "continuity_issues": [],
        "voice_drift": [],
        "flat_lines": [
            {"line_id": lid, "reason": "test"}
            for lid in (flat_lines or [])
        ],
        "arc_verdict": arc_verdict,
        "reroll_targets": [
            {"line_id": lid, "hint": "test"}
            for lid in (reroll_targets or [])
        ],
        "render_priority": list(render_priority or []),
    }


def _ledger(
    *,
    arc_verdict="strong",
    render_priority=None,
    flat_lines=None,
    reroll_history=None,
    cycle_count=0,
    extra_cast=None,
    extra_lines=None,
):
    """Build a small ledger dict with 4 character lines + 1 announcer."""
    cast = [
        {"char_id": "c01", "name": "EDNA", "speaker_role": "character"},
        {"char_id": "c02", "name": "MARLOWE", "speaker_role": "character"},
        {"char_id": "announcer", "name": "ANNOUNCER",
         "speaker_role": "announcer"},
    ]
    if extra_cast:
        cast.extend(extra_cast)
    lines = [
        {"line_id": "l001", "char_id": "c01", "speaker_role": "character",
         "text": "First.", "word_count": 1},
        {"line_id": "l002", "char_id": "c02", "speaker_role": "character",
         "text": "Second.", "word_count": 1},
        {"line_id": "l003", "char_id": "announcer",
         "speaker_role": "announcer", "text": "..."},
        {"line_id": "l004", "char_id": "c01", "speaker_role": "character",
         "text": "Third.", "word_count": 1},
        {"line_id": "l005", "char_id": "c02", "speaker_role": "character",
         "text": "Fourth.", "word_count": 1},
    ]
    if extra_lines:
        lines.extend(extra_lines)
    meta = {
        "story_critic_report": _critic_report(
            arc_verdict=arc_verdict,
            render_priority=render_priority,
            flat_lines=flat_lines,
        ),
        "cycle_count": cycle_count,
        "reroll_history": reroll_history or [],
    }
    return {"meta": meta, "cast": cast, "lines": lines}


# ---------------------------------------------------------------------------
# 1. defaults: every character line in ledger order
# ---------------------------------------------------------------------------


def test_defaults_plan_lists_every_character_line():
    data = _ledger()
    plan = build_render_plan(data)
    assert plan is not None
    assert plan["selection_mode"] == "all"
    assert plan["blocked"] is False
    assert plan["line_ids"] == ["l001", "l002", "l004", "l005"]
    assert plan["source_arc_verdict"] == "strong"
    assert plan["source_cycle_count"] == 0
    assert plan["candidate_pool_size"] == 4
    assert plan["excluded_flat_lines"] == []


# ---------------------------------------------------------------------------
# 2. render_max_n caps the plan
# ---------------------------------------------------------------------------


def test_render_max_n_caps_the_plan():
    data = _ledger()
    plan = build_render_plan(data, render_max_n=2)
    assert plan["line_ids"] == ["l001", "l002"]
    assert plan["applied_max_n"] == 2


def test_render_max_n_zero_disables_the_cap():
    data = _ledger()
    plan = build_render_plan(data, render_max_n=0)
    assert len(plan["line_ids"]) == 4
    assert plan["applied_max_n"] == 0


# ---------------------------------------------------------------------------
# 3. dramatic_peaks_only reorders by render_priority
# ---------------------------------------------------------------------------


def test_dramatic_peaks_only_reorders_by_render_priority():
    data = _ledger(render_priority=["l005", "l001", "l004", "l002"])
    plan = build_render_plan(data, render_selection="dramatic_peaks_only")
    assert plan["selection_mode"] == "dramatic_peaks_only"
    # Critic peaks first (l005, l001, l004, l002) -- in critic-stamped order.
    assert plan["line_ids"] == ["l005", "l001", "l004", "l002"]


def test_dramatic_peaks_only_keeps_unprioritized_lines_in_tail():
    # render_priority names only 2 of the 4 character lines; the other
    # two fall to the tail in ledger order.
    data = _ledger(render_priority=["l005", "l001"])
    plan = build_render_plan(data, render_selection="dramatic_peaks_only")
    assert plan["line_ids"][:2] == ["l005", "l001"]
    assert set(plan["line_ids"][2:]) == {"l002", "l004"}


# ---------------------------------------------------------------------------
# 4. protagonist_only filters to the most-spoken character
# ---------------------------------------------------------------------------


def test_protagonist_only_filters_to_top_character():
    # c01 (EDNA) has 2 lines (l001, l004); c02 has 2 (l002, l005). Tie:
    # cast-roster order breaks it -- c01 comes first -> EDNA is protag.
    data = _ledger()
    plan = build_render_plan(data, protagonist_only=True)
    assert plan["selection_mode"] == "protagonist_only"
    assert plan["line_ids"] == ["l001", "l004"]


def test_protagonist_only_picks_higher_count_over_cast_order():
    extra = [
        {"line_id": "l006", "char_id": "c02",
         "speaker_role": "character", "text": "X.", "word_count": 1},
    ]
    data = _ledger(extra_lines=extra)
    plan = build_render_plan(data, protagonist_only=True)
    # c02 (MARLOWE) now has 3 lines vs c01's 2 -> MARLOWE wins.
    assert plan["line_ids"] == ["l002", "l005", "l006"]


# ---------------------------------------------------------------------------
# 5. manual_line_ids supersedes everything (including the arc-block)
# ---------------------------------------------------------------------------


def test_manual_line_ids_overrides_and_lifts_arc_block():
    data = _ledger(arc_verdict="flat", cycle_count=2)
    # Even with the arc-verdict gate active, an explicit manual list
    # ships exactly those line_ids.
    plan = build_render_plan(data, manual_line_ids="l004, l001")
    assert plan["selection_mode"] == "manual"
    assert plan["blocked"] is False
    assert plan["line_ids"] == ["l004", "l001"]


def test_manual_line_ids_drops_unknown_ids():
    data = _ledger()
    plan = build_render_plan(data, manual_line_ids="l999,l001,l002,bogus")
    assert plan["line_ids"] == ["l001", "l002"]


# ---------------------------------------------------------------------------
# 6. arc-verdict gate blocks when cap reached + verdict still bad
# ---------------------------------------------------------------------------


def test_arc_verdict_gate_blocks_render_after_cap():
    data = _ledger(arc_verdict="mid_collapse", cycle_count=2)
    plan = build_render_plan(data)
    assert plan["blocked"] is True
    assert plan["line_ids"] == []
    assert "mid_collapse" in plan["blocked_reason"]
    assert plan["source_cycle_count"] == 2


def test_arc_verdict_gate_blocks_on_flat_too():
    data = _ledger(arc_verdict="flat", cycle_count=2)
    plan = build_render_plan(data)
    assert plan["blocked"] is True
    assert plan["line_ids"] == []


def test_arc_verdict_gate_does_not_block_uneven():
    data = _ledger(arc_verdict="uneven", cycle_count=2)
    plan = build_render_plan(data)
    assert plan["blocked"] is False
    assert plan["line_ids"]  # non-empty


# ---------------------------------------------------------------------------
# 7. arc-verdict gate does not fire pre-cap
# ---------------------------------------------------------------------------


def test_arc_verdict_gate_dormant_when_cycle_count_below_cap():
    # cycle_count=1 -> the reroll loop has not yet exhausted -> gate
    # stays dormant even on the worst arc_verdict.
    data = _ledger(arc_verdict="mid_collapse", cycle_count=1)
    plan = build_render_plan(data)
    assert plan["blocked"] is False
    assert plan["line_ids"]


# ---------------------------------------------------------------------------
# 8. flat_lines exclusion unless rerolled
# ---------------------------------------------------------------------------


def test_flat_lines_excluded_by_default():
    data = _ledger(flat_lines=["l002"])
    plan = build_render_plan(data)
    assert "l002" not in plan["line_ids"]
    assert plan["excluded_flat_lines"] == ["l002"]


def test_rerolled_flat_line_is_kept():
    history = [
        {"cycle": 1, "line_id": "l002", "hint": "h",
         "status": "rerolled", "old_text": "x", "new_text": "y"},
    ]
    data = _ledger(flat_lines=["l002"], reroll_history=history)
    plan = build_render_plan(data)
    assert "l002" in plan["line_ids"]
    assert plan["excluded_flat_lines"] == []


def test_failed_reroll_does_not_save_flat_line():
    history = [
        {"cycle": 1, "line_id": "l002", "hint": "h",
         "status": "compose_failed", "detail": "x"},
    ]
    data = _ledger(flat_lines=["l002"], reroll_history=history)
    plan = build_render_plan(data)
    assert "l002" not in plan["line_ids"]
    assert "l002" in plan["excluded_flat_lines"]


# ---------------------------------------------------------------------------
# 9. never raises on malformed input
# ---------------------------------------------------------------------------


def test_never_raises_on_malformed_ledger():
    # build_render_plan tolerates a missing meta / lines / cast and
    # degrades gracefully; the outermost try/except returns None on a
    # genuinely broken input so HuMo falls back to render-all.
    plan = build_render_plan({})
    # Empty pool -> empty plan, not None (the inner function still runs).
    assert plan is not None
    assert plan["line_ids"] == []
    assert plan["candidate_pool_size"] == 0


def test_never_raises_on_garbage_critic_report():
    data = _ledger()
    data["meta"]["story_critic_report"] = "not a dict"
    # Should still produce a plan -- the garbage critic stamp is
    # treated as empty.
    plan = build_render_plan(data)
    assert plan is not None
    assert plan["line_ids"] == ["l001", "l002", "l004", "l005"]


# ---------------------------------------------------------------------------
# 10. constants
# ---------------------------------------------------------------------------


def test_default_constants_match_widget_defaults():
    assert DEFAULT_RENDER_MAX_N == 6
    assert RENDER_SELECTION_CHOICES == ("all", "dramatic_peaks_only")


if __name__ == "__main__":  # pragma: no cover - manual smoke only
    sys.exit(pytest.main([__file__, "-v"]))
