"""Story-QA quality/liveness split and dynamic repair regression.

The quality judge may request MICRO_REPAIR_NEEDED or REJECT, but neither may
abort an otherwise ledgerable episode. The spine repeatedly scopes guarded
creative repairs, re-runs the independent judge, and records a quality floor if
the backend remains stubborn. Structural freeze/readiness gates stay separate.
"""
from __future__ import annotations

from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
_WRITER_SRC = (_REPO / "nodes" / "OTR_LedgerScriptWriter.py").read_text(encoding="utf-8")
_SPINE_SRC = (_REPO / "nodes" / "_otr_story_spine.py").read_text(encoding="utf-8")
def test_writer_never_raises_on_subjective_story_reject():
    assert "OTR reject gate" not in _WRITER_SRC
    assert 'meta.pop("story_verdict", None)' in _WRITER_SRC
    assert '"legacy_reject_ignored"' in _WRITER_SRC


def test_legacy_reject_cleanup_is_after_the_spine_call():
    spine_call = _WRITER_SRC.find("run_post_script_spine")
    reject_check = _WRITER_SRC.find('meta.pop("story_verdict", None)')
    assert spine_call != -1, "writer must call run_post_script_spine"
    assert reject_check != -1, "writer must clear a stale legacy signal"
    assert reject_check > spine_call, (
        "legacy cleanup must run after the spine owns quality repair"
    )


def test_spine_reject_enters_dynamic_loop_and_can_only_quality_floor():
    assert 'in ("MICRO_REPAIR_NEEDED", "REJECT")' in _SPINE_SRC
    assert "story_qa_repair_cycles" in _SPINE_SRC
    assert '"quality_floor"' in _SPINE_SRC
    assert 'meta["story_verdict"] = "REJECT"' not in _SPINE_SRC
    assert '"ok_reject"' not in _SPINE_SRC


def test_spine_calls_scrub_with_repair_available_false():
    """Sprint 4 wiring: micro-repair runs upstream now, so the scrub is
    always told repair_available=False (mechanical, fail-closed, last)."""
    assert "repair_available=False" in _SPINE_SRC, (
        "scrub_ledger must be called with repair_available=False"
    )
    assert "repair_available=not repair_used" not in _SPINE_SRC, (
        "the old repair_available=not repair_used wiring must be gone"
    )


def test_qa_pass_is_opt_in_default_off():
    """BUG-LOCAL-302: the Story QA / reject LLM must be GATED OFF by default --
    no call and no tokens -- and run only when OTR_ENABLE_STORY_QA=1.
    A credit-billed taste pass remains opt-in even though it is nonterminal."""
    import os as _os
    from nodes import _otr_story_spine as _spine

    _prev = _os.environ.pop("OTR_ENABLE_STORY_QA", None)
    try:
        assert _spine.qa_enabled() is False, "QA must default OFF"
        _os.environ["OTR_ENABLE_STORY_QA"] = "1"
        assert _spine.qa_enabled() is True, "OTR_ENABLE_STORY_QA=1 opts in"
        _os.environ["OTR_ENABLE_STORY_QA"] = "0"
        assert _spine.qa_enabled() is False
    finally:
        if _prev is None:
            _os.environ.pop("OTR_ENABLE_STORY_QA", None)
        else:
            _os.environ["OTR_ENABLE_STORY_QA"] = _prev

    # The run_story_qa LLM call must live behind the qa_enabled() gate.
    assert "def qa_enabled" in _SPINE_SRC
    assert "OTR_ENABLE_STORY_QA" in _SPINE_SRC
    gate_idx = _SPINE_SRC.find("if not qa_enabled()")
    qa_call_idx = _SPINE_SRC.find("run_story_qa(")
    assert gate_idx != -1, "the QA pass must be gated by qa_enabled()"
    assert qa_call_idx > gate_idx, (
        "run_story_qa must be called only inside the qa_enabled() gate"
    )
