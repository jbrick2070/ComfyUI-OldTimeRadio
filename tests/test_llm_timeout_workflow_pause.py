"""S22.1 (IMP-26) + S22.2: LLM-timeout workflow-pause subclass tests.

S22.1: ``_run_with_timeout`` raises ``_LLMTimeoutWorkflowPause`` (a
subclass of ``_LLMTimeout``) instead of the base class. This lets
downstream consumers branch on the more specific type while existing
``except _LLMTimeout`` handlers continue to match.

S22.2: end-to-end check that a real ComfyUI queue halts before the
next FLUX node fires. Requires the ComfyUI test harness which the
bare-pytest test env doesn't have; the manual procedure is documented
in docs/manual-smoke-tests.md.
"""
from __future__ import annotations

import time

import pytest


def test_pause_is_subclass_of_llm_timeout():
    """Existing ``except _LLMTimeout`` handlers must still match
    the new subclass -- no downstream regressions."""
    from nodes.story_orchestrator import (
        _LLMTimeout,
        _LLMTimeoutWorkflowPause,
    )
    assert issubclass(_LLMTimeoutWorkflowPause, _LLMTimeout)


def test_timeout_raises_pause_subclass():
    """_run_with_timeout on a slow callable must raise the subclass.

    Uses a sleep + a 0.1 s budget; the worker thread sleeps past
    the deadline and the executor's TimeoutError converts to
    _LLMTimeoutWorkflowPause.
    """
    from nodes.story_orchestrator import (
        _run_with_timeout,
        _LLMTimeoutWorkflowPause,
    )

    def _slow():
        time.sleep(0.5)
        return "should not reach"

    with pytest.raises(_LLMTimeoutWorkflowPause) as exc:
        _run_with_timeout(_slow, timeout_sec=0.1, phase_label="TEST")
    assert "exceeded" in str(exc.value)
    assert "TEST" in str(exc.value)


def test_pause_message_mentions_orphan_and_rerun():
    """The exception message must guide the operator -- 'orphan
    worker still on GPU' + 'Re-run the workflow' are the two action-
    relevant phrases."""
    from nodes.story_orchestrator import (
        _run_with_timeout,
        _LLMTimeoutWorkflowPause,
    )

    def _slow():
        time.sleep(0.3)

    with pytest.raises(_LLMTimeoutWorkflowPause) as exc:
        _run_with_timeout(_slow, timeout_sec=0.05, phase_label="TEST")
    msg = str(exc.value)
    assert "orphan" in msg.lower()
    assert "Re-run" in msg or "re-run" in msg


@pytest.mark.skip(
    reason="S22.2: requires ComfyUI test harness; manual procedure "
    "in docs/manual-smoke-tests.md"
)
def test_llm_timeout_halts_queue_before_flux_node():
    """End-to-end: ScriptWriter timeout -> FreezeCascade -> FluxPortrait.
    FluxPortrait must NOT execute. Expected stderr:
    _LLMTimeoutWorkflowPause traceback. Expected: no CUDA OOM, no
    cudaErrorIllegalAddress.

    The skip is intentional -- the integration harness is S20-tier
    stretch work. The manual smoke-test doc is the executable
    contract until then.
    """
    raise NotImplementedError(
        "see docs/manual-smoke-tests.md S22.2 procedure"
    )
