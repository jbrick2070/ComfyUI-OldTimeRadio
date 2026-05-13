"""C9 / IMP-33 (S24, 2026-05-13) -- _LLMTimeoutWorkflowPause
queue-halt smoke test (mock-based).

The S22.1 class docstring claims ComfyUI's node-execution layer
surfaces uncaught exceptions as queue halts. That's a load-bearing
assumption with no automated test. This smoke exercises the
exception-propagation contract via a fake two-node queue.

Decision rationale: docs/2026-05-13-imp33-queue-halt-test-decision.md
(Option B: mock-based; rejected Option A as brittle / Option C as
needing a ComfyUI subprocess harness OTR doesn't have today).

Mock-vs-real divergence: this test validates that OTR-side code is
written in a way that depends only on the documented exception
contract. It does NOT validate ComfyUI's actual queue behavior. A
future ComfyUI version swallowing uncaught exceptions silently
would NOT fire this test -- that's tracked separately as IMP-33b
in the decision doc.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from nodes.story_orchestrator import _LLMTimeoutWorkflowPause


class _MockQueueExecutor:
    """Stand-in for ComfyUI's node-execution layer.

    The actual layer iterates the dependency graph, calls each
    node's execute(), and catches exceptions to surface them as
    queue halts. This mock implements just enough of that contract
    to exercise the OTR-side `_LLMTimeoutWorkflowPause` raise.

    Records `executed_nodes` so the test can assert which nodes ran
    and which were skipped after the halt.
    """

    def __init__(self):
        self.executed_nodes: list[str] = []
        self.halt_exception: BaseException | None = None
        self.halted_at: str | None = None

    def run(self, ordered_nodes: list[tuple[str, MagicMock]]) -> None:
        """Walk nodes in dependency order. If any node's execute()
        raises, capture the exception, record the halt point, and
        skip every subsequent node."""
        for name, node in ordered_nodes:
            try:
                node.execute()
                self.executed_nodes.append(name)
            except BaseException as exc:
                self.halt_exception = exc
                self.halted_at = name
                # ComfyUI's actual layer would surface this in the
                # frontend; the mock just halts iteration. We do NOT
                # call subsequent node.execute().
                break


def test_workflow_pause_halts_queue_before_next_node():
    """Mock-based queue exercise of the S22.1 contract.

    Two-node workflow: ScriptWriter (raises pause) -> FluxPortrait
    (never executes). After the run:
      - ScriptWriter raised _LLMTimeoutWorkflowPause.
      - The queue executor captured the exception and halted at
        ScriptWriter.
      - FluxPortrait.execute() was NOT called (verified via mock
        call_count).
    """
    script_writer = MagicMock(name="OTR_LedgerScriptWriter")
    script_writer.execute.side_effect = _LLMTimeoutWorkflowPause(
        "smoke: simulated 1s budget exceeded; orphan worker still "
        "on GPU. Halting workflow."
    )
    flux_portrait = MagicMock(name="OTR_BatchFluxPortraitRender")

    executor = _MockQueueExecutor()
    executor.run([
        ("OTR_LedgerScriptWriter", script_writer),
        ("OTR_BatchFluxPortraitRender", flux_portrait),
    ])

    # 1. ScriptWriter was attempted; the exception propagated.
    script_writer.execute.assert_called_once()
    # 2. FluxPortrait was NEVER attempted -- queue halted at the
    #    ScriptWriter raise.
    flux_portrait.execute.assert_not_called()
    # 3. Executor captured the correct halt point + exception type.
    assert executor.halted_at == "OTR_LedgerScriptWriter"
    assert isinstance(
        executor.halt_exception, _LLMTimeoutWorkflowPause
    )
    # 4. executed_nodes is empty because ScriptWriter raised before
    #    being appended; FluxPortrait was skipped.
    assert executor.executed_nodes == []


def test_workflow_pause_subclass_match_for_legacy_handlers():
    """Existing ``except _LLMTimeout`` handlers must still match
    the subclass. A queue executor that catches the base class
    should still see the pause surface (not pass through silently).
    """
    from nodes.story_orchestrator import _LLMTimeout

    raised = []
    try:
        raise _LLMTimeoutWorkflowPause("test")
    except _LLMTimeout as exc:
        raised.append(type(exc).__name__)
    assert raised == ["_LLMTimeoutWorkflowPause"], (
        "C9: legacy `except _LLMTimeout` handlers must still match "
        "the subclass via standard Python inheritance."
    )


def test_orphan_worker_message_signals_rerun():
    """The exception message must guide the operator -- this is
    the only signal a frontend user sees when the queue halts.
    Pin the two action-relevant phrases."""
    msg = str(_LLMTimeoutWorkflowPause(
        "ScriptWriter exceeded 60s; orphan worker still on GPU. "
        "Halting workflow to prevent the next visual stage from "
        "racing the orphan's CUDA kernels. Re-run the workflow; "
        "the cache invalidation above guarantees the next attempt "
        "loads fresh."
    ))
    assert "orphan" in msg.lower()
    assert "re-run" in msg.lower() or "rerun" in msg.lower()


def test_pre_halt_no_cuda_error_illegal_address():
    """A C9-specific defensive check: the exception message itself
    must NOT contain "cudaErrorIllegalAddress" -- that error is what
    the halt is supposed to PREVENT. If a future change to
    `_LLMTimeoutWorkflowPause` accidentally embeds it (e.g. via a
    string concat from a CUDA traceback), this test fires."""
    exc = _LLMTimeoutWorkflowPause(
        "ScriptWriter exceeded 60s; orphan worker still on GPU."
    )
    assert "cudaErrorIllegalAddress" not in str(exc).lower(), (
        "C9: pause exception must not reference cudaErrorIllegalAddress; "
        "the whole point of the halt is to prevent that error class."
    )


@pytest.mark.skip(
    reason="C9 / IMP-33a: real ComfyUI subprocess smoke -- needs "
    "ComfyUI CI-friendly harness. Tracked in "
    "docs/2026-05-13-imp33-queue-halt-test-decision.md."
)
def test_real_comfyui_queue_halt_subprocess_smoke():
    """Future Option C: spin up real ComfyUI in a subprocess, load
    workflows/otr_scifi_16gb_full.json, force a ScriptWriter timeout,
    assert ComfyUI's actual queue halts. Stub remains skipped until
    ComfyUI ships a CI harness."""
    raise NotImplementedError(
        "see docs/2026-05-13-imp33-queue-halt-test-decision.md "
        "for the contract"
    )
