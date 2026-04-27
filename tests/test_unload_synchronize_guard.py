"""Regression test for BUG-LOCAL-073: model.cpu() during unload faults
on a dirty CUDA context and zombifies the ComfyUI server.

Bug: in `nodes/story_orchestrator.py:_unload_llm()`, `model.cpu()` was
called without first synchronising the CUDA stream. If a prior kernel
left dirty memory (a pending illegal access not yet observed), the
.cpu() walk would dereference invalid pointers, the NVIDIA driver
would promote the access to a process-level fault, and the python
process would become an unkillable CUDA-locked zombie holding port
8000. Live evidence (2026-04-26 PM): two-LLM-split run hit this at
the OpenClose -> Director transition, log line `[StoryOrchestrator]
model.cpu() during unload failed: CUDA error: an illegal memory
access was encountered`, server unresponsive after.

Fix: call `torch.cuda.synchronize()` BEFORE `.cpu()`. If a kernel
fault is pending, synchronize raises a clean Python exception and we
SKIP the cpu() walk entirely -- the process stays alive, the cache
gets dropped, empty_cache() resets the allocator, and the next phase
starts with a usable GPU context. This test exercises the
synchronize-failure path without needing real CUDA hardware.

Mirrors the guard added to `batch_audiogen_generator.py` for the
AudioGen unload path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Add the project root so test imports resolve (matches the pattern in
# tests/test_cast_fuzzy_consolidate.py).
_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parent
sys.path.insert(0, str(_REPO))


# ---------------------------------------------------------------------------
# Minimal exec-based extraction of `_unload_llm` so we don't need to import
# the full story_orchestrator module (which would pull transformers, torch,
# comfy etc.). The pattern matches tests/test_cast_fuzzy_consolidate.py.
# ---------------------------------------------------------------------------


def _extract_unload_llm_function():
    """Use ast to find the exact line range of `_unload_llm`, then exec
    just those source lines. This avoids the relative-import noise that
    follows the function in the module."""
    import ast
    src_text = (_REPO / "nodes" / "story_orchestrator.py").read_text(encoding="utf-8")
    src_lines = src_text.splitlines()
    tree = ast.parse(src_text)
    target = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "_unload_llm":
            target = node
            break
    if target is None:
        raise RuntimeError("could not find _unload_llm via ast")
    start = target.lineno - 1
    end = target.end_lineno  # ast end_lineno is inclusive
    block = "\n".join(src_lines[start:end])
    ns = {
        "__name__": "_otr_unload_test",
        "_LLM_CACHE": {
            "model": None,
            "tokenizer": None,
            "device": None,
            "quantized": False,
            "model_id": None,
        },
        "log": MagicMock(),
        "_runtime_log": MagicMock(),
    }
    exec(compile(block, "story_orchestrator.py", "exec"), ns)
    return ns


# ---------------------------------------------------------------------------
# Fake torch + fake model + fake gc — controlled environment for testing
# the synchronize/cpu sequencing without real CUDA.
# ---------------------------------------------------------------------------


class _FakeCuda:
    def __init__(self, sync_raises=None, empty_raises=None):
        self.sync_raises = sync_raises
        self.empty_raises = empty_raises
        self.sync_calls = 0
        self.empty_cache_calls = 0
        self.memory_allocated_value = 0.03e9
        self.memory_reserved_value = 0.13e9

    def is_available(self):
        return True

    def synchronize(self):
        self.sync_calls += 1
        if self.sync_raises is not None:
            raise self.sync_raises

    def empty_cache(self):
        self.empty_cache_calls += 1
        if self.empty_raises is not None:
            raise self.empty_raises

    def memory_allocated(self):
        return self.memory_allocated_value

    def memory_reserved(self):
        return self.memory_reserved_value


class _FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


class _FakeModel:
    def __init__(self, cpu_raises=None):
        self.cpu_raises = cpu_raises
        self.cpu_calls = 0

    def cpu(self):
        self.cpu_calls += 1
        if self.cpu_raises is not None:
            raise self.cpu_raises


def _run_unload(sync_raises=None, cpu_raises=None, empty_raises=None):
    """Run _unload_llm in a controlled fake-cuda environment.

    Returns (ns, fake_model, fake_cuda) so the test can introspect.
    """
    ns = _extract_unload_llm_function()
    fake_model = _FakeModel(cpu_raises=cpu_raises)
    fake_cuda = _FakeCuda(sync_raises=sync_raises, empty_raises=empty_raises)
    fake_torch = _FakeTorch(fake_cuda)
    fake_gc = MagicMock()
    ns["_LLM_CACHE"]["model"] = fake_model
    ns["_LLM_CACHE"]["tokenizer"] = MagicMock()
    # Patch the lazy imports inside the function. Function body does
    # `import gc; import torch` -- we have to intercept those module lookups.
    with patch.dict("sys.modules", {"torch": fake_torch, "gc": fake_gc}):
        # Mock comfy.model_management because the code tries to import it.
        comfy_mod = MagicMock()
        comfy_mod.model_management.soft_empty_cache = MagicMock()
        with patch.dict("sys.modules", {"comfy": comfy_mod, "comfy.model_management": comfy_mod.model_management}):
            ns["_unload_llm"]()
    return ns, fake_model, fake_cuda, fake_gc


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSyncSucceedsHappyPath:
    """When CUDA is healthy, both synchronize() and cpu() run."""

    def test_normal_unload_calls_synchronize_then_cpu(self):
        ns, model, cuda, gc_mod = _run_unload()
        assert cuda.sync_calls == 1, "synchronize() must be called before cpu()"
        assert model.cpu_calls == 1, "cpu() must run when sync succeeds"

    def test_normal_unload_clears_cache(self):
        ns, model, cuda, gc_mod = _run_unload()
        assert ns["_LLM_CACHE"]["model"] is None
        assert ns["_LLM_CACHE"]["tokenizer"] is None

    def test_normal_unload_runs_gc_and_empty_cache(self):
        ns, model, cuda, gc_mod = _run_unload()
        gc_mod.collect.assert_called()
        assert cuda.empty_cache_calls == 1


class TestSyncFails_BUG073_RecoveryPath:
    """When synchronize() raises (latent CUDA fault), cpu() must NOT run.

    This is THE regression case for BUG-073. The whole reason for the
    guard is: do not let .cpu() walk dirty memory after a kernel fault,
    because that walk is what zombifies the process.
    """

    def test_synchronize_failure_skips_cpu_walk(self):
        ns, model, cuda, gc_mod = _run_unload(
            sync_raises=RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            )
        )
        assert cuda.sync_calls == 1, "synchronize() was attempted"
        assert model.cpu_calls == 0, (
            "cpu() must NOT be called when synchronize fails -- this is "
            "the bug we are guarding against"
        )

    def test_synchronize_failure_still_clears_cache(self):
        ns, model, cuda, gc_mod = _run_unload(
            sync_raises=RuntimeError(
                "CUDA error: an illegal memory access was encountered"
            )
        )
        assert ns["_LLM_CACHE"]["model"] is None, "cache must be dropped even after sync failure"
        assert ns["_LLM_CACHE"]["tokenizer"] is None

    def test_synchronize_failure_still_empties_cache(self):
        ns, model, cuda, gc_mod = _run_unload(
            sync_raises=RuntimeError("CUDA fault")
        )
        # empty_cache() resets the allocator state -- must run even if cpu()
        # was skipped, otherwise the next phase sees stale fragmentation.
        assert cuda.empty_cache_calls == 1


class TestCpuFailsAfterSyncSucceeds:
    """If synchronize succeeds but cpu() still fails for some other reason
    (rare race: a kernel started after sync completed), the function must
    NOT propagate the exception -- just log it and continue cleanup."""

    def test_cpu_failure_caught_and_cleanup_continues(self):
        ns, model, cuda, gc_mod = _run_unload(
            cpu_raises=RuntimeError("post-sync cpu fault")
        )
        assert cuda.sync_calls == 1
        assert model.cpu_calls == 1
        # Cache still cleared, gc still runs, empty_cache still runs
        assert ns["_LLM_CACHE"]["model"] is None
        gc_mod.collect.assert_called()
        assert cuda.empty_cache_calls == 1


class TestEmptyCacheFails:
    """empty_cache() faulting is the worst case but still must not raise."""

    def test_empty_cache_failure_does_not_propagate(self):
        # Should not raise even when empty_cache itself faults.
        ns, model, cuda, gc_mod = _run_unload(
            empty_raises=RuntimeError("allocator state corrupt")
        )
        assert ns["_LLM_CACHE"]["model"] is None
        assert cuda.empty_cache_calls == 1


class TestGuardOrdering:
    """The order is load-bearing: synchronize MUST come before cpu."""

    def test_synchronize_called_before_cpu_in_happy_path(self):
        # Use an instrumented model that records call order.
        order = []
        ns = _extract_unload_llm_function()
        fake_model = _FakeModel()
        fake_model.cpu = lambda: order.append("cpu")  # type: ignore[assignment]
        fake_cuda = _FakeCuda()
        fake_cuda.synchronize = lambda: order.append("sync")  # type: ignore[assignment]
        fake_cuda.empty_cache = lambda: order.append("empty_cache")  # type: ignore[assignment]
        fake_torch = _FakeTorch(fake_cuda)
        ns["_LLM_CACHE"]["model"] = fake_model
        ns["_LLM_CACHE"]["tokenizer"] = MagicMock()
        with patch.dict("sys.modules", {"torch": fake_torch, "gc": MagicMock()}):
            comfy_mod = MagicMock()
            with patch.dict("sys.modules", {"comfy": comfy_mod, "comfy.model_management": comfy_mod.model_management}):
                ns["_unload_llm"]()
        # synchronize must precede cpu; both must precede empty_cache
        assert order.index("sync") < order.index("cpu")
        assert order.index("cpu") < order.index("empty_cache")


class TestNoOpIfNoModel:
    """When _LLM_CACHE['model'] is None, the function must be a no-op
    that doesn't even touch torch.cuda."""

    def test_no_model_means_no_cuda_calls(self):
        ns = _extract_unload_llm_function()
        fake_cuda = _FakeCuda()
        fake_torch = _FakeTorch(fake_cuda)
        # model already None from initial state
        with patch.dict("sys.modules", {"torch": fake_torch, "gc": MagicMock()}):
            comfy_mod = MagicMock()
            with patch.dict("sys.modules", {"comfy": comfy_mod, "comfy.model_management": comfy_mod.model_management}):
                ns["_unload_llm"]()
        assert fake_cuda.sync_calls == 0
        assert fake_cuda.empty_cache_calls == 0
