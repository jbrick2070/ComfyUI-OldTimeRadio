"""PBUG-20260825-04: the NF4 tripwire trusted a race-prone process-global
signal, and a timeout's own workflow-pause could be silently swallowed.

Two independent, INDEPENDENTLY-VERIFIED fixes, tested here:

1. ``_otr_model_loader._bug098_scan_linear4bit_devices`` -- extracted from
   the inline tripwire so it can be tested directly. It replaces
   ``torch.cuda.memory_allocated()`` delta (process-wide, corrupted by a
   concurrent orphan worker freeing tensors in the same window) with a
   MODEL-LOCAL check: does every ``bitsandbytes.Linear4bit`` module's own
   weight tensor actually carry data on a CUDA device. Live evidence this
   was needed: an 8 GB RTX 4060 render showed
   ``linear4bit_count=592 is_loaded_in_4bit=True vram_delta=-0.00GiB`` --
   every module-level signal said the load worked; only the noisy
   process-wide delta disagreed and killed a working load.

2. ``story_orchestrator._LLMTimeoutWorkflowPause`` must always reach the
   node boundary. Both NewsCuration and NewsCurationDeep's LLM-ranking
   helpers used to catch it via a broad ``except Exception`` and (when
   ``load_config`` is None) silently fall back to shuffle order -- letting
   the main thread immediately start ANOTHER LLM load while this phase's
   orphan worker is still alive on GPU (generation is not cancellable
   mid-token; ``_run_with_timeout`` abandons the worker, it does not stop
   it). That overlap window is where PBUG-20260825-04 was found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")

from nodes import _otr_model_loader as ml  # noqa: E402
from nodes import story_orchestrator as so  # noqa: E402


# --------------------------------------------------------------------------- #
# Fix 1: _bug098_scan_linear4bit_devices
# --------------------------------------------------------------------------- #

class _FakeLinear4bit:
    """Shaped like bitsandbytes.nn.Linear4bit closely enough for the scan:
    the scan only reads type().__name__, type().__module__, and .weight.device.
    """
    __module__ = "bitsandbytes.nn.modules"

    def __init__(self, device):
        self.weight = torch.zeros(1, device=device)


# Rename the class per-instance isn't possible for __name__ checks against
# type(obj).__name__, so give the scan a class actually named Linear4bit.
_FakeLinear4bit.__name__ = "Linear4bit"
_FakeLinear4bit.__qualname__ = "Linear4bit"


class _FakeOrdinaryLayer:
    """A non-quantized module the scan must ignore."""

    def __init__(self):
        self.weight = torch.zeros(1, device="cpu")


class _FakeModel:
    """Exposes named_modules() the way a real nn.Module does, with a mix of
    quantized and ordinary layers at distinct paths."""

    def __init__(self, linear4bit_devices):
        self._mods = {
            f"layer.{i}.q": _FakeLinear4bit(dev)
            for i, dev in enumerate(linear4bit_devices)
        }
        self._mods["embed"] = _FakeOrdinaryLayer()

    def named_modules(self):
        return list(self._mods.items())


def test_scan_reports_all_cuda_when_every_linear4bit_weight_is_on_cuda():
    model = _FakeModel(["cuda", "cuda", "cuda"])
    count, off_cuda = ml._bug098_scan_linear4bit_devices(model)
    assert count == 3
    assert off_cuda == []


def test_scan_flags_a_linear4bit_module_stuck_on_cpu():
    """The genuine failure mode this tripwire exists to catch: real
    quantized modules that did NOT materialize on GPU."""
    model = _FakeModel(["cuda", "cpu", "cuda"])
    count, off_cuda = ml._bug098_scan_linear4bit_devices(model)
    assert count == 3
    assert len(off_cuda) == 1
    assert "layer.1.q=cpu" in off_cuda[0]


def test_scan_ignores_non_linear4bit_modules():
    model = _FakeModel([])  # only the ordinary CPU-resident embed layer
    count, off_cuda = ml._bug098_scan_linear4bit_devices(model)
    assert count == 0
    assert off_cuda == []


def test_scan_does_not_use_process_global_vram_state():
    """The regression itself. A concurrent orphan freeing tensors elsewhere
    in the process must NOT influence this function's verdict -- it takes
    no VRAM-counter input at all, only the model object. Simulate a
    'contaminated' process by actually moving the global allocator (if
    CUDA is present) or simply asserting the function signature carries no
    such parameter -- either way, a real weight tensor's device is ground
    truth regardless of what memory_allocated() reports elsewhere."""
    import inspect
    sig = inspect.signature(ml._bug098_scan_linear4bit_devices)
    assert list(sig.parameters) == ["model"], (
        "the scan must depend ONLY on the model object, never on a "
        "process-wide VRAM counter -- that dependency is exactly what "
        "made the old tripwire race-prone"
    )
    # And a model whose weights are genuinely on CUDA passes, full stop,
    # with no VRAM bookkeeping involved anywhere in the call.
    model = _FakeModel(["cuda"])
    count, off_cuda = ml._bug098_scan_linear4bit_devices(model)
    assert count == 1 and off_cuda == []


def test_scan_survives_a_broken_module_tree():
    """named_modules() raising must report -1, not crash the loader."""
    class _Explodes:
        def named_modules(self):
            raise RuntimeError("synthetic module-tree corruption")

    count, off_cuda = ml._bug098_scan_linear4bit_devices(_Explodes())
    assert count == -1
    assert off_cuda == []


# --------------------------------------------------------------------------- #
# Fix 2: _LLMTimeoutWorkflowPause must reach the node boundary
# --------------------------------------------------------------------------- #

def test_llm_timeout_workflow_pause_is_not_swallowed_by_news_curation():
    """Source-level pin: the except clause ordering in both news-ranking
    helpers must catch _LLMTimeoutWorkflowPause and re-raise it BEFORE the
    broad except Exception that used to swallow it.

    A source check (rather than a full behavioral repro) is deliberate
    here: reproducing the real race needs an actual timed-out background
    LLM thread on a real GPU, which is exactly the 4060-only condition this
    bug was found under and cannot be faithfully simulated in the test
    suite (CUDA_VISIBLE_DEVICES='' in this env). What CAN be pinned
    precisely is the control-flow contract: the specific exception type
    must never be caught by the generic handler.
    """
    import ast
    import inspect

    checked = _count_pause_before_broad_except(inspect.getsource(so))
    assert checked == 2, (
        f"expected exactly 2 try/except blocks pairing "
        f"_LLMTimeoutWorkflowPause with a broad Exception handler "
        f"(NewsCuration + NewsCurationDeep); found {checked}. If this "
        f"count changed intentionally (a helper added or removed), update "
        f"this test's expectation in the same change."
    )


def _count_pause_before_broad_except(src: str) -> int:
    """How many try/except blocks in ``src`` catch
    ``_LLMTimeoutWorkflowPause`` (bare or ``module.``-qualified) BEFORE a
    broad ``except Exception``, and actually re-raise rather than swallow
    it. Shared by both propagation-guard tests below."""
    import ast

    tree = ast.parse(src)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Try):
            continue
        handler_types = []
        for h in node.handlers:
            if h.type is None:
                handler_types.append("bare-except")
            elif isinstance(h.type, ast.Name):
                handler_types.append(h.type.id)
            elif isinstance(h.type, ast.Attribute):
                handler_types.append(h.type.attr)
        if "Exception" not in handler_types:
            continue
        if "_LLMTimeoutWorkflowPause" not in handler_types:
            continue
        checked += 1
        pause_idx = handler_types.index("_LLMTimeoutWorkflowPause")
        exc_idx = handler_types.index("Exception")
        assert pause_idx < exc_idx, (
            "except _LLMTimeoutWorkflowPause must come BEFORE except "
            "Exception in the same try block, or the broad handler catches "
            "it first and this fix is a no-op. handler order: "
            f"{handler_types}"
        )
        # And it must actually re-raise, not swallow.
        pause_handler = node.handlers[pause_idx]
        assert any(isinstance(s, ast.Raise) for s in pause_handler.body), (
            "the _LLMTimeoutWorkflowPause handler must re-raise, not "
            "swallow the pause"
        )
    return checked


def test_llm_timeout_workflow_pause_is_not_swallowed_by_rss_seed_fetch():
    """PBUG-20260825-04 follow-up, found by kibitz r1 (cursor lane) on the
    orphan-lifecycle design round: the propagation fix above lets the pause
    escape NewsCuration/NewsCurationDeep correctly, but ONE LEVEL UP,
    ``OTR_LedgerScriptWriter._fetch_rss_seed_or_die`` (the real caller,
    via ``_otr_source_payload.py``'s science_rss wrapper -- which has no
    try/except of its own) caught it via its own broad ``except
    Exception`` and recast it as a generic 'RSS fetch failed' RuntimeError,
    erasing the type before it ever reached a node boundary. Same fix,
    same pattern, one more call-chain level.
    """
    from nodes import OTR_LedgerScriptWriter as writer
    import inspect

    checked = _count_pause_before_broad_except(
        inspect.getsource(writer._fetch_rss_seed_or_die)
    )
    assert checked == 1, (
        "_fetch_rss_seed_or_die must catch _LLMTimeoutWorkflowPause and "
        "re-raise it before its broad except Exception -- otherwise the "
        "pause dies here and never reaches the node boundary regardless "
        "of the story_orchestrator-level fix"
    )
