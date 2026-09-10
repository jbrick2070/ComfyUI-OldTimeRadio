"""The LLM teardown's Metal branch -- the one that was missing, and what it cost.

`_teardown_gpu_for_entry` had a canonical six-step sequence in its docstring and
implemented all six ONLY for CUDA. Steps 1-3 (model.to("cpu"), del, gc.collect)
are platform-neutral and always ran; steps 4-6 sat behind
`if torch.cuda.is_available():` with no counterpart, so on Apple Silicon the
Python object was correctly freed and PyTorch's MPS caching allocator kept its
reserved pool forever.

WHAT IT ACTUALLY COSTS, and the first answer here was WRONG. This file
originally said the Metal pool RATCHETED across an episode's 8-12 writer
reloads until the OS killed the process. `load_llm` washes with
`soft_empty_cache` before every load and that DOES release MPS, so the previous
copy was gone before the next one arrived -- there is no ratchet.

The measured mechanism (PBUG-20260908-03 CORRECTION) is a 2x WINDOW:

    model resident on mps          MPS 2056.5 MiB
    after model.to("cpu")          MPS 2056.5 MiB   <- pool still held, and a
                                                      CPU copy now exists
    after torch.mps.empty_cache()  MPS    0.5 MiB

Step 1 copies to CPU and releases nothing, so the writer sat double-counted for
the entire gap between stages -- the window the video models, Kokoro and
StableAudio3 load into on a 16 GB machine. This teardown collapses it.

A trace confirmed there is NO leaked reference in this path -- the entry is
detached, LLM_CACHE cleared, the model moved to cpu and reaped. That is what
makes the fix a missing call rather than a lifetime bug, and it is why the
measurement below is the whole argument.
"""
from __future__ import annotations

import gc
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _loader_source():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "nodes", "_otr_model_loader.py")
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _teardown_block():
    src = _loader_source()
    start = src.index("def _teardown_gpu_for_entry")
    end = src.index("def unload_llm", start)
    return src[start:end]


def test_the_metal_branch_exists_at_all():
    """The regression this file exists for: steps 4-6 with no Metal path."""
    block = _teardown_block()
    assert "torch.mps.empty_cache()" in block, (
        "_teardown_gpu_for_entry has no torch.mps.empty_cache(). On Apple "
        "Silicon that means every LLM unload frees the object and returns none "
        "of the memory, and an episode does this 8-12 times with an 8.7 GB "
        "model.")


def test_cuda_still_wins_and_is_untouched():
    """The 5080 must not change. The Metal path is an `elif`, so a CUDA host
    takes the branch it always took and never evaluates the Metal test."""
    block = _teardown_block()
    cuda_at = block.index("if torch.cuda.is_available():")
    mps_at = block.index("torch.backends.mps.is_available()")
    assert cuda_at < mps_at, "the Metal check must not precede the CUDA one"
    assert "elif" in block[cuda_at:mps_at], (
        "the Metal branch must be an elif -- an independent `if` would run "
        "torch.mps calls on a CUDA box that may not have an mps build")
    for call in ("torch.cuda.empty_cache()", "torch.cuda.ipc_collect()",
                 "torch.cuda.synchronize()"):
        assert call in block, "CUDA teardown lost %s" % call


def test_the_metal_branch_is_guarded_for_older_torch():
    """`torch.mps` is absent on older builds; touching it must not raise inside
    a teardown whose contract is 'never raises'."""
    block = _teardown_block()
    assert 'getattr(torch, "mps", None)' in block, (
        "guard torch.mps with getattr -- this function must never raise")


@pytest.mark.skipif(
    not (sys.platform == "darwin"),
    reason="the measurement below only means anything on Apple Silicon")
def test_del_and_gc_do_not_return_the_memory_but_empty_cache_does():
    """THE MEASUREMENT THAT IS THE WHOLE ARGUMENT.

    Steps 1-3 of the teardown end at gc.collect(). This shows that is where the
    memory ISN'T: a 2 GB allocation survives del+gc in the allocator's pool and
    only `torch.mps.empty_cache()` hands it back. If a future torch makes gc
    sufficient, this test starts failing and the extra call can go."""
    torch = pytest.importorskip("torch")
    if not torch.backends.mps.is_available():
        pytest.skip("not an Apple Silicon host")

    held = torch.mps.driver_allocated_memory()
    block = torch.randn(1024, 1024, 256, device="mps")   # ~1 GiB
    torch.mps.synchronize()
    peak = torch.mps.driver_allocated_memory()
    assert peak - held > 512 * 1024 * 1024, "allocation did not take effect"

    del block
    gc.collect()
    after_gc = torch.mps.driver_allocated_memory()
    assert after_gc - held > 512 * 1024 * 1024, (
        "del + gc.collect() returned the memory on its own, which is where the "
        "old teardown stopped -- if torch now does this, the Metal branch is "
        "no longer load-bearing and this test should be revisited rather than "
        "deleted")

    torch.mps.empty_cache()
    torch.mps.synchronize()
    after_empty = torch.mps.driver_allocated_memory()
    assert after_empty - held < 128 * 1024 * 1024, (
        "torch.mps.empty_cache() did NOT return the pool; the fix does not "
        "work on this torch build")
