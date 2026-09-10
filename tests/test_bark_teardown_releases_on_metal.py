"""`_unload_bark` returned nothing on Metal, and ORDERING is why it mattered.

`_unload_bark` ran `del` + `gc.collect()` + a BARE `torch.cuda.empty_cache()`.
On Apple Silicon that last call is a silent no-op, so the Python object was
correctly reaped and PyTorch's MPS caching allocator kept Bark's ~4.2 GB
reserved.

WHY THIS SITE AND NOT THE OTHER TWENTY-FOUR `empty_cache` CALLS. An audit of
every one of them found that most are harmless: they sit behind
`if torch.cuda.is_available():` (a no-op, not a bug), or they are followed by a
`soft_empty_cache`, which DOES release MPS -- see
`comfy/model_management.py`. Their window closes on its own.

That also corrects the story this repo told itself for a day. PBUG-20260908-03
was filed claiming the Metal pool RATCHETS across load cycles, and the log's own
later correction retracts that mechanism: the measured behaviour is a bounded
2x WINDOW, because `load_llm` washes with `soft_empty_cache` before every load.
The fix that PBUG produced is right; the reason it gave is not. Any comment or
docstring still telling the ratchet story is describing something the repo has
disproven.

`_unload_bark` is the exception, and ordering is the whole reason.
`load_llm` performs its wash FIRST and calls `_unload_bark()` AFTER it, so on
the bark path the writer's weights materialize on top of a dead-but-reserved
Bark pool with no wash in between -- both models resident at once on a machine
whose total is 16 GB, where an overrun REBOOTS THE HOST rather than failing the
render.

MEASURED ON THE M4, both directions, before this test existed:
    shipped code:  baseline 0.5 -> allocated 1024.5 -> after teardown 1024.5 MiB
    with the fix:  baseline 0.5 -> allocated 1024.5 -> after teardown    0.5 MiB
The `del` was not quietly doing the work; without the Metal branch nothing
returned the pool.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_LIB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "nodes", "_otr_bark_lib.py")


def _unload_block() -> str:
    """Exactly `_unload_bark`, bounded at the next module-level definition.

    A fixed character window over-reads into the neighbouring helpers, and this
    file's own first run proved it: the `.to("cpu")` assertion below failed on
    a string that belongs to a different function entirely.
    """
    with open(_LIB, encoding="utf-8") as fh:
        src = fh.read()
    start = src.index("def _unload_bark")
    nxt = re.search(r"\n(?:def |class |# -{10,})", src[start:])
    return src[start:start + nxt.start()] if nxt else src[start:]


def test_the_metal_branch_exists_and_cuda_still_runs_first():
    """Structural, so it holds on the NVIDIA boxes that cannot run the measurement."""
    block = _unload_block()
    assert "torch.mps.empty_cache()" in block, (
        "_unload_bark lost its Metal branch; on Apple Silicon it is back to "
        "reaping the Python object while the allocator keeps Bark's pool")
    cuda_at = block.index("torch.cuda.empty_cache()")
    mps_at = block.index("torch.mps.empty_cache()")
    assert cuda_at < mps_at, (
        "the cuda branch must be tested FIRST so an NVIDIA host takes it and "
        "never reaches the Metal branch")
    assert "elif" in block[cuda_at:mps_at], (
        "the Metal release must be an elif, not a second unconditional call -- "
        "a CUDA host would otherwise run both")


def test_the_cuda_call_is_guarded_now_that_a_branch_follows_it():
    """The guard was ADDED here; that is not an add-only diff, so it is pinned.

    On CUDA the condition is True and the call still runs, which is why this is
    behaviour-preserving on the 5080. Without the guard the `elif` could never
    be reached at all.
    """
    block = _unload_block()
    guard = "if getattr(torch, \"cuda\", None) and torch.cuda.is_available():"
    assert guard in block, (
        "the bare torch.cuda.empty_cache() must stay guarded, or the Metal "
        "elif is unreachable and this fix silently reverts")


def test_it_does_not_copy_the_model_to_cpu():
    """A deliberate omission, and it would be a silent CUDA regression.

    `model.to("cpu")` before the release is correct for the LLM teardown, which
    holds a live reference. `_unload_bark` has already `del`-ed its only
    references, so a copy would buy nothing -- and on CUDA it would add a real
    4.2 GB device-to-host copy to every teardown on a machine this fix is not
    for.
    """
    # CODE ONLY. The function's own comment explains why the copy is absent
    # and therefore contains the literal string -- this test's first run
    # matched that comment and failed on the very thing it was asserting.
    code = "\n".join(line.split("#", 1)[0]
                     for line in _unload_block().splitlines())
    assert '.to("cpu")' not in code
    assert ".to('cpu')" not in code


@pytest.mark.skipif(sys.platform != "darwin", reason="Metal allocator only")
def test_the_teardown_actually_returns_unified_memory():
    """The measurement, which is the whole argument.

    Deliberately a 256 MiB stand-in rather than Bark itself: this runs on a
    16 GB machine where the real 4.2 GB model would compete with whatever else
    is resident, and the claim under test is about the ALLOCATOR, not about
    Bark's weights.
    """
    torch = pytest.importorskip("torch")
    if not (getattr(torch, "mps", None) and torch.backends.mps.is_available()):
        pytest.skip("no Metal device")

    import nodes._otr_bark_lib as bl

    def held_mib():
        return torch.mps.driver_allocated_memory() / (1024 ** 2)

    class Stand(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(
                torch.empty(64 * 1024 * 1024, device="mps"))

    saved = bl._BARK_CACHE
    try:
        torch.mps.empty_cache()
        torch.mps.synchronize()
        baseline = held_mib()

        bl._BARK_CACHE = {"model": Stand(), "processor": object(),
                          "device": "mps"}
        torch.mps.synchronize()
        allocated = held_mib()
        assert allocated - baseline > 128, (
            "the stand-in did not reach the Metal allocator (%.1f -> %.1f MiB); "
            "the measurement below would prove nothing"
            % (baseline, allocated))

        bl._unload_bark()
        torch.mps.synchronize()
        after = held_mib()
    finally:
        bl._BARK_CACHE = saved

    assert after - baseline < 64, (
        "_unload_bark left %.1f MiB of unified memory reserved (baseline %.1f, "
        "allocated %.1f, after %.1f). On this platform that pool is the same "
        "RAM the writer is about to load into, and load_llm calls this AFTER "
        "its own wash -- so nothing else will return it."
        % (after - baseline, baseline, allocated, after))
