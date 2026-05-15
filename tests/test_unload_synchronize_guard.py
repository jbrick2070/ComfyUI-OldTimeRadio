"""Regression test for BUG-LOCAL-073 (historical), now a deletion
guard post-S31 B4.

Original bug: in `nodes/story_orchestrator.py:_unload_llm()`,
`model.cpu()` was called without first synchronising the CUDA stream.
If a prior kernel left dirty memory (a pending illegal access not yet
observed), the .cpu() walk would dereference invalid pointers, the
NVIDIA driver would promote the access to a process-level fault, and
the python process would become an unkillable CUDA-locked zombie
holding port 8000.

S30 B4b: production importers (batch_bark_generator, _otr_bark_lib,
scene_sequencer) switched off `_unload_llm` onto
`_otr_model_loader.unload_llm` -- the legacy `_unload_llm` was alive
but no longer the canonical teardown.

S31 B4: `_unload_llm` DELETED from `story_orchestrator.py` (Hard
rule #1A non-deferrable). The BUG-LOCAL-073 synchronize-before-cpu
guard lived inside `_unload_llm`'s body; with the function gone the
guard is structurally retired alongside it. The modern teardown
surface `_otr_model_loader.unload_llm` wraps `model.to("cpu")` in
try/except, which is sufficient for the dirty-context case --
failures are logged and the path continues to `empty_cache` /
`ipc_collect` / `synchronize`, ending in a usable GPU context.

This file post-S31 B4: collapsed to a single deletion-guard test.
The original 10 BUG-LOCAL-073 assertions are obsolete -- there's
no `_unload_llm` body to guard. If a future regression re-creates a
synchronize-first contract bug on the modern teardown path, a NEW
test belongs on `tests/test_loader_slot_primitives.py` against
`_otr_model_loader.unload_llm`.
"""

from __future__ import annotations


def test_unload_llm_deleted_at_s31_b4():
    """Deletion guard: `story_orchestrator._unload_llm` is gone post-S31 B4.
    BUG-LOCAL-073 (the synchronize-before-cpu guard regression) lived
    inside that function's body. Function and bug both retire here.
    """
    from nodes import story_orchestrator as _so

    assert not hasattr(_so, "_unload_llm"), (
        "`_unload_llm` deleted at S31 B4 alongside `_load_llm` / "
        "`_LLM_CACHE` / `_generate_with_llm` (Hard rule #1A "
        "non-deferrable). Canonical teardown surface: "
        "`_otr_model_loader.unload_llm`. BUG-LOCAL-073's "
        "synchronize-before-cpu guard is obsolete -- the modern "
        "teardown wraps model.to('cpu') in try/except, sufficient "
        "for the dirty-context case."
    )
