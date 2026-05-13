# OTR Manual Smoke Tests

Procedures that require ComfyUI Desktop running locally and can't be
exercised by bare pytest. Each entry is the executable contract for a
gated CI gap.

Format: sprint-id + scenario name + steps + expected results + how to
report failure.

---

## S22.2 -- LLM timeout halts queue before FLUX node

**What this proves.** When an LLM phase times out via
`nodes.story_orchestrator._run_with_timeout`, the cascade raises
`_LLMTimeoutWorkflowPause` (subclass of `_LLMTimeout`) and ComfyUI's
node-execution layer halts the queue before the next visual stage
fires. This prevents a FLUX / LTX / HuMo node from racing the orphan
LLM worker's still-running CUDA kernels (cudaErrorIllegalAddress).

**Setup.**

1. Open ComfyUI Desktop at http://localhost:8000.
2. Load `workflows/otr_scifi_16gb_full.json`.
3. Find the `OTR_LedgerScriptWriter` node (id=1, "Story Writer (LPL v2.0)").
4. Temporarily lower the LLM timeout for the test:
   - Easiest: edit `nodes/story_orchestrator.py::_run_with_timeout`
     and pass `timeout_sec=1.0` to the writer's call (or set via a
     fixture / monkeypatch if you have one wired).
   - Restore after the test.

**Steps.**

1. Queue the workflow.
2. Watch `localhost:8000` stderr / ComfyUI console.
3. Observe: the writer LLM call exceeds the 1-second budget.
4. Observe: stack trace shows `_LLMTimeoutWorkflowPause` (NOT just
   `_LLMTimeout`). The class name is the success signal.
5. Observe: queue halts. Next node downstream (FreezeCascade -> Bark
   -> ... -> FluxPortrait) does NOT execute.
6. Observe: no `cudaErrorIllegalAddress` in the console.
7. Observe: no Python interpreter crash; ComfyUI Desktop UI remains
   responsive.

**Expected end state.**

- Workflow queue is empty (the timeout halted it).
- `OTR_LedgerScriptWriter` node shows red error border in the canvas.
- Console traceback mentions `orphan worker still on GPU` and
  `Re-run the workflow` (from the exception message body).
- VRAM allocation eventually drops back to baseline once the orphan
  worker completes its forward pass and Python's GC releases its
  tensors (5-30 s depending on prompt size).

**If the test fails.**

- Symptom A: stack trace shows `_LLMTimeout` not the subclass.
  -> `_run_with_timeout` was reverted to raise the base class. Check
  `git log nodes/story_orchestrator.py` for an inadvertent revert.
- Symptom B: FluxPortrait runs despite the timeout.
  -> ComfyUI's node-execution layer changed its handling of uncaught
  exceptions. Re-evaluate the assumption documented in
  `_LLMTimeoutWorkflowPause.__doc__`. May need a ComfyUI version pin
  or a different halt mechanism.
- Symptom C: `cudaErrorIllegalAddress` appears.
  -> The cache invalidation in `_run_with_timeout` didn't fire, or
  the workflow halt didn't actually prevent the next CUDA op. Log
  the full traceback to BUG_LOG.md immediately and open a follow-up.

**Restore.**

- Revert the `timeout_sec=1.0` edit.
- `git checkout nodes/story_orchestrator.py` if the edit was uncommitted.

---
