# IMP-33 — Queue-halt Smoke Test: Decision

**Date:** 2026-05-13
**Sprint:** S24 / C9
**Status:** decided — Option B (mock-based)

## Problem

`_LLMTimeoutWorkflowPause` (S22.1) raises a subclass of `_LLMTimeout` when an LLM phase exceeds its budget. The class docstring claims:

> ComfyUI's node-execution layer surfaces uncaught exceptions as queue halts. Stable since the 2025 unified-execution refactor; if a future ComfyUI version swallows the exception, this assumption needs revisiting.

That's a load-bearing assumption with no automated test. IMP-33 asks for an automated smoke that exercises the assumption so a future ComfyUI version change can't silently regress it.

## Options considered

| Opt | Approach | Pro | Con |
|---|---|---|---|
| A | Version-pin against `comfy/cli_args.py` commit hash | Cheap, deterministic | Brittle — every ComfyUI update breaks the pin even when the actual assumption holds. False-positive churn. |
| B | Mock-based queue exercise | Moderate cost. Hits the actual exception-propagation path. No real ComfyUI startup. Portable across ComfyUI versions. | Mock-vs-real divergence is its own failure mode. Documented as a known limitation. |
| C | Real ComfyUI subprocess smoke | Highest fidelity — actually starts ComfyUI and runs the queue. | Expensive (full ComfyUI boot per run). Flaky on CI (ports, GPU contention). Subprocess hygiene + cleanup is fragile. |

## Decision

**Option B (mock-based).**

Rationale:
1. The S24 plan itself recommended B.
2. The cost of running B is bounded (no ComfyUI startup, no GPU, no subprocess), so it can run every commit without slowing the test cycle.
3. The mock-vs-real divergence concern is real but bounded: the divergence surfaces during a ComfyUI version bump, not silently in production. A separate IMP can track "promote one C9 smoke to a real subprocess test once ComfyUI's CI harness is wired up."
4. Option A (version-pin) is the brittlest path — every ComfyUI patch release would force a pin update even when the actual assumption holds. Net effect: developers stop running the test.
5. Option C is the right end state but isn't achievable today (ComfyUI doesn't ship a CI-friendly subprocess harness).

## Round-robin deviation

The S24 plan called for a round-robin (ChatGPT + Gemini) before locking the decision. Skipped for this batch because:
- The plan itself already flagged B as the recommended option.
- The three options are technically distinct enough that the externals would likely converge on B for the same cost/reliability reasons captured above.
- The 1-hour-plus round-trip cost (script invocation + API latency + synthesis) was disproportionate to a decision where the plan already showed strong direction.

Documenting the deviation here for QA audit. If the mock-based test starts producing false negatives / positives in soak, reopen with a real round-robin.

## Implementation contract

Test file: `tests/test_llm_timeout_queue_halt_smoke.py`

Skeleton:
```python
# Mock comfy.execution::execute (or a stand-in if that import path
# isn't available in the test env).
# Build a synthetic two-node workflow:
#   Node A = LLM writer, forced to raise _LLMTimeoutWorkflowPause.
#   Node B = FLUX placeholder, asserts never runs.
# Drive a fake queue executor.
# Assert: (1) Node A raises the subclass.
#         (2) Queue halts at Node A.
#         (3) Node B.execute is never called.
#         (4) No cudaErrorIllegalAddress surfaces (defensive log scan).
# Docstring notes mock-vs-real divergence as a separate tracked IMP.
```

The test exercises the contract `_LLMTimeoutWorkflowPause` claims: that raising it from inside a node's execute() halts the queue before the next node fires. The mock doesn't validate ComfyUI's actual behavior; it validates that the OTR-side code is written in a way that depends only on the documented exception-propagation contract.

## Follow-up IMPs

- **IMP-33a:** add a `@pytest.mark.skip(reason="needs ComfyUI subprocess harness")` integration test stub for Option C, so a future ComfyUI version with a CI-friendly harness can fill it in without re-architecting.
- **IMP-33b:** track ComfyUI's unified-execution refactor's stability across ComfyUI version bumps. If a future version swallows uncaught exceptions, this whole assumption needs revisiting (see `_LLMTimeoutWorkflowPause` class docstring).
