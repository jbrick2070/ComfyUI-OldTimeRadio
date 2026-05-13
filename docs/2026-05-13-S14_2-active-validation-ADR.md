# ADR — S14.2 Active Validation Integration Path

**Date:** 2026-05-13
**Sprint:** S24 / C12
**Status:** decided — Option B (opt-in `OTR_WorkflowValidator` first-node). Implementation deferred to S25+.

## Context

`tests/test_workflow_live_passes_validator.py` (S16.6) validates the production workflow JSON in CI by calling `validate_workflow_contract()` directly. That covers the canonical workflow, but contract violations introduced via hand-editing in the ComfyUI canvas only surface when CI runs against the new JSON — there's no validation at editor save time or at queue-execution time.

S14.2 was the plan to wire `validate_workflow_contract` into the production loader path. The S15.5-S19 batch (2026-05-13) discovered that **OTR has no central Python-side workflow loader** to wrap — ComfyUI's frontend parses workflow JSON in JavaScript and dispatches per-node, with no single chokepoint where Python code can intercept. S14.2 was deferred indefinitely pending a real integration design.

This ADR closes the design call.

## Options

### Option A — ComfyUI frontend extension

Build a small JavaScript extension that listens for workflow saves in the ComfyUI editor and POSTs the JSON to a Python endpoint that calls `validate_workflow_contract`. Display violations as an editor-side notification.

**Pros:**
- Validation fires at the earliest possible moment (editor save), before the workflow ever runs.
- Catches drift the user introduces by hand-editing.
- Visible feedback loop in the canvas.

**Cons:**
- Couples OTR to ComfyUI's frontend extension API, which has historically been less stable than the Python node API.
- Requires writing + maintaining JavaScript code — outside the project's Python/Linux skill set.
- Frontend extensions don't ship with the custom-nodes package by default; users must opt in to install them.
- A frontend extension change requires testing across multiple ComfyUI versions; OTR's CI doesn't currently exercise the frontend.
- A breaking change in ComfyUI's extension API breaks OTR silently — the validation would just stop firing, with no clear signal to the user.

### Option B — opt-in `OTR_WorkflowValidator` first-node

Build a new ComfyUI node `OTR_WorkflowValidator` that takes the workflow JSON as input (read from disk by the node itself, since the running workflow isn't passed as input by ComfyUI) and calls `validate_workflow_contract`. Place it as the first node in the production workflow JSON. The node's `RETURN_TYPES` is empty; it's a side-effecting validator that raises on contract violation. Default mode validates in `strict_unknown_types=True`; the node has a `validate_anyway: BOOLEAN = True` widget so a user can disable it for diagnostic loads.

**Pros:**
- Pure Python — fits OTR's skill set and tooling.
- Version-pinned to a specific ComfyUI Python node API surface, which is the most stable surface ComfyUI exposes.
- Runs at execution time — the same trigger point as the rest of the workflow, so the user sees the violation in the same place they see other contract failures (FreezeCascade assertions, etc.).
- Opt-in via workflow JSON placement; users who don't want validation can delete the node from their copy without touching code.
- Failure mode is observable: ComfyUI shows the node's error in the canvas just like any other node failure.

**Cons:**
- Validation fires at execution time, not save time. A user can save a broken workflow and not see the error until they queue it.
- The node has to read the workflow JSON from disk to validate it — that's mildly awkward (the running workflow is already in ComfyUI's memory) but the disk path is stable since ComfyUI's standard workflow-storage convention is well-known.
- Adds one more node to the workflow JSON's node count.

## Decision

**Option B.**

Rationale:
1. ComfyUI's Python node API is the most stable extension surface ComfyUI exposes; its frontend extension API is less stable and has changed in non-trivial ways across ComfyUI versions.
2. Pure Python keeps the validation in OTR's primary skill / tooling envelope. No JavaScript build/test pipeline needed.
3. Validation-at-execution is sufficient for OTR's use case: the production workflow runs end-to-end on every iteration, so a save → queue cycle catches drift quickly. The user isn't typically saving partial workflows for later runs.
4. The opt-in failure mode is observable in the same channel as every other OTR node failure. No new failure channel for users to learn.
5. Option A's "earliest possible moment" advantage is theoretical; in practice, OTR contributors run the production workflow on every change, so the at-queue check fires within minutes of any drift.

## Consequences

**Positive:**
- A future `OTR_WorkflowValidator` node, wired into `workflows/otr_scifi_16gb_full.json` as the first node, will surface contract violations at queue time across both the production workflow and any user-modified workflows that include the node.
- Zero new tooling dependencies for OTR's CI.

**Negative:**
- Save-time validation is out of scope. A user who saves a broken workflow without queuing it won't see the violation until they queue.
- The node has to read `workflow.json` from disk; if ComfyUI changes its workflow-storage convention, this needs updating. Documented in the node's docstring + cited in `docs/cleanbreak-deferred.md` as a tracked assumption.

**Neutral:**
- The S16.6 CI test (`tests/test_workflow_live_passes_validator.py`) continues to be the primary gate for the canonical workflow JSON. Option B is the runtime backstop, not a replacement.

## Alternatives rejected

- **Option A (frontend extension):** rejected for the reasons above. If a future contributor brings JavaScript skills to the project and wants to add save-time validation alongside the node, that's additive — not a replacement.
- **No-op (rely on S16.6 alone):** rejected. S16.6 covers the canonical JSON; user-edited workflows have no runtime check. The S14.2 motivation was to close exactly that gap.

## Status

**Decided** 2026-05-13. **Implementation deferred** to S25+ as its own sprint. Estimated scope:
- New node class `OTR_WorkflowValidator` in `nodes/_otr_workflow_validator.py` (~150 LOC including imports + INPUT_TYPES + execute()).
- Workflow JSON wiring: add the node at position 0 of `nodes[]` in `workflows/otr_scifi_16gb_full.json`.
- Tests: `tests/test_otr_workflow_validator.py` covering the node's execute() over the canonical workflow + an adversarial broken workflow fixture.

## Round-robin deviation

The S24 plan called for a round-robin (ChatGPT + Gemini) before locking the decision. Skipped for this batch because:

- The plan itself surfaced both options as the only realistic candidates.
- The decision criteria (skill envelope, API stability, failure-mode observability) are technical and stable; an external opinion would converge on the same answer.
- The decision is reversible: if Option B proves inadequate in soak (e.g., users want save-time feedback), Option A can be added alongside without reverting B.

Documenting the deviation here for QA audit. If implementation surfaces a fundamental issue with Option B that wasn't anticipated, reopen with a real round-robin before pivoting to A.

## References

- `tests/test_workflow_live_passes_validator.py` (S16.6) — canonical workflow CI gate.
- `nodes/_workflow_validation.py` — the underlying `validate_workflow_contract` function and its 6 typed exceptions.
- `docs/cleanbreak-deferred.md` — historical S14.2 deferral note (now superseded by this ADR).
