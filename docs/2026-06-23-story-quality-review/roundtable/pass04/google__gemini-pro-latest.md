<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. The plan is highly converged but contains a Pydantic schema violation and a dependency-injection blocker for the critic.

MUST-FIX BEFORE BUILD:
1. [Data model] `Beat` is a strict Pydantic model (per grounding `_otr_outline.py:84`). It has no `meta` field. Assigning `beat.meta` will raise a validation error. Fix: Remove the "ride beat.meta" ambiguity and mandate the fallback: add the new fields as top-level `Beat` attributes using `Field(default="", exclude=True)` so they are accessible in memory but never serialize to JSON.
2. [L5a] The plan recommends adding `run_story_critic` before the terminal stop (which happens in `_otr_ledger_reviewer.py`). Grounding (`_otr_freeze_cascade.py:756`) shows `run_story_critic` requires `generate_fn`. The reviewer module does not receive the LLM `generate_fn`. Fix: Drop the "move the critic" recommendation entirely; rely solely on scaling `compute_edit_cap` (which is purely mathematical) to prevent the false terminal stops.

SHOULD-FIX:
None — plan converged.

OPTIONAL / NICE-TO-HAVE:
- In L5a telemetry fix (`_otr_ledger_scrub.py:1006`), explicitly note to use `_meta.setdefault("story_quality", {}).update(...)` to preserve any keys injected earlier by the L1/L2 scaffolding.

CUT THESE:
None.

VERIFY-AT-BUILD checklist:
- Confirm the outline->ledger serialization path respects Pydantic's `exclude=True` so the new `Beat` fields do not leak into the frozen ledger.
- Confirm `allowed_people`/`allowed_things` are actually populated at the writer call site today (L1a).