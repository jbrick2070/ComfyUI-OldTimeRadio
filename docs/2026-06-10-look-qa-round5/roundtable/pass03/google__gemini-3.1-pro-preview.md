<!-- requested_model: google/gemini-3.1-pro-preview | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: yes-with-fixes. Solid plan, but F3's anchor placement defeats its own consistency guard.

MUST-FIX BEFORE BUILD:
1. [F3] Guard Defeat: Prepending `subject_anchor` (which contains the appearance and face/speaking tokens) to the text *before* `_prompt_is_consistent` guarantees the guard will always pass, making it impossible to fail object-only prompts. 
   Fix: Run `_prompt_is_consistent` on the raw LLM text *before* prepending the anchor, or have the guard explicitly check the LLM's unanchored contribution. 

SHOULD-FIX:
1. [F2] Trace Observability Gap: `run_episode` cannot magically know the `prompt_source` (e.g., `env` vs `brief+beat`). 
   Fix: Explicitly stamp `req["prompt_source"] = ...` inside `build_request_from_shot` so `run_episode` can read it when building the trace row.
2. [F5] Announcer Cast Lookup: The plan says "resolved from the CAST table by name match ('ANNOUNCER')". `build_execution_plan` does not currently receive the `ledger` directly.
   Fix: Pass `ledger` into `build_execution_plan` so it can iterate `ledger.get("cast", [])` to find the announcer's `char_id` by name.

OPTIONAL / NICE-TO-HAVE:
- [F1] The `_env_int` helper is a nice addition, but consider putting it in `_otr_shared` or `motion_common.py` if other engines might need similar clamping in the future.

CUT THESE (over-engineering):
- None. The fixes are tightly scoped to the operator's directives and the F1-F6 constraints.

[ASSUMPTION] I am assuming `OTR_LedgerScriptWriter.py` (referenced in F4) exists and has the described pre-freeze scrub site, as it was not provided in the grounding excerpts.