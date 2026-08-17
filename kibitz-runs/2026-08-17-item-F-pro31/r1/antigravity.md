VERDICT: no. The plan targets a non-existent sampling mechanism, violates the KILL 2 data-starvation constraint, and ignores a second producer (`OTR_LedgerScriptWriter`) that will silently overwrite any partial fix.

MUST-FIX BEFORE BUILD:
1. [Second Producer Overwrite] The plan targets only the initial `SafeOpenBrief` generation. Concrete fix: Apply the exact same fix to the post-composition rewrite in `nodes/OTR_LedgerScriptWriter.py` inside the `announcer_intro_rewrite` helper context, where it calls `_OTRLC.compose_announcer_intro` with a newly constructed `SafeOpenBrief`.
2. [Data Starvation Contract] The plan attempts to pass `source_meta` to a generator that is deliberately starved to prevent spoilers (KILL 2). Concrete fix: Do not pass the full metadata record. If using Shape A, thread only `work_title` into `OutlineRequest` and `SafeOpenBrief` to minimize the spoiler surface.
3. [False Premise of Sampling] The plan assumes the announcer frame is "sampled independently". Concrete fix: Revise the plan to address the reality that `setting` is an LLM-generated string in `_otr_outline._MacroShape`, not drawn from a static list.

SHOULD-FIX:
1. [Shape C Investigation] `nodes/_otr_passage_selector.py` is indeed dead code (no production callers, only referenced in comments in `_otr_episode_budget.py`). Concrete fix: Verify if `select_passage` was abandoned due to a fatal flaw before attempting to wire it up or proceeding with Shape A.
2. [Acceptance Test] Unit tests cannot catch LLM hallucinations. Concrete fix: The acceptance test must assert that the generated `setting` does not contain proper names outside the `cast` (if Shape B is chosen) or assert the prompt contains `work_title` (if Shape A).

CUT THESE:
1. Threading the full `source_meta` record to the announcer generator. Safe to cut because passing the full record violates the `SafeOpenBrief` docstring constraint and risks leaking plot outcomes (KILL 2).

[ASSUMPTION] I assume `work_title` alone (e.g., "The Tempest") does not leak the plot outcome sufficiently to violate the spirit of KILL 2, although as the driver notes, "The Tragedy of Romeo and Juliet" might telegraph an ending.
