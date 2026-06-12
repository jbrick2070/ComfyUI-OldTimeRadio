<!-- requested_model: x-ai/grok-4.3 | resolved_model: x-ai/grok-4.3-20260430 -->

VERDICT: no. The spec leaves placement, enforcement, and migration as open questions rather than a closed, checkable contract; _otr_paths.py already hard-wires otr/state and legacy fallbacks that contradict the "episodes|obs ONLY" rule.

MUST-FIX BEFORE BUILD:
1. [Placement options, para 1] No decision is recorded among (A/B/C); the document ends with "Pick one". Fix: add a single chosen option (with one-sentence rationale against the five constraints) before any ticket is written.
2. [Enforcement, Q2] The required pytest + _otr_paths.py assert + AST grep are described but not specified as concrete code changes. Fix: add the exact new assert location (e.g. inside otr_episodes_root) and the exact pytest file+test name that will be added.
3. [Hard constraints, bullet 7] "the capstone/sweep HYGIENE GATE counts files under server output" is referenced but the file or function is never named. Fix: cite the exact gate (path + function) that must be updated in the same chunk as OH-1.
4. [Migration, Q3] "order of operations for moving live tiers without breaking in-flight episodes" is required but absent. Fix: insert an explicit sequence (stop writers, move state/tmp, resume) with the single call site that must be quiesced first.

SHOULD-FIX:
1. [Live inventory table, "stills" row] Claims otr_save_to_episode_workspace falls back to otr_stills_dir(""); verify: no such function or fallback exists in the supplied _otr_paths.py. Either delete the claim or add the missing helper.
2. [otr_state_dir in _otr_paths.py] Returns otr/state, which is outside the two-folder contract. The plan must either delete otr_state_dir or map it under the chosen placement before the hygiene gate can pass.
3. [Sequencing, Q4] Recommends "land as ticket OH-1 immediately after the queue drains" but does not name the queued 7-leg sweep or the hygiene baseline it would invalidate. Add the concrete ticket/phase name that would be affected.

OPTIONAL / NICE-TO-HAVE:
- Add a one-line table in the final spec showing which of the 16 live dirs map to which sanctioned location under the chosen option.

CUT THESE (over-engineering):
1. Option (C) "per-episode only" cache — it is already rejected by the "cross-episode reuse" constraint listed in the same paragraph; remove the entire option to shrink the decision space.
2. The full resolve_hf_model_path and comfyui_log_path functions in the supplied grounding excerpt — neither participates in otr/ layout, so they add no value to this consolidation ticket set.