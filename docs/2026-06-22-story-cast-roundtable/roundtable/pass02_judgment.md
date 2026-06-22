# R2 judgment log (Claude as judge)
Panel: GPT-5.5, Gemini-3.1-pro, DeepSeek-v4-pro. Spend ~$0.0511. Running total ~$0.10.

ACCEPTED (panel-convergent + grounded):
- Critic scope API `review_ledger(scope_line_ids, neighbor_window)` + reroll
  monotonic-decrease bail. [GPT+DeepSeek+anchor] -> FIX 1.
- Voice postcondition independent of cast_seed; fallback-or-raise, no silent None.
  [GPT+DeepSeek+anchor] -> FIX 2.
- Augment compose with SceneArcContext on LineRequest; do NOT rewrite to scene-level
  prose. [Gemini guard + GPT + DeepSeek] -> FIX 3.
- role_mismatch needs an R3 trace before fix; schema split + invariant matrix +
  migration. [GPT MUST-6/7 + DeepSeek] -> FIX 5 (R3).
- Per-speaker_role critic inclusion rules. [GPT MUST-4] -> FIX 4.

JUDGE CALL (conflict resolved):
- "flat" = deterministic `_is_flat()` code function [GPT/DeepSeek] vs "subjective,
  no code test" [Gemini] -> RESOLVED to rubric-guided critic: the five-dimension
  rubric in the critic PROMPT + a `failed_dimension` structured output the composer
  targets. Not a code algorithm (Gemini right), but consistent + targetable (GPT
  right). -> FIX 4.

VERIFY-AT-BUILD (R3 trace targets):
1. The upstream writer that stamps an engine name into a role/expected field
   (the role_mismatch source -- log showed reviewer "suggested expected='kokoro'").
2. `LineRequest` fields available to carry the SceneArcContext (reuse vs add).
3. The critic call site(s) to thread the `scope_line_ids` arg.
4. Workflow-JSON node IDs for writer/critic/reroll/cast (the wiring surface).

CONVERGENCE: R2 converged on 5 concrete fixes with APIs + sequencing. Advance to R3.
