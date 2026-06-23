<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. The core mechanism (feeding LLM critique into `diversity_hint`) violates the v0 architecture, and the loop introduces unhandled state mutation and an O(N*M) cost multiplier.

MUST-FIX BEFORE BUILD:
1. [The loop - Step 2] Repurposing `diversity_hint` for LLM critiques breaks v0 determinism. The grounding explicitly defines `diversity_hint` as a deterministic, index-keyed structural instruction drawn from a hardcoded tuple (`_DIVERSITY_HINTS`). Injecting arbitrary LLM text into this field destroys the pure structural variation v0 was built for.
   - Fix: Define a new, separate field in `OutlineRequest` (e.g., `prior_critique`) for v1 feedback. Leave `diversity_hint` strictly for v0's index-based steering.
2. [The loop - Step 3] State corruption via `build_sq_data`. The v0 grounding explicitly warns that `build_sq_data` "MUTATES intent and substitutes the generic crisis nouns". Running this in a loop will permanently corrupt the underlying beat intents for subsequent passes.
   - Fix: The `_build_and_compose` helper MUST deep-copy the outline and any shared state (like `roster` or `meta` if mutated) before applying `build_sq_data`.
3. [Open Questions - A] Unresolved O(N*M) complexity. The plan leaves open whether the outline build "MAY itself be the v0 best-of-N selector". If v1 loops 5 times, and v0 loops 6 times per outline, you are silently triggering 30 outline generations plus 5 full composes per run.
   - Fix: Force a decision now. When `OTR_STORY_REFINE_PASSES >= 2`, explicitly disable the v0 structural best-of-N selector (force `effective_n = 1` for the outline step) to prevent multiplicative bloat.

SHOULD-FIX:
1. [The loop - Step 4] Reusing `_otr_story_critic` across node boundaries. The plan notes the critic lives downstream in `OTR_LedgerFreezeCascade` (a separate node). Pulling its machinery upstream into `OTR_LedgerScriptWriter` risks circular dependencies or heavy refactoring not accounted for in the build order.
   - Fix: Do not attempt to reuse the downstream critic. Build the "new lean rubric prompt" directly in the writer node.

OPTIONAL / NICE-TO-HAVE:
- [The loop - Step 6] The keep-best comparator uses `ungrounded_crisis_density`. Since `build_sq_data` mutates intents (which zeroes out this density, as noted in the v0 grounding), ensure the StoryScore is calculated *before* `build_sq_data` runs in the loop, or the tie-breaker will evaluate to 0 for all passes.

CUT THESE (scope / over-engineering):
1. [Build order - Step 4] Optional remote opt-in + cost guard. Cut for v1. The plan already states v1 is local-only because "a refine pass is a FULL compose = many paid calls". Building remote infrastructure for a feature explicitly designed to be local-only is dead code.

[ASSUMPTION] The plan assumes a weak local LLM (which requires this B+ loop in the first place) is simultaneously capable of generating a highly accurate, actionable critique of its own work that will meaningfully steer the next generation, rather than just hallucinating or repeating itself.