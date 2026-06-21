<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Guaranteed build/runtime crashes in T1.1 (TypeError) and T1.2 (IndexError), plus a pipeline stall in T3.1.

MUST-FIX BEFORE BUILD:
1. [T1.1] `TypeError` crash on `None`. The formula `max_new_tokens=min(200,max(40,beat_target_words*4))` will throw a TypeError when `beat_target_words` is `None` (which the plan explicitly requires testing). Fix: Update formula to `max_new_tokens = 200 if beat_target_words is None else min(200, max(40, beat_target_words * 4))`.
2. [T1.2] `IndexError` crash on 0-character episodes. The empty case instructions say "force the contract onto the chosen last character slot". If an episode has only announcer/music beats (0 character slots), attempting to access the last character slot will crash. Fix: Add an early return/bypass for the contract if the character slot list is empty.
3. [T3.1] Pipeline stall via post-validator rejection. The plan dictates keeping the existing "post-validator key-term/opposed-wants checks". New arc shapes like `investigation_without_answer` or `slow_dread` will naturally fail standard resolution/opposed-wants checks, causing valid generations to be rejected. Fix: Branch or bypass the opposed-wants validator for non-standard `arc_shape`s.
4. [T1.1] `KeyError`/`NameError` on prompt formatting. [ASSUMPTION] `beat_lo` and `beat_hi` are not currently in the `_build_user_prompt` scope (as hinted by the Open Items section). Interpolating `{beat_lo}-{beat_hi}` will crash. Fix: Explicitly calculate and pass `beat_lo` and `beat_hi` into the prompt formatter dictionary before formatting.

SHOULD-FIX:
1. [T1.3] Infinite recompose loop / Undefined fallback. The deterministic post-check triggers a recompose if a HEDGE_LIST phrase is found. If the recomposed line *also* contains a hedge, the plan lacks a fallback. Fix: Explicitly enforce a max of 1 retry, and define the fallback (e.g., return the original line or strip the phrase).
2. [T2.3] Contradictory acceptance target. The target is `narration_self_address_lines=0`, but the fallback instruction is "fallback to original" if the recompose fails. If it falls back to the original violating line, the target mathematically cannot be 0. Fix: Change the acceptance target to `0 after 1 retry` or implement a safe regex strip for the fallback.
3. [T2.1] JSON schema violation risk. Stating "require them in the _otr_casting contract" implies changing upstream input requirements, which violates the C2 and Wiring invariants (ZERO external JSON edits). Fix: Ensure pronouns are either generated internally or made strictly optional with safe defaults.
4. [Sequencing] Missing T1.4 in the commit order. The sequencing section lists "T1.1, T1.2, T1.3 as three consecutive green chunks" but forgets T1.4, which is part of Sprint 1 and shares a region with T1.1. Fix: Append T1.4 to the Sprint 1 sequence before Sprint 2 begins.
5. [T1.3] Race condition on final character line. [ASSUMPTION] If lines are generated asynchronously or streamed, the final character line may not exist when `compose_announcer_outro` is called. Fix: Explicitly await the final character line generation, or strictly enforce sequential generation before the outro.

OPTIONAL / NICE-TO-HAVE:
- [T2.2] Define the source of `speech_signature`. If it requires an LLM call per cast member, it introduces hidden latency. If it's a static list, specify the list.
- [Measurement] The `length_ratio` target of `>= 0.85` might be mathematically impossible to reach if `beat_target_words` frequently hits the 200-token cap (approx 150 words) but the target expects more. Verify token-to-word math aligns with the 864-word target over 12 legs.

CUT THESE (over-engineering):
1. [T1.4] Ungating "perform indirectly / situation must be different" to *every* character beat. Why: Forcing this strong pivot constraint on every single line will cause the LLM to over-act, hallucinate constant drastic changes, and destroy scene continuity. Limit this rider to costly choices or act turns.