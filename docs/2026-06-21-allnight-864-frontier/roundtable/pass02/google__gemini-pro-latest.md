<!-- requested_model: ~google/gemini-pro-latest | resolved_model: google/gemini-3.1-pro-preview-20260219 -->

VERDICT: no. Breaks statelessness, conflates variables, and introduces a brittle text-scrubbing pass that risks corrupting the ledger.

MUST-FIX BEFORE BUILD:
1. [T1.1] Variable scope bug. `max_new_tokens=min(200, max(40, target_words*4))` uses `target_words`. If this is the *episode* target (864, as cited in the acceptance criteria), `864 * 4 = 3456`, meaning the cap is statically stuck at 200 for every line. Fix: Change to `beat_target_words` or `line_target_words`.
2. [T3.3] Stateful local file dependency. "Small local JSON of recently-used descriptors" introduces statefulness across runs. This will cause race conditions in concurrent builds/CI and violates stateless execution. Fix: Pass a history array in memory via the invocation payload, or drop the anti-repeat requirement entirely.
3. [T2.4] Brittle deterministic scrub. Stripping "He paces..." via `_NARRATION_LEAK_REGEXES` directly from the text will inevitably leave malformed punctuation, orphaned quotes, or sentence fragments (e.g., stripping "John says," leaves a dangling lowercase string). If it's a scrub, it risks ledger corruption; if it's a re-roll, it violates Invariant C3. Fix: Enforce this purely via negative prompt constraints in the OUTPUT FORMAT, and drop the regex scrub.
4. [Sprint 0 / T0.2] Unmeasurable baseline. A 6-leg smoke test relying on OS-entropy RNG for cast/style will have too much variance to statistically prove a 0.15 length ratio improvement. Fix: Hardcode a fixed seed-set for the smoke test so before/after runs are apples-to-apples.

SHOULD-FIX:
1. [T1.2] Fallback contract failure. If the slot falls back to `"d001"`, and `d001` is a generic/empty slot, it will not possess the `must_turn` contract required by the audit. Fix: Explicitly inject the `must_turn` contract into the `d001` fallback object if the fallback is triggered.
2. [T1.3] Null final line. Threading "the final character line" into the outro prompt assumes the final beat actually yielded dialogue. Fix: Add a null-check/fallback string for the final line before formatting the outro prompt.
3. [T2.1 / T2.2] Prompt token bloat. Adding gender, pronouns, and speech-register to every character card risks diluting the system prompt's attention on the actual story beats. Fix: Restrict the speech-register cue to a strict maximum of 3 words.

OPTIONAL / NICE-TO-HAVE:
- [T3.1] Add a validation step to ensure the new `arc_shape` templates don't accidentally drop required schema keys (`title`, `premise`, etc.) during LLM generation.

CUT THESE (over-engineering):
1. [T3.2] F9 -- Condition outline on dramatic_state. Cut this entirely. Reordering the macro generation spine (deriving dramatic state before the outline) is a massive structural change that risks breaking the LLM's context chain and violates the spirit of Invariant C4. T1.2 (F2) already solves the costly-choice binding cheaply.

[ASSUMPTION] I am assuming `_NARRATION_LEAK_REGEXES` is a regex replacement tool and not just a boolean matcher. If it's just a matcher, T2.4 implies dropping or re-rolling the line, which explicitly violates Invariant C3 (No QA-only round).
[ASSUMPTION] I am assuming `target_words` in T1.1 refers to the episode target defined in the Acceptance Targets header.