<!-- requested_model: ~anthropic/claude-sonnet-latest | resolved_model: anthropic/claude-4.6-sonnet-20260217 -->

VERDICT: no. BUG-276 proposes a band-aid while its root cause (why a `char_id='announcer'` line is dispatched to an engine whose `roles=("char_voice",)` at all) is never identified; without fixing the root cause the same crash reappears on any future bypass path.

---

## MUST-FIX BEFORE BUILD

**1. [BUG-276 / root-cause missing]** The grounding shows `BarkEngine.roles = ("char_voice",)` and `default_roles = ("char_voice",)`. A correct dispatcher that gates on `roles` should never route an `announcer`-tagged line to Bark regardless of preset status. The plan proposes a pre-Bark reroute guard but never explains _which dispatcher code path_ ignores the roles tuple when `bypass_freeze_halt=True` is active and routes `b018 char_id='announcer'` to Bark anyway. A reroute guard inside `generate_voice` (or just before it) is a symptom patch; the same bug can re-enter via any new bypass path. Fix required: identify the specific caller site that resolves engine for a line (verify: engine registry dispatch / per-line engine selector, not shown in grounding) and add a hard role-membership assertion there (`assert engine.roles intersects line.expected_roles`). Keep the fail-closed gate in `generate_voice` as the backstop only.

**2. [BUG-276 / bypass_freeze_halt interaction]** The soak that triggered the crash explicitly used `bypass_freeze_halt=True`, which bypassed Gate-2 (`apply_deterministic_cast_repairs`). The plan's proposed routing guard must be placed OUTSIDE the freeze-halt bypass path. The plan never states this, and if the guard is implemented in Gate-2 or conditional on the freeze flag, it will be silently skipped under bypass. Fix: document and enforce that the new routing guard is unconditional (not gated on `bypass_freeze_halt`).

**3. [BUG-276 / PD1 undefined behavior on reroute]** The plan proposes rerouting a no-preset announcer line to Kokoro but does not specify: what voice/preset Kokoro uses for the reroute, whether the resulting audio is included in final output, and whether that constitutes "audio that never breaks or degrades" (PD1). Replacing a hard crash with a Kokoro line using an arbitrary default voice is a content change on the dirty path; this must be explicitly scoped. Fix: define the exact fallback preset/voice used on reroute, add a log warning, and annotate clearly that PD1 byte-identity applies only to the clean path (no reroute fires).

**4. [BUG-295 / "bare mid-sentence token" not implementable]** "Bare mid-sentence token" is undefined. The plan cannot be implemented as written. Needed: (a) the exact regex pattern for multi-word ALL-CAPS name detection, (b) whether the check applies only inside `\*[^*]+\*` groups, anywhere in the string, or both, (c) whether the roster checked is the full cast or only names other than the current line's speaker (the existing filter already handles own-name). Without committing to these, two implementers will write incompatible code. Fix: specify regex (e.g., `r'\*[^*]*\b[A-Z]{2,}(?:\s+[A-Z]{2,})+\b[^*]*\*'` for inside-asterisk, separately for bare mid-sentence), specify roster scope (all OTHER characters' multi-word ALL-CAPS names), and commit.

**5. [BUG-264 / `news_close_brief` field unconfirmed]** The fix plan lists `news_close_brief` as a field to truncate. This field name does not appear in the grounding, and the bug description mentions only `script_brief`. [ASSUMPTION] If `news_close_brief` does not exist in `NewsBriefs`, the validator will silently do nothing for it or raise an AttributeError. Fix: verify the exact field names in `NewsBriefs` before writing the validator; do not add a truncation path for an unconfirmed field.

---

## SHOULD-FIX

**1. [BUG-264 / silent truncation]** No logging is specified when the `@model_validator` trims key_terms or truncates script_brief. Silent coercion makes production debugging impossible—if a model degrades and fires this path constantly, there is no signal. Fix: add `logger.warning("NewsBriefs coerced: key_terms %d→%d, script_brief %d→%d chars", ...)` inside the validator.

**2. [BUG-264 / which key_terms to keep is left open]** The plan raises "first N vs source-presence check" but doesn't commit. An open question in a fix plan means two reviewers implement different behavior. Fix: commit to first-N (deterministic, zero cost, consistent with the truncation-is-good-enough principle from BUG-307) and document it explicitly so the choice is visible in code comments.

**3. [BUG-276 / test scope too vague]** The plan requests "a test that asserts no speaker_role='character'/announcer line reaches Bark preset-less" but specifies neither the test file, the mock surface, nor whether it requires a live model. Given `BarkEngine.generate_voice` is reachable without loading Bark (the guard fires before `_load_bark`), a unit test can call `generate_voice(text, voice_preset=None, ...)` directly and assert `EngineUnusable` is raised—that already exists implicitly. The NEW test needed is one that exercises the dispatch layer with a no-preset announcer line and asserts it never reaches `generate_voice`. Verify: what mock is needed for the engine dispatcher, since that code is not shown in grounding.

**4. [BUG-295 / retry budget not specified]** The plan doesn't say whether the multi-word ALL-CAPS check shares the existing compose-loop retry budget or adds to it. If the existing loop already retries N times for other reasons and the new trigger fires on all N attempts with a bad model, the total attempts scale multiplicatively. Fix: explicitly state the new check uses the same existing retry counter (no new budget), and that after exhausting retries the draft is accepted with a warning log, not silently discarded.

**5. [BUG-295 / legitimate ALL-CAPS multi-word name in body]** The plan asks "any legitimate case where a multi-word ALL-CAPS roster name SHOULD appear in body text?" and leaves it open. An announcer line that reads "...and now ERIN SPENDER speaks about..." is a real plausible case that would trigger a false-positive retry. Fix: scope the check to lines where the speaker is NOT the announcer, and only flag when the name appears inside `*...*` (stage-direction bleed is the documented failure mode); bare mid-sentence should require a higher bar (e.g., no surrounding punctuation that would indicate address).

---

## OPTIONAL / NICE-TO-HAVE

- [BUG-264] Use per-field `@field_validator(mode="before")` for key_terms (list slice) and script_brief (string truncation) instead of a single `@model_validator(mode="before")`; this is consistent with BUG-307's existing coerce pattern and is less likely to accidentally suppress other field errors.
- [BUG-276] After the fix ships, add a structured soak entry: run with `bypass_freeze_halt=True` + weak model + a cast that includes an announcer, assert no `EngineUnusable(MALFORMED_CONFIG)` surfaces.
- [BUG-295] Log the leaked name and line id at WARNING level when the retry trigger fires, for corpus analysis.

---

## CUT THESE (over-engineering)

**1. [BUG-264 / V1 source-presence check for key_terms selection]** The plan floats "keep the terms present in source per the V1 check" as an alternative to first-N. This requires a string-search pass over the raw source article per validation attempt. The schema validator is already the performance-critical path (called on every structured-call attempt). First-N is O(1), produces deterministic behavior, and the goal is schema conformance not perfect term curation. Cut the source-presence option entirely; it adds code complexity for marginal quality gain on an already-degraded (weak-model) path.

**2. [BUG-295 / "retry AND scrub" hybrid]** The plan raises "retry vs in-place scrub vs both." Implementing both doubles the code surface: a retry path and a scrub-fallback path with its own regex substitution. The scrub path introduces new artifacts (grammatically broken sentences after name removal). Commit to retry-only within the existing budget; if retries are exhausted, accept with a warning. The scrub fallback is over-engineering for a non-fatal quality issue.

---

**[ASSUMPTION]** All claims about `_otr_ledger_reviewer.py`, the engine dispatcher, `_otr_line_composer.py` L1663-1692, and `NewsBriefs` schema fields are inferred from the bug descriptions alone; none of these files appear in the grounding excerpts. Every claim about those files must be verified against source before implementation.