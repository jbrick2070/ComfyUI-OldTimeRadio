<!-- requested_model: deepseek/deepseek-v4-pro | resolved_model: deepseek/deepseek-v4-pro-20260423 -->

VERDICT: no. The document is a discussion of open questions, not a build-ready plan. It leaves multiple implementation choices unresolved (trimming rule, fallback action, chokepoint location), lacks concrete integration steps, and omits mandatory regression-test definitions.

MUST-FIX BEFORE BUILD:
1. [BUG-264] No deterministic rule for which `key_terms` to keep when the list exceeds `_MAX_KEY_TERMS`. State that the field validator truncates to the **first** `_MAX_KEY_TERMS` items (e.g., `key_terms = key_terms[:7]`). A smart re-ordering can be a future enhancement.
2. [BUG-276] The action for a Bark-routed line missing a `v2/*` preset is undecided (reroute vs assign-default vs skip). Choose and specify: e.g., reroute to the Kokoro announcer engine (the already‑intended engine for announcer lines) and log a warning. If that engine is not loadable, skip the line gracefully (no crash). Document the fallback explicitly.
3. [BUG-276] The “single chokepoint” for the pre‑Bark guard is not identified. Identify the per‑line engine dispatch point (likely `_otr_cast_manager` or the voice‑selector in the rendering pipeline). Implement a check there: if the selected engine is Bark and `voice_preset` does not start with `"v2/"`, override the engine to the fallback engine (Kokoro) **before** `generate_voice` is called. The existing `fail‑close` raise in `eng_bark.py` remains as a last‑resort safety net.
4. [BUG-295] After exhausting compose retries, the plan does not specify what to do with a draft that still contains the leaked multi‑word ALL‑CAPS name. Decide: as a final fallback, strip the offending name from the text (e.g., replace with empty string or a generic `[name removed]` placeholder) to prevent that line from reaching the TTS with the leak. Only accept it unscrubbed if strip would make the line unusably short.
5. [BUG-295] The detection trigger relies on knowing which multi‑word ALL‑CAPS strings are roster names. The fix must dynamically compute the set of ALL‑CAPS roster names (e.g., from `cast` data) and filter for those containing whitespace. That logic must be placed inside the compose retry loop. Specify the source of the roster (likely the same `cast` store used elsewhere).
6. [All] No regression‑test plan is described for any bug. For each bug, create a synthetic input that triggers the old failure path and assert the new behaviour:  
   - BUG‑264: a pydantic test with a `NewsBriefs` input containing >7 `key_terms` and a `script_brief` longer than 350 chars, verifying that the model validates without error and the trimmed fields are used.  
   - BUG‑276: a pipeline test where a line with `char_id='announcer'` and no `voice_preset` is routed to Bark; assert the guard reroutes and no `EngineUnusable` is raised.  
   - BUG‑295: a compose‑loop test injecting a string containing `*ERIN SPENDER the monkeys' enclosure*`; assert the loop retries and the final draft contains no multi‑word ALL‑CAPS roster name (or is scrubbed).  
   Also add a test for each fix to confirm that **clean paths are untouched**, i.e., when all presets are valid / key_terms are within limits / text is clean, the output is byte‑identical to the current baseline (run existing audio‑comparison suite).
7. [BUG-276] The guard must be completely inert on the clean path. Explicitly verify that no change is made to the dispatch logic when `voice_preset` is valid; the existing code path (including the reach of `generate_voice`) must execute exactly as before. This must be confirmed by side‑by‑side tracing or by a test that records and compares the dispatch decisions for a golden workflow.

SHOULD-FIX:
- [BUG-264] Consider truncating `script_brief` at the last word boundary before the 350‑char limit to avoid mid‑word cuts. Not critical, but improves output quality.
- [BUG-276] To avoid forever masking a mis‑configuration, log a single‑episode summary of how many lines were rerouted and which characters were affected. This helps catch cast‑wiring errors later.
- [BUG-295] Set a retry‑count limit higher than 1 for this specific leak (e.g., up to 3 attempts) to reduce the chance of falling back to scrubbing, while still bounding compute.

OPTIONAL / NICE-TO-HAVE:
- [BUG-264] For higher quality, the trimmed `key_terms` could be re‑ordered so that terms matching words in the source text appear first (a V1‑check), but the simpler “first‑N” rule is acceptable for an immediate fix.
- [BUG-276] Add an environment variable (e.g., `OTR_SHOW_REROUTE_WARNINGS=1`) that toggles verbose logging, satisfying PD3 without rewiring any node inputs.

CUT THESE (over-engineering):
- None. All proposed changes are minimal and target the root cause.

[ASSUMPTION] All inferences about the existence of an engine dispatch function and the compose retry loop are based on the bug descriptions; the exact function names and call sites must be verified against the actual codebase (`verify: _otr_line_composer.py` retry loop, `verify: engine dispatch module`). The plan assumes that truncating `key_terms` via a `@field_validator(mode="before")` works with Pydantic v2 `max_length` — this must be tested with the actual model definition (`verify: news_interpreter.py` NewsBriefs class).