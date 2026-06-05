<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document still has unresolved implementation choices and includes unsafe options (especially BUG-276 skip/default/reroute ambiguity) that can break PD1 or mask cast-lock defects.

MUST-FIX BEFORE BUILD:
1. [BUG-276 Proposed fix direction] “rerouted or skipped-with-warning, NOT raise” is not build-safe. Skipping a line shortens the narrative and violates PD1; assigning/rerouting every preset-less Bark line can mask real malformed character cast config. Concrete fix: implement a pre-Bark guard with explicit branching:
   - if the line is announcer-tagged / resolves to the announcer row, route it to the announcer/Kokoro path before any Bark prepare/generate call;
   - if it is a character line resolved to Bark without a `v2/*` preset, do not skip or invent a preset; raise/halt with the existing malformed-config semantics before audio generation, with line id/char_id/role in the error.
   Keep `eng_bark.py`’s fail-closed check as the final safety net.

2. [BUG-276 Proposed fix direction] The “safest single chokepoint” is not identified. The only grounded code is `BarkEngine.generate_voice()`, where the invalid preset raises after dispatch has already selected Bark. A reroute cannot be implemented correctly inside `generate_voice()` because it has no grounded access to line role/cast routing and would be too late to choose Kokoro. Concrete fix: name and modify the actual per-line dispatch/routing function that selects the engine and maps `voice_ref_field = "voice_preset"` into Bark’s `voice_preset` argument. [ASSUMPTION] Verify the dispatch function and ensure the guard runs before Bark `prepare_text()` and before `generate_voice()`.

3. [BUG-276 / Hard constraints PD1] The plan does not prove the guard is inert on clean paths. Concrete fix: add a regression test with a valid Bark character line containing `voice_preset="v2/..."` and assert the same engine/ref/text reaches Bark as before; separately run the project’s audio byte-identity gate on a known-clean workflow. Also add a negative test that monkeypatches/stubs Bark and asserts no preset-less announcer line reaches `BarkEngine.generate_voice()`.

4. [BUG-264 Proposed fix] “so the schema always validates on attempt 1” is false. Trimming `key_terms` count and truncating `script_brief` only fixes length violations; missing fields, wrong types, invalid JSON, or other field constraints can still fail. Concrete fix: phrase and implement this as length-only coercion. Validators must only coerce:
   - `key_terms`: if it is a list, keep the first `_MAX_KEY_TERMS`; leave non-lists to normal validation.
   - individual key-term strings: keep existing BUG-307 behavior.
   - `script_brief`: if it is a string, truncate to `_MAX_SCRIPT_BRIEF_CHARS`; leave non-strings to normal validation.
   Do not swallow unrelated validation errors.

5. [BUG-264 Proposed fix] `news_close_brief` is introduced in the proposed fix, but the bug statement only grounds caps for `key_terms` and `script_brief`. Concrete fix: verify the actual `NewsBriefs` fields and constants in `nodes/news_interpreter.py`. If `news_close_brief` exists and has a defined cap, truncate it under that cap; otherwise remove it from the plan.

6. [BUG-295 Proposed fix] The leak detector is underspecified and will be hard to test. “ALL-CAPS MULTI-WORD roster name appears inside `*...*` group or as a bare mid-sentence token” needs exact matching rules. Concrete fix: define the predicate before build:
   - scan only the cleaned spoken body, not speaker labels or metadata;
   - build candidates from canonical roster names with at least two words;
   - compare against their uppercase form;
   - require token boundaries so `ERIN SPENDER` does not match inside a larger token;
   - include matches inside `*...*`;
   - include bare body matches not occupying the entire line;
   - preserve existing allowance for one-word names such as `Maeve.`.

7. [BUG-295 Proposed fix] Retry behavior has no terminal rule. If every retry contains the same leak, the bug can still ship, or an unbounded loop can be introduced if wired incorrectly. Concrete fix: use the existing compose retry budget only; on leak, reject that draft and retry. After the budget is exhausted, either fail the line cleanly for a rerun/halt or apply a narrowly logged deterministic scrub. Do not silently accept a leaked final line if the regression test is intended to assert “no leak.”

8. [BUG-295 / Hard constraints PD1] In-place scrubbing can create worse spoken text, as shown by the examples: removing `ERIN SPENDER` from “safe in the ERIN SPENDER” leaves “safe in the”. Concrete fix: prefer retry over scrub. If a fallback scrub is implemented, restrict it to removing leaked names inside stage-direction `*...*` groups; for bare mid-sentence body leaks, fail/retry rather than producing malformed narration. [ASSUMPTION] This assumes the line composer can request another draft at that point.

SHOULD-FIX:
1. [BUG-264 Questions] Choose “first N, preserve order” for `key_terms`. Source-presence ranking sounds better but likely requires extra context in the schema validator, which is not grounded here. Concrete fix: trim deterministically to the first `_MAX_KEY_TERMS`, log/debug-count the coercion, and leave source-aware ranking out unless the source text is already available at validation time.

2. [BUG-264 Proposed fix] Truncating `script_brief` can cut mid-sentence. Concrete fix: smallest safe implementation is hard cap truncation only; optionally prefer the last whitespace before the cap if it does not exceed the cap. Do not add model retries just to make the prose prettier.

3. [BUG-264 Tests] Add tests for:
   - 10 `key_terms` validates and returns exactly 7;
   - overlong individual key-term string still truncates under BUG-307 behavior;
   - overlong `script_brief` validates and is capped;
   - invalid non-list `key_terms` still fails;
   - clean in-cap payload is byte-for-byte/model-dump identical before/after validation.

4. [BUG-276 family] The soak used `bypass_freeze_halt=True`, so any fix relying only on `needs_full_rerun` or freeze-halt behavior can be bypassed again. Concrete fix: place the guard in the non-bypassable render dispatch path immediately before engine invocation.

5. [BUG-276 Tests] Add two separate tests:
   - announcer-tagged line with no `v2/*` preset is routed to announcer/Kokoro and Bark is not called;
   - character-role Bark line with no `v2/*` preset fails before Bark generation with a malformed-config/rerun-required error, not skip/default voice.

6. [BUG-295 Tests] Add synthetic tests using roster `["ERIN SPENDER", "MAEVE"]`:
   - reject/retry `*ERIN SPENDER the monkeys' enclosure*`;
   - reject/retry `safe in the ERIN SPENDER`;
   - allow `Maeve.`;
   - allow non-roster all-caps multi-word phrases;
   - verify title-case or normal prose mentions are treated according to the final rule, not accidentally caught by an uppercase normalization bug.

7. [Hard constraints PD3/PD6] The BUG-264 and BUG-295 fixes should not add widgets/sockets/model-pick controls. Concrete fix: keep all thresholds as existing constants or env vars only. No `INPUT_TYPES` changes unless workflow JSON rewiring is explicitly included.

OPTIONAL / NICE-TO-HAVE:
- [BUG-264] Emit a structured debug metric when coercion trims list count or brief length, so weak-model degradation is visible without failing the run.
- [BUG-295] Log the matched roster name and line id on retry, but do not log full generated content if that would make noisy test fixtures.
- [BUG-276] Include the resolved engine name, `voice_ref_field`, and raw preset in the guard error for malformed character lines.

CUT THESE (over-engineering):
1. [BUG-276 Proposed fix direction] Cut “assign-a-default-v2-preset.” It hides malformed cast config, changes the character voice, and risks PD1/audio identity on anything except the explicitly broken path.

2. [BUG-276 Proposed fix direction] Cut “skip-the-line.” It shortens the narrative and directly conflicts with “Audio is king.”

3. [BUG-264 Questions] Cut “prompt-the-model-shorter” as the primary fix. The bug is a deterministic schema overrun; local validators are smaller, offline, and avoid burning the retry ladder.

4. [BUG-264 Questions] Cut source-aware key-term re-ranking unless the source text is already available inside the validator. First-N deterministic trimming closes the validation failure with less wiring risk.

5. [BUG-295 Questions] Cut broad roster filtering for single-word names. The document already identifies legitimate one-word cross-character drama such as `"Maeve."`; keep the new detector scoped to multi-word uppercase roster names.