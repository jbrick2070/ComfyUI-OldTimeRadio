<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. Major mechanisms are explicitly deferred/undefined, and §2/§6 contain an acceptance-vs-failure-mode contradiction for b015/b017.

MUST-FIX BEFORE BUILD:
1. [§3 Repair / §7.2] DEFECT 2 has no codable repair path. “outline re-intent vs existing `needs_full_rerun` episode escalation” is still undecided, and the grounding only verifies the current critic/reroll seams, not a usable cascade API for critic-driven full rerun. Concrete fix: choose exactly one mechanism and specify the callable/data flow: critic finding shape -> cascade decision -> rerun trigger -> where the coherence hint is injected -> max rerun count -> final failure behavior. If using `needs_full_rerun`, cite/verify the actual function/field name and define who sets it.

2. [§3 Detection] The planned STANCE/MOTIVATION critic axis has no output contract. Existing `run_story_critic` has 5 craft dimensions plus continuity/voice drift; no stance axis is grounded. Concrete fix: define the exact issue dict/model fields and severity values, e.g. `{dimension:"stance_coherence", severity:"critical", character_id/name, target, prior_stance, new_stance, missing_turn_beat, line_ids, meta:{...}}`, and verify the existing parser accepts extra fields or update it. Do not rely on “meta” unless the critic issue schema actually supports it. [ASSUMPTION] Ledger-line `meta` does not automatically imply critic-result `meta`.

3. [§2 Tier 2/Tier 3 / §6] The plan cannot satisfy “leak count -> 0” for b015/b017. §2 says b015 and b017 are only touched by Tier 2 reroll, and on reroll failure “keeps the draft”; §2 also excludes b017 from the deterministic floor. §6 then requires caught defects to be absent from the final frozen ledger. Concrete fix: define a non-crashing but acceptance-enforceable fallback for unstrippable leaks: e.g. after composer reroll exhaustion, mark a structured hygiene failure and fail CI/acceptance before golden freeze, while production render may ship with LOUD log. Or add an episode/line recomposition escalation with a bounded retry count. Current text is contradictory.

4. [§2 Tier 2] “extend the existing one-shot stage-direction reroll to DETECT trailing + embedded + undelimited” is not implementable as written. The grounded existing API is `detect_leading_stage_business(text)->tuple[bool,str]`, which only detects leading lowercase business via `_leading_stage_strip`. Concrete fix: specify new function signatures and call sites, e.g. `detect_stage_business_for_reroll(text: Any, speaker_name: Any="")->tuple[bool,str]`, imported by `_otr_line_composer.compose_line` at the existing 2015-2060 block, replacing/augmenting `detect_leading_stage_business`. Define whether it returns only a boolean/hint or also spans/reasons for logging/tests.

5. [§2 Tier 3] The deterministic floor classification is underspecified and will not catch the actual corpus if it reuses the existing narration verb list. Grounding `_NARRATION_VERBS` lacks observed verbs/actions: `adjusts`, `clutches`, `taps`, `tightens`, `overrides`, and “fingers dancing”. Concrete fix: define the exact high-confidence action classifier vocabulary/patterns and include positive tests for b005/b010/b012 plus negative tests named in §2. If using a verb list, add the corpus verbs and keep it narrow; if using parsing/regex, specify the patterns.

6. [§2 Tier 3] Quote-boundary handling is not specified enough to code safely. The plan says “outside a matched quote pair” but does not define quote characters, escaping, apostrophes, nested quotes, scare-quotes, or malformed b015 behavior. Existing code treats single quotes as possible lead quotes in `_LEAD_QUOTES`; using that naively would interact badly with b011/b014 phrases like `'The Chronicle'`, `'alive'`, and `'frequency'`. Concrete fix: define a deterministic quote scanner, probably structural double quotes only (`"` plus curly double quotes), with apostrophes/single quotes ignored for dialogue-span matching unless explicitly justified. Add tests for b015, quoted titles, scare-quotes, and apostrophes.

7. [§2 Tier 3] `_strip_stage_directions(text)->Tuple[str,bool]` currently strips delimited directions anywhere, then leading bare only. The plan does not specify ordering/idempotence when adding trailing/embedded stripping. Concrete fix: state the exact order: delimited scrub -> high-confidence quote-anchored bare scrub -> existing leading bare floor, or another fixed order. Add idempotence tests: applying `_strip_stage_directions` twice must produce the same text/boolean second pass false.

8. [§4 Runtime coercion] The role coercion rule is not specified as a reusable invariant, so it is easy to implement differently at three sites. Concrete fix: add one helper with a concrete signature, e.g. `_coerce_speaker_role_for_char_id(line: dict, cast_ids: set[str], source: str)->tuple[dict,bool]`, or equivalent in the real model. It must define behavior for `char_id=None`, `char_id=""`, `char_id=="announcer"`, music roles, unknown cast ids, and known cast ids. Call it from `production_ledger.init_lines_from_outline`, `production_ledger.set_lines`, and `_otr_ledger_reviewer` before honoring `expected="announcer"`.

9. [§4 Runtime coercion] “Audit into `meta` or test logs” is too loose for build verification. Concrete fix: pick one runtime representation. If `meta` is used, verify the ledger line model actually has a mutable/serializable `meta` dict, then define keys, e.g. `meta["role_coercion"]={"prev":..., "new":..., "source":..., "reason":...}`. If test logs only, specify the logger name/message substring tests assert on.

10. [§7.4 / §0 / §2] The strong-model NO-OP fixture is undefined, but acceptance depends on “ZERO strips/rerolls fired.” Concrete fix: name the exact fixture path and define counters/assertions: count composer stage-business reroll attempts, deterministic stage-direction strips, role coercions, and full reruns. Without a fixture and counters, this acceptance cannot be automated.

11. [§6] “No-bypass BASELINE re-smoke FIRST” is operator-gated and resets a resident server; it cannot be a normal build prerequisite. Concrete fix: split manual baseline procedure from automated build gates. The build should run deterministic unit/fixture tests and skip/mark manual re-smoke separately.

12. [§3 Repair] Full episode rerun has no resource ceiling. Grounding shows current targeted reroll has `MAX_REROLL_CYCLES=2`; the proposed episode-scope path has no comparable bound. Concrete fix: set a hard max full-rerun count, timeout budget, and deterministic seed derivation. Define what happens if the weak model still returns an unresolved critical stance reversal.

SHOULD-FIX:
1. [§2 Tier 1] “Stage-3 beat prompt instructs PURE SPOKEN WORDS” lacks the actual prompt-builder location. Grounding only verifies `_otr_line_composer.compose_line` and hygiene functions; the prompt seam is not shown. Concrete fix: identify the exact function/file where the line composer prompt is assembled and the exact text to add. Verify it does not alter workflow JSON.

2. [§2 Tier 3] Well-formedness checks are named but not defined. Concrete fix: define `is_well_formed_spoken_line(text)->bool` rules: non-empty after whitespace normalization, balanced structural double quotes, no orphan leading/trailing quote fragments, no doubled spaces, no dangling comma/semicolon created by stripping. Reuse existing `is_truncated` where applicable.

3. [§2 / scripts/story_quality_scan.py] The plan references `story_quality_scan` deltas but does not say whether the scanner imports the same new detector/classifier as the engine. Concrete fix: have the scan use the same detection helpers to avoid tests passing with one detector and production using another.

4. [§3 Detection] The target constraint “central object + protagonist” is ambiguous without field names. Concrete fix: specify how the critic prompt obtains those names/ids from the ledger/outline. If the central object is not a first-class field, define deterministic extraction or pass it explicitly. [ASSUMPTION] No grounded central-object field is provided.

5. [§3 Detection] “missing turn beat” needs an encoding. Concrete fix: require the critic to return either an existing beat/line id where the turn should have occurred or a string reason. Do not require a beat id if the critic cannot map it deterministically.

6. [§4] The CI-only invariant must exclude non-spoken/music rows explicitly. Current invariant says `role=announcer => char_id=="announcer"`; grounding shows music roles exist and may have empty text/ids. Concrete fix: write invariant over allowed roles: if `char_id in cast_ids` then `speaker_role=="character"`; if `speaker_role=="announcer"` then `char_id=="announcer"`; music/sfx handled separately.

7. [§1 / §4] “COERCE, NEVER CRASH” needs logger behavior. Concrete fix: specify log level and logger, e.g. `logger.warning("ROLE_COERCION ...")`, and require no exception in render path but test-only assert helpers may raise.

8. [§2 Tier 3] Punctuation cleanup after stripping needs rules. b005 after stripping should become `"Not before I amplify it. The world deserves to hear this."` or no quotes depending existing conventions; specify expected exact output strings for b005/b010/b012.

9. [§2] Negative fixtures are required but not enumerated. Concrete fix: add exact strings and expected unchanged outputs before implementing destructive strip.

OPTIONAL / NICE-TO-HAVE:
- Add per-line `meta["hygiene_events"]` audit for deterministic strips, with reason/span, if schema allows it.
- Add a single summary metric to the test output: rerolls, deterministic strips, role coercions, full reruns.
- Keep the optional UN scope-jump telemetry out of acceptance unless it is purely read-only.

CUT THESE (over-engineering):
1. [§5] Cut optional UN escalation telemetry for this build. The plan already cuts the gate, and telemetry adds another scanner path without affecting acceptance.
2. [Header / §6] Cut panel attribution and manual operator reset details from the coder kickoff. They do not affect implementation and can live in runbook notes.
3. [§1 Audio lane c] Cut operator-gated golden recapture from automated acceptance. Keep only pre-TTS frozen-ledger text comparison in CI; recapture is a manual release activity, not a coding gate.