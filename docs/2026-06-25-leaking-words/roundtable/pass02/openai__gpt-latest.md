<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The core verifier is underspecified at the callable/data-model level, and the plan contradicts itself on whether the correctness layer is mandatory/default-off/best-effort.

MUST-FIX BEFORE BUILD:
1. [Layer 2 / Placement vs audio spine] Defect: “MANDATORY deterministic final verifier” conflicts with “ships DEFAULT-OFF / dark” and “default best-effort with telemetry.” Acceptance requires 0 shipped leaks, which is impossible if the layer is off or allowed to ship best-effort by default. Concrete fix: define one mode for the build: e.g. `strict_local_clean=True` in acceptance/CI and release promotion; dark mode only for pre-release telemetry. Add explicit failure behavior: raise/block render after repair budget exhausted when strict is true.

2. [Layer 2] Defect: no implementable public interface is specified for the verifier/repair gate. There is no function signature, return shape, reason-code enum, or integration point into `compose_line` / writer before TTS. Existing grounded APIs return different shapes: `detect_stage_business_for_reroll -> tuple[bool,str,str]`, `sanitize_transcript_text -> tuple[str,list]`, `LineResult(text, compose_flags, validation_findings)`. Concrete fix: add a single API, e.g.:
   `verify_and_repair_line(text: str, req: LineRequest, policy: EntityPolicy, *, strict: bool, repair_fn: Callable | None) -> VerificationResult`
   with fields `text`, `changed`, `defects: tuple[Defect]`, `needs_recompose`, `failed`, `compose_flags`.

3. [Layer 2 / one-repair budget] Defect: “existing one-repair/recompose budget” is not a single existing budget. Grounding shows `compose_line_draft` has `max_attempts`, stage-direction reroll via `_stage_dir_repair_attempted`, quality recursive reroll, and Stage 3 recursive repair via `_stage3_repair_attempted`. Adding four more gates without a central guard can stack multiple recomposes or silently ship after only draft retries. Concrete fix: define a per-line repair budget object/counter shared by all verifier defects, or explicitly thread a new `_leak_repair_attempted` guard through recursive `compose_line` calls.

4. [Stage-direction leak] Defect: the proposed regex is not enough to implement safe extraction. It does not define quote normalization, balanced-vs-unbalanced ordering, whether output keeps/removes wrapper quotes, or what happens with trailing text outside the quote. It can also conflict with the malformed-quote rule if run before quote validation. Concrete fix: implement with existing `segment_double_quotes()` after curly quote normalization. Required order:
   - normalize double quotes;
   - if quote count is odd and quote is internal: defect `malformed_quote`, recompose/fail;
   - if balanced and outside segment 0 matches `^[A-Z][a-z]+(?:ing|ed),\s*$` and segment 1 is non-empty: return segment 1 stripped of wrapper quotes, only if well-formed/non-empty.
   Add reason code `capitalized_participle_before_quote`.

5. [Malformed quotes] Defect: “any INTERNAL odd quote” is ambiguous and not directly codable. Current `sanitize_transcript_text` only drops a single edge wrapper via `_balance_wrapper_quotes`; `segment_double_quotes` counts straight quotes after curly normalization and ignores apostrophes. Concrete fix: define exact predicate:
   `norm = curly_to_straight(text); odd = norm.count('"') % 2 == 1; edge_wrapper = norm.startswith('"') ^ norm.endswith('"') and norm.count('"') == 1; internal_odd = odd and not edge_wrapper`.
   Internal odd should set `needs_recompose=True`; edge wrapper may continue through `sanitize_transcript_text`.

6. [Caps-name vocative] Defect: “ALL-CAPS token that EXACTLY matches a roster full name” is internally inconsistent because full names contain spaces and are not a single token. Existing `scrub_self_vocative(text, speaker_name)` only strips the speaker’s own name, case-insensitive, and `compose_line` does not call it in the shown deterministic strip pipeline. Existing `_scrub_or_flag_roster_leak` explicitly does NOT flag vocatives. Concrete fix: define a full-name phrase matcher over `req.allowed_people`, e.g. `(?<![\w'])FULL\ NAME(?![\w'])`, sorted longest first, and wire it into `compose_line` after `cast_strip` and before `detect_phantom_names`.

7. [Caps-name vocative] Defect: “title-case or drop the vocative” is not deterministic. These produce different spoken text and different acceptance behavior. Concrete fix: choose one policy per shape:
   - leading `FULL NAME[,!:-]+ rest` -> drop vocative;
   - trailing `rest[, ]+FULL NAME[.!?]?$` -> drop vocative and preserve terminal punctuation;
   - if product wants names spoken, title-case instead, but then the acceptance fixture must expect title-cased output, not removal.
   Do not leave this as an implementor choice.

8. [Caps-name vocative / Acceptance gate] Defect: the negative fixture “a legitimate emphatic vocative” can contradict the proposed detector if it is an all-caps roster full name at leading/trailing vocative position. Concrete fix: specify the negative fixture exactly. If `YUKI MARTIN!` is supposed to be legitimate, the proposed detector cannot pass. If only first-name vocatives like `YUKI!` are legitimate, state that the detector only targets full names with spaces.

9. [News-bleed] Defect: required policy data does not exist in grounded `LineRequest`. There are `allowed_roster`, `allowed_people`, `allowed_things`, and `grounded_nouns`, but no `allowed_proper_nouns` / `banned_source_proper_nouns`. “Ledger schema frozen” also means this cannot be added as persisted line schema unless it is transient or stored in existing `meta`/flags. Concrete fix: add a transient dataclass, e.g. `EntityPolicy(allowed: frozenset[str], banned: frozenset[str])`, built before line composition and passed to the verifier. Normalize with `casefold()` and keep phrase boundaries.

10. [News-bleed / build_allowed_roster] Defect: the plan says verify whether `_otr_line_composer.build_allowed_roster` whitelists news/key terms; grounding confirms it does. `build_allowed_roster(..., key_terms=...)` uppercases and merges every key term into `allowed_roster`. If `President Trump` arrives as a key term, the existing phantom-name gate will clear it. Concrete fix: split source news entities from fictionalized premise terms before calling `build_allowed_roster`; pass only fictional/world terms as `key_terms` or `allowed_things`, and pass raw-source entities into `banned_source_proper_nouns`.

11. [News-bleed] Defect: “REQUIRES a news-abstraction step” is not implementable as written. No schema is defined for the abstraction output, no extractor is specified for raw news proper nouns, and no invariant says banned terms cannot also appear in allowed terms. Concrete fix: define output shape:
   `NewsAbstraction(fictional_conflict_objects: tuple[str,...], allowed_world_nouns: tuple[str,...], banned_source_proper_nouns: tuple[str,...])`
   and invariant `banned ∩ allowed == ∅` unless explicitly overridden with an allow reason.

12. [Layer 3 optional online LLM cleaner] Defect: “output JSON `{clean_text,...}`” has no schema, parser, or existing API in the grounded code. `_otr_repair_prompts.py` factories are for `structured_call` repair prompts, not a line-cleaner API; `compose_line`’s `creative_fn` returns plain text. Concrete fix: either cut Layer 3 for this build, or define a Pydantic model and call path through the actual structured-call helper, including JSON parse failure handling and rejection behavior.

13. [Layer 3 optional online LLM cleaner] Defect: “REJECT if … it changes non-target words” is not codable without target span data from Layer 2. The current detectors mostly return booleans/reason strings, not spans. Concrete fix: make every Layer 2 defect return `target_spans: tuple[Span(start:int,end:int,reason:str)]`; then validate diff by requiring all edits to fall inside those spans after normalization.

14. [Acceptance gate] Defect: “0 shipped instances across BOTH lanes” is not tied to any executable command, flag state, or corpus location. Concrete fix: add a deterministic unit test module with fixtures and expected `VerificationResult`, and a writer-level integration test with `strict_local_clean=True` that asserts no line text contains the positive leak patterns after the final pre-TTS pass.

SHOULD-FIX:
1. [Stage-direction leak] The regex only handles one-word participles/adjectival past forms before a quote. It misses common close variants like `Breathless, "..."`, `Still gasping, "..."`, or `Gasping for air, "..."`. If intentionally out of scope, add positive/negative tests proving only the shipped `Gasping, "..."` class is targeted. Otherwise extend structurally but cap prefix length.

2. [Layer 2 / existing sanitizers] Current `compose_line` shown does not call `sanitize_transcript_text`, `scrub_leading_stage_direction`, `strip_quote_anchored_stage_direction`, or `clean_spoken_character_line` in its final deterministic strip pipeline. [ASSUMPTION] They may run in `_otr_ledger_scrub.py`/writer. Verify actual writer order. If not already wired before persistence/TTS, wire the new verifier there, not only in composer drafts.

3. [News-bleed] Define phrase matching carefully. A naive substring banned-noun scan will false-positive on substrings and possessives. Concrete fix: compile banned phrases with `(?<![\w'])... (?![\w'])`, normalize curly apostrophes, and test possessive forms separately if needed.

4. [Layer 2 telemetry] The plan mentions telemetry but not flag names. Existing aggregation counts `compose_flags` by prefix. Concrete fix: emit stable flags like `leak_repair:capitalized_participle_quote`, `leak_recompose:malformed_quote`, `leak_blocked:banned_source_noun`.

5. [Strict local clean] [ASSUMPTION] `_otr_config.py` needs a real accessor/env name for `strict_local_clean`. Add one function, e.g. `strict_local_clean_enabled()`, and avoid reading env vars ad hoc in verifier code.

6. [Layer 2 / prompt line] The upstream prompt line says “no real-world proper names,” but current `_SYSTEM_PROMPT` says “Use only proper nouns listed under NAMED ENTITIES” and later “Ground this line in the news facts.” Without the news-abstraction split, the prompt can still invite source-name bleed. Fix prompt wording only after the allowed/banned data model is in place.

OPTIONAL / NICE-TO-HAVE:
- Add a small `ReasonCode` enum instead of free-form strings so tests do not depend on log text.
- Add a corpus test around curly quotes specifically, because existing `segment_double_quotes` folds `“”` but the proposed regex only partially accounts for them.
- Add a debug log showing before/after text only when changed; avoid logging every clean line.

CUT THESE (over-engineering):
1. [Layer 3 optional online LLM cleaner] Cut for the first build. Layer 2 is supposed to be the mandatory correctness layer; Layer 3 adds JSON schema, diff validation, online/offline mode branching, and another model failure mode without being needed for the four acceptance fixtures.

2. [Frontier writer product-tier recommendation] Cut from the implementation task. The same verifier should run regardless of model; model-tier recommendations do not affect build correctness.

3. [Action-preservation references / split_stage_business telemetry] Already marked CUT; keep it cut. The required fix is spoken-text cleanliness before TTS, not preserving leaked stage business.