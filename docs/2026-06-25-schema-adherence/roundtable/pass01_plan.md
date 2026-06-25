# MODEL-AGNOSTIC SCHEMA ADHERENCE -- CONVERGED PLAN (pass01, post-R1)

Synthesized from the R1 anchor + GPT-5.5 / Gemini-3.1-pro / DeepSeek-v4-pro,
each panel claim grounded against nodes/_otr_structured_call.py +
nodes/_otr_repair_prompts.py. Spend R1 ~$0.11.

## THE KEY REFRAME (grounded): extend two proven patterns, don't build a subsystem
The codebase ALREADY ships the load-bearing pattern in two places:
- `cast_membership_repair` resolves a failure class DETERMINISTICALLY (Levenshtein,
  no LLM) and hands `structured_call` a finished instance -> no LLM repair call.
- `_clamp_overlong_strings` (CONFIRMED _otr_structured_call.py:326-356): on a
  ValidationError it clamps over-long string fields to `max_length` and
  re-validates -- "fires solely on a would-fail output -- a good model's result is
  never touched (byte-identical)."
The model-agnostic field fix is ONE MORE instance of this exact pattern.

## THE CORE INVARIANT -- strict-first (this is the byte-identity guarantee)
`_parse_and_validate` already does `schema.model_validate(data)` first. RULE:
**on validation SUCCESS, return the instance unchanged -- no normalization, no
defaulting, no repair.** Every new tolerance fires ONLY inside the
`except ValidationError` arm. So for any input that already validates -- any
model, any transport -- behavior is byte-identical by construction (this is how
`_clamp_overlong_strings` already behaves; we extend the same arm).

## THE BUILD (lean, ordered; all behind strict-first => byte-identical happy path)

1. **Per-pass field taxonomy (do FIRST -- it gates 2 + 5).** One table per
   structured schema: required+load-bearing fields (MUST stay fail-loud), known
   alias/synonym keys (the deterministic map), and any provably non-load-bearing
   field eligible for a deterministic default. No field is defaulted without a
   row here + a test proving the omission is non-semantic. (all 3 panel + DS#2)

2. **Deterministic key-normalization, two complementary forms:**
   - **(a) Native pydantic aliases (PREFERRED where stable).** On the schemas,
     `Field(validation_alias=AliasChoices("beat_index","index",...))` +
     `populate_by_name=True`. Native, declarative, zero-token, fires at validation
     time -- so a model that emits `index` validates on attempt 1. (Gemini SHOULD#1)
   - **(b) `_normalize_field_keys` helper (the cross-cutting fallback), mirroring
     `_clamp_overlong_strings`.** In the `except ValidationError` arm, when the
     errors are `missing` required fields, snap WHITELISTED synonym keys present in
     `data` onto the schema field names, then re-validate. WHITELIST-EXACT ONLY:
     no fuzzy/edit-distance, no positional mapping (a strong CUT from GPT#7 +
     Gemini + DS#4 -- those invite silent-wrong). Deterministic; on-failure only.

3. **Skip the structural-retry rung for a ValidationError (Gemini#3 -- the token
   fix).** Today the ladder is base -> structural retry (SAME prompt, LOWER temp)
   -> typed repair. For format variance the structural rung just makes an
   opinionated model confidently repeat its own keys -> the ~90k-token burn we
   saw. Route a `ValidationError`/`PostValidationError` STRAIGHT to typed repair
   (Attempt 3); keep the structural rung only for `JSONDecodeError` (a real
   sampling glitch). Saves tokens for every model, not just frontier.

4. **Give the typed repair the schema (GPT#3 + Gemini#1 -- CONFIRMED gap).** The
   `RepairPromptFactory` protocol carries only `(original_prompt, failed_output,
   error)` -- no `schema` -- so `schema_field_repair` is "flying blind." Extend the
   protocol (or a wrapper builder that closes over the schema) so the repair turn
   appends `schema.model_json_schema()` (exact field names + types). REPAIR TURN
   ONLY -> zero byte-identity risk. This is the highest-leverage repair upgrade.
   Optionally surface the dropped/rejected keys in the repair error (the
   `extra="ignore"` blind-spot, Gemini#4) -- but do it by COMPUTING the rejected
   keys for the repair text, NOT by flipping schemas to `extra="forbid"` globally
   (that would fail any benign extra on attempt 1 and risk byte-identity).

5. **Cover the hand-rolled passes (GPT#5 + DS#5 + F).** `structured_call` shipped
   "MODULE + TESTS only; call-site migration deferred Sprint 2B onward" -- some
   passes still hand-roll. A guarantee that only covers migrated passes is false.
   Audit every structured pass; reach all of them by EITHER migrating onto
   `structured_call` OR factoring strict-first + `_normalize_field_keys` into a
   shared `parse_validate_tolerant` both paths call. The inventory sets build order.

6. **Offline conformance harness + telemetry (my SHOULD#2 + GPT#6).** A fixture
   corpus of real divergent shapes -- canonical-valid, alias-key (the Opus
   `{index,lever,beat_index}`), extra-field, nested-wrapper, overlong-string,
   prose-wrapped, genuinely-unparseable -- run through parse/repair, asserting each
   VALIDATES or FAILS-LOUD for the right reason. Model-agnostic, no GPU; pins the
   regression. Plus per-helper counters (attempts, failure class, key-normalize
   fired?, LLM repair fired?) so a soak shows the retry tax shrinking.

## GROUNDED CORRECTIONS TO THE PROBLEM STATEMENT
- **Transport-agnostic constraint reworded.** `_invoke_slot` (CONFIRMED :260-269)
  already asks remote OpenRouter slots for `response_format={"type":"json_object"}`
  -- a no-op for local fns, byte-identical local path. This is `json_object` mode
  (a model-agnostic "return valid JSON" request), NOT the banned provider
  `json_schema`/tool mode. New constraint wording: **json_object mode is an ALLOWED
  opportunistic optimization; correctness MUST NOT depend on it (the parse/repair
  ladder still runs); provider json_schema/tool mode stays banned** (provider-
  specific + not universally supported). Do NOT remove the existing branch
  (GPT's "remove it" = MISREAD).
- The ladder is 3 attempts; the entrypoint comment says "4-attempt retry ladder"
  (CONFIRMED :360 vs `_DEFAULT_MAX_ATTEMPTS=3` :67). Fix the comment.

## INVARIANTS (reject any change that breaks one)
strict-first => local byte-identity; whitelist-exact aliases only (no fuzzy /
positional) => deterministic + no silent-wrong; load-bearing fields stay
fail-loud; tolerance reaches the hand-rolled passes too; everything offline-
verifiable; UTF-8 no BOM; SFW; never force a transport/model.

## CUT (with reason)
- Lever B wholesale required->optional (all 3 + anchor): trades fail-loud for
  silent-wrong. Keep only per-field defaults that have a taxonomy row + a test.
- Lever C schema-in-BASE-prompt (all 3 + anchor): byte-identity break + token
  bloat. Schema goes in the REPAIR turn only.
- Lever E broad / fuzzy / positional coercion (GPT#7 + Gemini + DS#4): silent-
  wrong. Whitelist-exact key mapping only.
- More bespoke repair-prompt factories (GPT CUT#5): solve field variance with the
  shared deterministic normalizer, not another prose factory.

## VERIFY-AT-BUILD (UNVERIFIABLE in R1)
- GPT#7: does `make_dispatching_repair_factory` route by prose substring-matching
  the error? If so, standardize stable error codes (e.g. OTR_ERR_MISSING_FIELD)
  and match those, not English. (read _otr_repair_prompts.py dispatch)
- GPT#6: the exact `normalize_length` StorySpine schema + that aliases/normalizer
  hook there; include the real failing fixture in the harness.
- DS ASSUMPTION: does the byte-identity regression corpus cover ALL structured
  passes, or only some? If incomplete, the strict-first gate is necessary but the
  test coverage must be widened.
- Gemini ASSUMPTION: do the base prompts already name the JSON keys in English? If
  not, there is a latent prompt-engineering gap independent of tolerance.
