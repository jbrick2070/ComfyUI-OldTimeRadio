<!-- requested_model: ~openai/gpt-latest | resolved_model: openai/gpt-5.5-20260423 -->

VERDICT: no. The document has the right failure in view, but it is not build-ready because the chosen architectural boundary is unresolved, the current code already violates the stated transport-agnostic constraint, and the plan has no universal schema/field-criticality inventory for applying tolerance safely.

MUST-FIX BEFORE BUILD:

1. [Hard constraints 1] + [_otr_structured_call.py::_invoke_slot] The plan says “NO reliance on a provider’s native JSON-schema/tool mode” and “transport-agnostic,” but the grounded code branches on `getattr(slot_fn, "_otr_openrouter", False)` and passes `response_format={"type": "json_object"}` only for OpenRouter. Even if this is JSON-object mode rather than full JSON-schema mode, it is still provider/transport-specific behavior in the structured layer. Concrete fix: decide explicitly whether provider JSON-object mode is allowed as an optional optimization. If not, remove that branch and keep robustness entirely in prompt/parse/repair/coercion. If yes, amend [Hard constraints 1] to say provider JSON-object mode may be used opportunistically but must not be required for correctness, and add tests proving the same malformed-shape fixture repairs identically without that kwarg.

2. [Candidate levers A/E] + [Open questions 3/4] The plan proposes tolerant field mapping and shape coercion but does not define the canonical schema inventory, per-pass synonym map, or load-bearing field boundaries. This is the central safety line between “benign format variance” and “silent-wrong,” and it is currently an open question. Concrete fix: before implementation, produce a structured-pass inventory: schema name, required fields, load-bearing fields, allowed aliases/synonyms, fields eligible for deterministic default, fields that must still fail. Whitelist exact aliases only; do not implement fuzzy/near-miss snapping until each field has an explicit approved mapping.

3. [Hard constraints 2] + [Candidate lever C] The plan wants schema-in-prompt up front but also requires local byte-identity and says local default prompts must be unchanged. That creates an unresolved architectural split: applying C universally violates byte identity; applying it only remotely violates the model/transport-agnostic story and may make behavior depend on OpenRouter vs local. Concrete fix: do not put schema-in-base-prompt in the first build. Put schema details only into typed repair prompts or into a validation-failure-only path so already-valid local outputs are not touched. If base-prompt schema injection is later needed, gate it behind an explicit opt-in feature flag with byte-identity tests proving default local path unchanged.

4. [Candidate lever B] + [Hard constraints 4] “Relax required->optional-with-default” directly conflicts with fail-loud unless the document names exactly which fields are non-load-bearing and what the deterministic default means semantically. Without that, missing required data can become silent-wrong. Concrete fix: cut broad B from the initial build. Permit defaults only in the schema inventory from item 2, and require a test per default showing the omission is non-semantic.

5. [Current architecture] + [Hard constraints 5] + [Candidate lever F] The plan says the fix must be cross-cutting, including not-yet-migrated hand-rolled passes, but it does not define migration order, compatibility wrapper, or a hard “done” criterion. If A/D/E only land in `structured_call`, the architecture will still fail in the hand-rolled passes. Concrete fix: make F mandatory for the build or define a shared lower-level `parse_validate_with_tolerance` function used by both `structured_call` and hand-rolled passes until migration is complete. Add a checklist of every structured pass and whether it uses the shared path.

6. [What broke] + [Candidate levers A/E/D] The live failure is object-shape/key variance for `normalize_length`, but the plan never shows the target schema or the exact required field that was missing. The proposed aliases `index<->beat_index` are speculative relative to the shown error because the absent required key is not named in the document excerpt beyond “Field required.” Concrete fix: include the failing schema excerpt and the failed JSON fixture for `normalize_length`, then define the exact alias/coercion behavior for that schema. Do not generalize from `index`, `lever`, `beat_index` until the missing canonical field is identified.

7. [Hard constraints 3] + [Candidate lever E] The plan mentions deterministic coercion but leaves “by position/synonym” open. Positional extraction from “whatever shape arrived” is dangerous because JSON object ordering and model-emitted list/object reshaping can encode different semantics. Concrete fix: ban positional mapping in v1 except for explicitly documented list schemas where index semantics are already part of the schema. Use exact-key and exact-whitelist alias mapping only.

SHOULD-FIX:

1. [Current architecture] + [_otr_structured_call.py docstrings] The document describes a 3-rung ladder, but the source has a section comment “Public entrypoint -- the 4-attempt retry ladder” while the code/defaults implement three attempts. This is a correctness/documentation contradiction that will confuse design review and tests. Concrete fix: rename that source comment to “3-attempt retry ladder” or change the implemented ladder, but do not leave both claims.

2. [Current architecture] + [_otr_structured_call.py::_parse_and_validate] The plan frames the current layer as fail-loud/no silent tolerance, but the grounded code already performs deterministic clamping of over-long top-level strings after validation failure. That is tolerance, not pure fail-loud. It may be acceptable, but the problem statement omits it from the candidate model. Concrete fix: add existing `_clamp_overlong_strings` behavior to [Current architecture] and decide whether the new alias/coercion layer follows the same “only after strict validation fails” pattern.

3. [Candidate lever D] D says “Put the literal JSON schema” into typed repair, but the existing `RepairPromptFactory` protocol receives only `original_prompt`, `failed_output`, and `error`; it does not receive `schema`. [ASSUMPTION] Unless the factory closes over schema at each call site, D cannot be implemented cleanly through the current protocol. Concrete fix: either extend `RepairPromptFactory` to accept `schema: type[BaseModel]`, or define a wrapper factory builder that closes over the schema and schema-summary text.

4. [Open questions 1] The plan asks whether “C + D” is enough, but the live failure says “a strong model kept its own format” even after reprompting. More prompting alone may still fail with opinionated models. Concrete fix: make deterministic alias normalization after strict-validation failure part of the minimal path for known benign key variance. Use D to reduce retries, but do not rely on D alone for model-agnosticism.

5. [Hard constraints 2] The byte-identity requirement is stated only for local default models, but the tolerance design must also preserve object identity for any already-valid output from any transport. Concrete fix: specify the invariant as: strict validation is attempted first; if it succeeds, return that instance without normalization, defaulting, clamping, or repair. Only failed validation enters tolerance.

6. [What broke] The cost claim says ~90k tokens were burned on a 420w episode, “much of it” on retries, but the proposed fixes do not include instrumentation to measure retry-token savings or catch future retry storms. Concrete fix: add structured logging counters per helper: attempts, failure class, whether deterministic tolerance fired, whether LLM repair fired, and estimated prompt/output tokens if available. Keep it local/offline-safe.

7. [Current architecture] + [_otr_repair_prompts.py::make_dispatching_repair_factory] Repair dispatch depends on substring matching in error messages such as `"locked cast"`, `"named_character"`, `"too_long"`, and pydantic repr text for payload null. That is brittle at the architecture level if this becomes the universal robustness layer. Concrete fix: standardize post-validator error codes as structured values or stable prefixed strings, e.g. `OTR_ERR_TOO_LONG`, and match those, not prose.

8. [Hard constraints 5] “Cross-cutting” includes UTF-8, SFW, offline-first, and all structured passes, but the candidate levers only address JSON shape. Concrete fix: split non-JSON global constraints out of this plan or add explicit acceptance tests for them. As written, they are requirements with no mechanism.

OPTIONAL / NICE-TO-HAVE:

- Add a small corpus of adversarial model outputs: canonical-valid, alias-key, extra-field, nested-wrapper, null-payload, overlong-string, prose-wrapped JSON, and genuinely unparseable. Use it across every structured pass that opts into tolerance.
- Add a schema-summary helper that renders field names/types from pydantic models for repair prompts only.
- Add a human-readable “why not provider schema mode” note so future contributors do not reintroduce OpenRouter-only fixes.

CUT THESE (scope / over-engineering):

1. [Candidate lever E] Cut broad “pulls the schema’s fields out of whatever shape arrived (by position/synonym)” for v1. It is too broad and invites silent-wrong. Safe to cut because the live failure can be handled by strict-first validation plus explicit whitelist aliases for known key variance.

2. [Candidate lever A] Cut fuzzy “near-miss keys” snapping for v1. Exact aliases like `beat_index -> index` are deterministic and reviewable; fuzzy matching can map semantically different fields if two schema names are similar. Safe to cut because the reported failure used recognizable alternate keys, not typos requiring edit-distance repair.

3. [Candidate lever B] Cut global required-to-optional relaxation. It weakens fail-loud behavior and is unnecessary for field-name variance. Safe to cut because alias normalization can recover present-but-differently-named values without inventing missing content.

4. [Candidate lever C] Cut schema-in-every-base-prompt for the first build. It risks byte-identity drift and token bloat. Safe to cut because repair-time schema guidance plus deterministic alias normalization can address the observed failure without changing already-valid first attempts.

5. [_otr_repair_prompts.py] Avoid adding more bespoke repair-prompt factories until the universal field-alias mechanism and pass inventory exist. The current six/seven classes already show factory proliferation. Safe to cut because schema-field variance should be solved by shared deterministic validation/tolerance, not another per-failure prose prompt.

6. [_otr_structured_call.py::_invoke_slot] Cut or quarantine OpenRouter-only `response_format={"type": "json_object"}` as a correctness dependency. Safe to cut if the parse/repair layer is truly model-agnostic; if retained, it must be documented as optional optimization only, not part of the guarantee.