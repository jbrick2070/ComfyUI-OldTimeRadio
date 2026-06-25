# MODEL-AGNOSTIC SCHEMA ADHERENCE -- PROBLEM STATEMENT (for the roundtable)

## The operator constraint that frames everything
OTR ships to other people. **The operator does NOT control which writer model a
user runs, and will NOT force a local-vs-remote choice** -- a user may plug in a
local in-process model (mistral-nemo), a local Ollama model (gemma), or any remote
OpenRouter model (GPT, Claude, Grok, DeepSeek, Gemini, ...). The structured-JSON
layer must therefore be **model-agnostic**: it must not break or silently degrade
because a writer formats its JSON differently than the schema's authors assumed,
and it must not nudge anyone toward a particular model or transport.

## What broke (live evidence, 2026-06-25)
A frontier-writer run (Claude Opus via openrouter:slot-a, mistral on the technical
calls) exposed the gap. The `normalize_length[openrouter:slot-a]` structured pass
**exhausted its 3-rung retry ladder and soft-failed**:

```
Field required [type=missing, input_value={'index': 14, 'lever': 'S...', 'beat_index': 14}, input_type=dict]
[OTR_StructuredCall] 'normalize_length[...]' attempt 3/3: typed repair at temperature=0.100
[OTR_StructuredCall] 'normalize_length[...]' exhausted the retry ladder after 3 attempt(s)
[OTR_StorySpine] length normalization failed: StructuredCallFailedError(...)
```

Root cause is NOT extra fields (pydantic v2 ignores extras by default). It is
**field-name / shape variance**: Opus emitted its own keys (`index`, `lever`,
`beat_index`, ...) and a key the schema marks REQUIRED was therefore absent. The
re-prompt ladder asked again; a strong model kept its own format; the ladder
exhausted. Consequences: (a) the length-normalization step was skipped -> several
shipped lines ran long / un-trimmed; (b) ~90k tokens burned on a 420w episode,
much of it on the doomed 3-rung retries (cost + latency tax that scales with how
"opinionated" the model is). The pipeline did NOT crash (it soft-failed and
continued) -- but output quality + cost both suffer, and this will hit ANY
sufficiently-capable or simply-different model.

## Current architecture (grounded; `nodes/_otr_structured_call.py`)
- One shared ladder `structured_call(...)`: base attempt -> structural retry
  (LOWER temp, per the "2B principle": raising entropy during repair causes MORE
  structural hallucination) -> typed repair at static 0.10. Fail-loud
  `StructuredCallFailedError` (no silent sentinel).
- `repair_prompt_factory` Protocol (typed factories in `_otr_repair_prompts.py`);
  a factory MAY short-circuit and return a finished pydantic instance itself --
  e.g. `cast_membership_repair` resolves phantom names with a Levenshtein matcher,
  NO LLM call. `default_repair_prompt_factory` prepends a `CRITICAL:` directive
  naming the validation error + echoes the failed output + restates the prompt.
- `post_validator` for CONTENT failures pydantic cannot see (voice preset outside
  the pool, key-terms too few, speaker outside the locked cast).
- Schemas are pydantic v2 `BaseModel`s; some already set `extra="ignore"`
  (news_interpreter), many do not declare a policy.
- NOT every structured pass is migrated onto `structured_call` yet (the module
  docstring notes call-site migration was deferred "Sprint 2B onward"); some
  passes still hand-roll call->parse->validate->repair.
- JSON extraction is centralized in `_otr_json.parse_first_json_object` (handles
  fenced / prose-wrapped output already).

## Hard constraints (a fix that breaks one is rejected)
1. **Model-agnostic + transport-agnostic.** Works identically whether the writer
   is local in-process, local Ollama, or remote OpenRouter. NO reliance on a
   provider's native JSON-schema/tool mode (that is provider-specific AND would
   force a transport). The robustness lives in OUR parse/repair/prompt layer.
2. **Local byte-identity preserved.** The local default models currently produce
   schema-valid output; the existing regression corpus + `test_audio_byte_
   identical` must stay green. Changes must be byte-identical for inputs that
   already validate today (no reshaping the happy path).
3. **Determinism.** Seed-keyed; no new entropy. A given (model, prompt, seed)
   yields the same parse/repair decision every run.
4. **Fail-loud, never silent-wrong.** A genuinely unparseable result still raises;
   we are not papering over real failures, only over benign format variance.
5. 100% local-capable, offline-first; UTF-8 no BOM; SFW; cross-cutting (every
   structured pass, incl. the not-yet-migrated hand-rolled ones).

## Candidate levers (NOT yet chosen -- the roundtable's job is to converge)
- **A. Tolerant field mapping / aliasing.** pydantic `Field(alias=...)` +
  `populate_by_name=True` + a small synonym map (index<->beat_index, etc.); a
  pre-validation key-normalizer that snaps near-miss keys to schema fields.
- **B. Relax required->optional-with-default** for non-load-bearing fields, so a
  missing field defaults instead of failing the whole object. (Risk: hides real
  omissions; must keep load-bearing fields required.)
- **C. Schema-in-the-prompt up front.** Emit the exact field list + a one-line
  example object in the BASE prompt for every structured pass, so any model knows
  the keys before its first attempt (cuts retries for ALL models, not just on
  repair). (Risk: prompt-token cost; possible byte-identity shift on the local
  happy path -- must gate so the local default prompt is unchanged.)
- **D. Smarter typed repair.** Put the literal JSON schema (field names+types) +
  the specific missing key into the repair turn, not just the error string.
- **E. Lenient extractor pre-pass.** Before strict validation, a deterministic
  coercion that pulls the schema's fields out of whatever shape arrived (by
  position/synonym), THEN validate. Closest to "never fail on shape, only on
  semantics."
- **F. Migrate the stragglers** onto `structured_call` so the hardening is
  universal (some passes still hand-roll).

## Open questions for the panel
1. What is the **minimal, highest-leverage** combination (vs. over-engineering)?
   Is the right answer "C + D" (prompt the schema up front + repair with the
   schema), or does it need "A/E" (tolerant mapping) to truly be model-agnostic?
2. How do we add tolerance WITHOUT breaking local byte-identity -- i.e., make the
   new behavior a no-op for inputs that already validate? (gating strategy)
3. Where does tolerance end and fail-loud begin -- which fields are load-bearing
   enough that a missing value MUST still fail rather than default?
4. Is a pre-validation key-normalizer (E) safe + deterministic, or does fuzzy key
   mapping invite silent-wrong? Whitelist-only synonyms?
5. Does schema-in-prompt (C) belong in EVERY base prompt, or only as the repair
   escalation (D), to bound token cost?
6. Anything we are missing about making a multi-model structured layer robust that
   the OTR ladder does not already have?

## Out of scope
Prose quality / which model writes best (separate question); the coda-bridge
validator strictness; the news-brief "Central object" artifact (separate cleanups).
