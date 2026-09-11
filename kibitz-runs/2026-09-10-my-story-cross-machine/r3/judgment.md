# R3 grounded judgment

Root sole editor/judge. Driver anchor predates dispatch. Gemini3.8FlashHigh and
CursorGrok4.6High both read actual Windows files. No architecture code exists yet.
The diagnostics-only diff remains a separately reviewed deterministic chunk.

## Accepted amendments

- Both reviewers identify abort/reconcile ordering. Writer tail currently calls
  reconcile unconditionally. New code must call abort OR reconcile, with abort
  idempotently retaining its guarded rollback receipt. Guard a later reconcile
  from resealing an aborted window. Do not require a new enum/state framework:
  a stored terminal receipt suffices. Gemini's claim that stamp_transition(None)
  itself deletes the degradation key is false: _otr_content_transition only
  clears its transition key. The incorrect no-op outcome still warrants a guard.
- Cursor clarifies HF root routing: canonical-root resolution must precede
  validate_model_id's cache scan on native HF, but remote/GGUF curated identities
  are recognized before that side effect. Reuse existing suffix normalization
  and catalog routing; do not duplicate a model catalog or change GGUF identity.
- Cursor correctly distinguishes Python string offsets from UTF-8 hash bytes.
  Use one transient SourceDocument per nonempty raw field, Python start/end char
  offsets and an explicit coordinate version. No claim that those offsets index
  bytes. Root independently verified SourceSpan and _assert_tiles semantics.
- Fit capability belongs on every scheduler closure, bound and unbound. Native
  probe and generation share exact CPU preparation including template options.
  Remote/GGUF lack that native tokenizer capability; preserve their existing
  estimated admission, label estimates, and react to actual provider capacity.
- Keep review helper accounting and separate durable source_reviews at existing
  save/finally checkpoints. Whole-row authorization calls count as model calls.
- Additional independent reader Einstein found two real coverage gaps: source
  and candidate marginal coverage is not their pair coverage; evidence must be
  in the actual delivered review packet. See adaptive_coverage_followup.md.
  Full qualification never arises from diagonal-only checks or inferred relevance.
- Additional independent reader Euler inspected installed LMFE0.11.3: root-array
  ListParsingState captures the implicit limit during parser construction. Set
  CharacterLevelParserConfig(max_json_array_length=0) BEFORE construction, then
  set the effective parser config to0 AFTER prefix builder sets its alphabet.
  Both are needed. Test real root and nested arrays above20 plus explicit limits.

## Already required; no new blockers

Gemini must-fix2/3/4/5/6/7 and should-fix1/2/3/6/7 restate explicit R2 final
requirements: B before adaptive D, skip the None tracker, root before admission,
separate raw_completion and reviewer attempts, normalized native pin key, exact
span splicing, structural checks first, true halt reason, message subtype
preservation, context verification and genuine model capacity. Cursor must-fix
3/8/9/10/11 and should-fix1-8 likewise reinforce existing owner/test obligations.
Keep them in the build checks rather than calling unimplemented code a plan bug.

Both visual cache observations are grounded. Current consumers use named fields
of _NormalizedPrompt. Appending a default preserves current attributes/indexing
and old construction, NOT arbitrary six-value unpacking. No such unpacking was
found by repo search. Do not make that false compatibility claim. Current source
receipt must replace prior metadata in cache and fresh branches.

## Rejected or narrowed

- Cursor1 proposes a second author-call retry outside structured_call. Reject.
  Inner structured_call exhausts retryable capacity into StructuredCallFailedError,
  which is NOT in outer _ATTEMPT_ERRORS. Real torch OOM/provider/cancel errors
  are already terminal; prompt_no_room is nonretryable. There is no automatic
  outer creative retry from the proposed inner max2 exhaustion. Keep the existing
  author ladder and explicitly test that unavailable/schema-invalid reviews
  yield uncertain receipts, real resource errors preserve ownership/type, and
  only a grounded conflict returns a PostValidationError-producing string.
  Never unwrap an inner retryable capacity error inside the author validator;
  never feed failed review JSON into author repair. No recursive author validator
  is attached to the review call.
- Cursor2 says the new class marker disables the existing P2/P3 spiral tracker.
  It cannot: those My Story calls are currently unbound and never attach that
  tracker. The explicit provider-capacity contract intentionally removes this
  arbitrary open-string cutoff wherever bound, while keeping repetition checks.
  Do not silently retain a fixed 2048-token story restriction on other opted-in
  authoring callers. Ordinary non-opted-in bound calls retain their guard.
- Gemini minimum64char/depth8 and Cursor hard work-item cap are arbitrary story
  rejection gates. Reject. Use strictly reducing nonempty source splits, reducing
  note groups, bounded repair of non-progress and true envelope/provider failure.
  An iterative work queue avoids Python recursion depth as an accidental limit.
- Gemini dict[str,str] replacement proposal loses duplicate-key detection after
  JSON parsing and wrongly forbids deleting a localized direction. Keep a list
  of typed span-id replacements, enforce exactly one per authorized span, allow
  an empty localized replacement where valid, and validate the final spoken row.
  Whole-row conversion still requires its separate model authorization.
- Do not add a precompiled schema cache: fresh parser/enforcer state is the fix.
  Keep tokenizer preprocessing only. No JSONPath, fixed fact ontology, additional
  visual retry engine, compulsory summary or model-to-meta teardown is justified.

## R4 target

Confirm the complete amended integration contract, especially early/shared
grammar and native capacity work, and identify any genuinely new material.
Stop repeated reviews at convergence. If adaptive semantics has an unresolved
design question, explicitly scope it while releasing independently settled
shared fixes; do not label the whole campaign complete prematurely.
