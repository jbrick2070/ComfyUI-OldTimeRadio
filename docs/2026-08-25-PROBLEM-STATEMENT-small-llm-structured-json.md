# PROBLEM STATEMENT -- small LLMs cannot hold JSON syntax on `scifi_news_pro`, and the fix already exists in deleted code

> ## CORRECTED 2026-08-25 AFTER r1 (Cursor). READ THIS BOX BEFORE THE BODY.
>
> The r1 panel confirmed the ORPHANED-HOOK diagnosis but **rejected grammar
> binding as the headline fix**, and caught four errors below. The body is
> kept as written so the reasoning is auditable; these corrections win.
>
> 1. **THE HEADLINE FIX IS WRONG.** Cursor: *"It will parse. That is not the
>    same as the lane working."* Under LMFE the legal minimum for `DossierLLM`
>    is one fact plus three empty buckets, `facts_to_keep` is NEVER
>    source-checked, and a 2B model's highest-probability constrained
>    completion sits near that floor. Grammar would convert tonight's LOUD
>    failure into a SILENT hollow dossier -- a green parse hiding a useless
>    extraction. That is worse, not better.
> 2. **THE OPERATOR'S OWN IDEA IS THE RECOMMENDED FIX** (his words: *"can't we
>    just say write this text, and the Python puts it right in the JSON"*).
>    Cursor's MUST-FIX #2: *"Adopt Python-assembled extract for P0 (and
>    news_read), not grammar-as-the-fix. Model writes labeled lists / prose;
>    Python constructs `DossierLLM` / `NewsCloseRead`. Keep pydantic +
>    `_filter_dossier_entities` + news-read post_validator. One contract for
>    small and big."* That is the operator's stated preference (work for BOTH
>    sizes) and it removes the JSON-packaging burden from the model entirely.
> 3. **"IT FAILED AT `json.loads`" IS WRONG** (section 2 below). `json.loads`
>    never runs. `parse_first_json_object` raises when
>    `extract_first_json_block` returns empty, with position **hardcoded to 0**
>    (`nodes/_otr_json.py:81-96`). So "line 1 column 1 (char 0)" is a SENTINEL,
>    not a location -- it does NOT mean the break is past the logged head. It
>    means "no complete top-level object": incomplete JSON, a malformed fenced
>    block (the fence path fail-closes and does not fall through), a top-level
>    array, or empty output all produce that identical string.
> 4. **"THE PRO LANE IS MAXIMALLY LENIENT ABOUT CONTENT" IS OVERCLAIMED.** Only
>    P0's post_validator is a no-op. `_make_news_read_validator`,
>    `_make_casting_validator` and `_make_pitch_validator` are real content
>    gates that a 2B model will still fail AFTER syntax is fixed. Even P0 is
>    not parse-then-accept: `facts_to_keep min_length=1` and the nonblank
>    `@field_validator` are invisible to any grammar.
> 5. **MY CITED EVIDENCE WAS DESTROYED, BY ME.** Section 7 cites
>    `docs/2026-08-25-llm-image-upscale-sweep-receipt.json`, which now shows
>    7 passed / 0 failed at 0.1 min per leg -- because a later `--dry-run` of
>    the same driver OVERWROTE it. The real failure record is preserved at
>    **`docs/2026-08-25-leg1-dossier-failure-evidence.md`**, extracted from
>    the surviving server log. Nothing may be promoted to `PROD_BUG_LOG.md` or
>    the Bible from the receipt.
> 6. **`_counting` OPACITY IS CROSS-BANK**, not news_pro-only: the same
>    attribute-stripping wrapper exists in `_otr_shakespeare_sources.py`,
>    `_otr_media_archive_interpreter.py`, `_otr_public_domain_sources.py` and
>    `news_interpreter.py`. Fix the pattern, not one function.
> 7. **DO NOT BLANKET-BIND** inside `structured_call`. Cursor enumerated ~10
>    caller schemas that are LMFE-hostile (unbounded strings with
>    `max_new_tokens=None`, models whose legal minimum is empty/default,
>    ledger-clean `max_length` fields where a ValidationError-plus-repair would
>    become a hard mid-sentence clip). Allowlist, dossier first, if at all.
> 8. **GGUF IS A WEAKER CONSTRAINT, NOT AN EQUIVALENT ONE.** The backend
>    downgrades `json_schema` to llama-cpp `{type: json_object, schema: ...}`;
>    `$defs`/`$ref` handling is lossy and `strict` may be ignored. Small GGUF
>    rows can still emit `{}`. Keep the post-hoc ladder.
> 9. **SECTION 6 (upscale) IS CUT FROM THIS ARC** -- different pipeline,
>    different risk, different proof. It shipped separately as its own change.

**Written 2026-08-25 from a live failure.** Operator's two ideas that night --
*"look at that old dead scifi news code to help tech llm repair so we get a
clean ledger"* and the upscale idea (separate, section 6) -- both landed. The
first one landed harder than expected: **the old dead lane contains the exact
reference implementation of the fix the surviving lane needs.**

Nothing here is implemented. This is the statement, not the change.

---

## 1. The live failure

Sweep leg 1 (`docs/2026-08-25-llm-image-upscale-sweep-receipt.json`),
`technical_model = google/gemma-4-E2B-it`, `source_bank = scifi_news_pro`,
act_count 1. The P0 dossier pass failed **all three ladder attempts** -- base
call, structural retry at halved temperature, typed repair -- each with the
identical error:

    no decodable top-level JSON object found: line 1 column 1 (char 0)

Attempt 1 generated **503 tokens against a 700-token budget**
(`_MAX_NEW_TOKENS = {"dossier": 700}`, `nodes/_otr_scifi_news_pro.py:264`), so
it was NOT truncated by the cap -- the model believed it had finished and had
emitted something that is not a decodable JSON object. The log keeps only the
first 400 chars (`_otr_structured_call.py::_raw_head`), which look like
well-formed JSON, so the break is past the visible head.

## 2. THE CRITICAL DISTINCTION -- this is a SYNTAX failure, not a content failure

**`scifi_news_pro` is ALREADY maximally lenient about content**, and this is
the single most important fact for anyone tempted to "relax the contract."

Per `docs/2026-08-15-graduated-extraction-span-reader-enumeration.md` (written
before the rip, still on disk): the pro lane's `_make_dossier_validator`
**returns `None` unconditionally**. It never refuses an extraction for
unverifiability. Unverifiable entities and numbers are DROPPED after the call
by `_filter_dossier_entities`, delete-only, on the reasoning that "no
structural retry can fix knowledge." **There are no spans anywhere in the pro
lane.** That same doc records the pro lane as *already shipping the graduated
behaviour in its strongest form*.

So the model never even reached the validator. It failed at `json.loads`.
**Relaxing anything about WHAT the model says cannot help**, because nothing
about what it said was ever going to be rejected. The only thing killing this
lane on a small model is raw JSON SYNTAX.

**Corollary, and it kills a tempting idea:** reviving the old `scifi_news`
lane for small models would be a step BACKWARD. That lane had the STRICTER
contract -- `FactIndexV4` with `SourceSpanV4` rows whose literal identity
`payload[field][start:end] == quote` was enforced. It is the lane whose
measured `quote_not_literal` failures prompted the graduated-extraction work
in the first place. It is harder on a small model, not easier.

## 3. WHAT THE OLD LANE ACTUALLY HAS THAT PRO DOES NOT -- grammar binding

This is the part worth mining, and it is not the repair logic.

`_otr_scifi_codex.py` (deleted in `dae1fb3c`, recoverable via
`git show dae1fb3c^:nodes/_otr_scifi_codex.py`) carried
`_bind_local_slot_schema` at `:1593`:

```python
def _bind_local_slot_schema(slot_fn, schema_model) -> GenerateFn:
    """Bind a local Transformers scheduler closure to one exact schema."""
    binder = getattr(slot_fn, "_otr_bind_schema", None)
    if not callable(binder):
        return slot_fn
    bound = binder(schema_model)
    if not callable(bound):
        raise CodexPackContractError(
            "local structured slot returned a non-callable schema binding")
    return bound
```

and called it at `:2501-2504`, binding **both** the slot fn and the repair fn:

```python
bound_slot = _bind_local_slot_schema(slot_fn, result_type)
bound_repair_slot = (
    _bind_local_slot_schema(repair_slot_fn, result_type)
    if repair_slot_fn is not None else None)
```

**That is grammar-constrained decoding, per call, on the exact result type.**
With it bound, lm-format-enforcer masks any token that would make the JSON
invalid -- including EOS while the object is still open, which is precisely
how tonight's 503-token attempt died.

**`scifi_news_pro` never had this.** Independently confirmed by a Cursor CLI
investigation the same night, which traced the whole chain and found:

* `_otr_bind_schema` is still ATTACHED by the live scheduler
  (`nodes/OTR_LedgerScriptWriter.py:729-737`, gated on `provider == "local"`).
* It is **read by nothing in production** -- only by
  `tests/test_writer_slot_routing.py:306-349`, which still proves it works.
* The comment above it claiming *"the SciFi structured invoker uses it"* is a
  FOSSIL: that invoker was `_otr_scifi_codex._bind_local_slot_schema`, deleted
  with the lane. `structured_call` never grew a replacement.
* `docs/WRITER_INPUT_MATRIX.md:21-27` already records the consequence
  plainly: *"no shipped lane binds a grammar during decoding any more ... live
  extension space, not a live path."*

**So this was never a decision to leave pro unconstrained. It is an orphaned
capability left behind when its only consumer was retired.** No entry in
`PROD_BUG_LOG.md` or `GO_FORWARD_PLAN.md` says "do not bind grammar on pro."

## 4. A SECOND, INDEPENDENT GAP -- `_counting` strips the markers

Even wiring section 3 naively would not reach the failing passes.

Every `scifi_news_pro` technical pass is wrapped by `_counting`
(`nodes/_otr_scifi_news_pro.py:530-541`) before it is handed to
`structured_call`:

```python
def _counting(slot_fn):
    box = {"calls": 0}
    def _fn(msgs, *, temperature, max_new_tokens):
        box["calls"] += 1
        return slot_fn(msgs, temperature=temperature,
                       max_new_tokens=max_new_tokens)
    return _fn, box
```

That wrapper copies **no** `_otr_*` attributes and accepts **no**
`response_format`. So `_otr_bind_schema` and `_otr_supports_json_object` both
vanish before `structured_call` ever sees the function. Used at `:1579`
(dossier), `:4434` (cast_aliases), `:4441` (news_read), `:4533` (casting).

**Consequence for GGUF rows specifically:** the existing `json_object` force
in `invoke_structured_slot` (`_otr_structured_call.py:696-710`) is proven on
the RAW scheduler closure (`tests/test_gguf_registry.py:418-439`) but is
stripped by `_counting` too -- so the GGUF technical rows in the sweep
(`unsloth/gemma-4-12b-it-GGUF`, `unsloth/Qwen3-8B-GGUF`) are ALSO unconstrained
at sampling on these passes, for a different reason than the transformers row.

## 5. THE SHAPE OF THE FIX (not implemented; needs its own arc)

Cursor's proposal, recorded for the next window to attack rather than adopt:

* **A.** Bind inside `structured_call` once per call, for every attempt: if
  `slot_fn` exposes a callable `_otr_bind_schema` and the bound schema is not
  already `schema`, replace it with `slot_fn._otr_bind_schema(schema)`. Same
  for `repair_slot_fn`. This covers dossier/aliases/news_read/casting and any
  other `structured_call` caller, with no news_pro-specific fork -- and it is
  the shape the deleted lane used (section 3).
* **B.** Make `_counting` a transparent transport: copy `_otr_*` attributes
  and forward `response_format`/`stop`. Without this, A never sees the hook.
* **C.** GGUF is currently second-class: `_build_truncating_generate_fn`
  returns `make_gguf_generate_fn` with sampling only and NO schema, before the
  `if schema_model is not None` block. Pass
  `schema_to_response_format(schema_model)` there (the same mapping
  `make_constrained_generate_fn` already uses) and attach the bind hook for
  `gguf_native`, not just `provider == "local"`.
* **D.** Do NOT touch `OTR_LedgerScriptWriter.py:4407-4415` (the writer's own
  SlotJobFields constrained path -- the one live consumer) and do not bind the
  creative P3 markup path; that is a different design.

**Adversarial caveats to carry in, all Cursor's, all worth keeping:**
binding LMFE turns `Field(max_length=...)` into a DECODE CEILING rather than
post-hoc hygiene; `NewsCloseRead.news_close_read` is `min_length=1` with NO
`max_length` and `max_new_tokens=None`, so bind it only with eyes open (or do
dossier first); lm-format-enforcer 0.11.3 mishandles numeric JSON-Schema
`const` (DossierLLM does not use one -- recheck per schema); and pydantic
`@field_validator` rules are NOT in the JSON schema, so the post-hoc ladder
stays regardless.

## 6. THE UPSCALE IDEA (operator, same night) -- separate, and also real

*"See that a still only upscale once and hold the frame ... if the upscalers
were smart that this is the same frame I'm not gonna waste tokens on it, just
reuse the last upscaled frame."*

Correct, and cheaper than it sounds. The model loop
(`nodes/otr_silent_composite.py::_run_model_pipeline`) is a pure spatial map:
decode N frames -> per-batch `engine.upscale()` -> fit/pad -> encode. Real-ESRGAN
in eval mode is deterministic, so **identical input frame => bit-identical
output frame**. A content hash of each decoded frame, with reuse of the prior
output on a match, is pure memoization: it cannot change a single output byte,
only skip work.

Measured cost it would remove: a one-act `still_*` episode ran **18+ segments
at 3-4 minutes each (105+ minutes)** where the canonical `upscale_engine='off'`
default publishes in 10-20. On `still_flat` every frame in a beat is the SAME
held image, so a hash-and-reuse collapses hundreds of model calls per segment
into one.

**Two distinct optimisations, do not conflate them:**
* **(a) frame-identity memoization** -- helps `still_flat` and every gap/floor
  fill; bit-exact; low risk.
* **(b) upscale-then-transform** -- for `still_pan` / `still_motion` the frames
  genuinely differ (pan/zoom), so (a) does not help; the win there is to
  upscale the SOURCE STILL once and perform the pan at high resolution. That
  changes the visual pipeline order and is a real design change, not a cache.

Also worth knowing before touching this: a model-skipping fast path already
exists at `nodes/otr_silent_composite.py:881`
(`engine is None or engine.name == "off" or not sharpen`), but every measured
segment took the MODEL PATH -- i.e. still-lane segments are being treated as
sharpened real-clip footage. Worth understanding WHY before adding a cache in
front of it.

## 7. Why this is admissible

Live production artifact: sweep leg 1, real traceback, recorded in
`docs/2026-08-25-llm-image-upscale-sweep-receipt.json`. The upscale cost is
measured from the same night's leg 3 server log. Neither is promoted to the
Bug Bible yet -- that waits on a verified fix, per standing process.
