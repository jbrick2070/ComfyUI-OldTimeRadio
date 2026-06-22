# Story-Quality R2 C1 + C2 -- fix plan (ARCHITECTURE hardened, pass01)

> REVIEW FOCUS NEXT PASS (pass02): **CODING.** The exact extractor contracts
> (sources, normalize/dedupe, cast-name + boilerplate exclusion, cap, empty
> fallback, value shape), the central-object object-likeness heuristic, and
> idempotency. Ground vs `_otr_dramatic_state` / `_otr_casting` / the news brief.

## DECISION (panel-converged, judged)
- **C1 = INJECTION-ONLY. CUT the reroll gate** (all 4 panelists). The composer
  lacks beat position, so a "no anchor + no proper noun" gate would burn the single
  reroll on valid short lines ("Yes.", "Stop!") and is not model-agnostic. The
  positive INJECTION of concrete anchors is the lever; the existing S3/C5 gates
  already catch generic output.
- **DETERMINISTIC derivation (default), no new LLM call** (GPT + me + DeepSeek;
  Gemini dissented for centrality). Resolution: extract from the CURATED
  `key_terms` (already LLM-distilled salient entities in the news brief) plus a
  proper-noun/number regex over `meta["news"]` -- this gives the LLM quality Gemini
  wanted on the INPUT while the SELECTION stays deterministic + C7-safe. A cheap
  cached LLM extraction stays a documented OPT-IN behind a flag (default off).
- **ZERO-RIPPLE injection** (all 4): append to existing strings, NO signature/schema
  change. C1 -> append to `canon_header`; C2 -> append to `news_close_brief`.
- **CUT the act-1 outline nudge** (all 4): the derivation runs AFTER the outline,
  so the outline cannot reference it; the announcer close is the consumer.

## INVARIANTS
Ledger `{cast,lines,meta}` schema FIXED -> values ride FREE-FORM meta
(`meta["specificity_anchors"]` = JSON list of short strings;
`meta["central_object"]` = short plain string, "" = unavailable); NO new Pydantic
fields. Deterministic + idempotent (do NOT recompute if meta already carries the
key -- C7-safe on resume). Model-agnostic. NO LineRequest / compose_announcer_outro
signature change. NO workflow-JSON change. UTF-8 no BOM.
**Audio invariant clarified (GPT):** "byte-identical SPINE" = no audio-pipeline /
workflow-JSON change; the generated script intentionally changes with these prompt
edits (that is the craft lift). The clean indextts2 byte-identical fixture is
unaffected by a meta/prompt-context add.

## C1 -- specificity anchors
- `derive_specificity_anchors(news, key_terms, cast_names) -> list[str]` (pure):
  source priority key_terms -> proper nouns + numerals in `news`; normalize/dedupe
  case-insensitively; EXCLUDE any term matching a cast/speaker name + generic
  outlet/date boilerplate; cap 3-5; empty list when nothing concrete.
- Inject ONLY when non-empty: append a delimited, non-authoritative block to
  `canon_header` in the writer BEFORE the LineRequest is built (zero ripple).
  Wording bounded (GPT): "When natural, ground a line in one of these concrete
  anchors; do not force them into every line." Store `meta["specificity_anchors"]`.
- The proper-noun helper (if built) is SCAN-ONLY (diagnostics), never a live gate.

## C2 -- central story-object
- `derive_central_object(key_terms, news) -> str` (pure, CONSERVATIVE): prefer the
  most "object-like" `key_term` -- a concrete physical thing, NOT a person /
  organization / place; OMIT ("") when no confident candidate (never invent).
  Idempotent.
- Thread via `news_close_brief` append: "Central object, if useful: <object>" BEFORE
  the `compose_announcer_outro` call; the shipped S2 close intent already says "use
  the central object if set" -> it is picked up naturally. No new kwarg.

## VERIFY-AT-BUILD (panel-flagged seams)
1. The `canon_header` build site in the writer (mutable string before LineRequest;
   reaches `_build_user_prompt(req)` unchanged).
2. The `news_close_brief` value + the `compose_announcer_outro` call site in the
   writer (mutable string before the call).
3. `meta` is a free-form dict that round-trips unknown keys (not a strict Pydantic
   model).
4. The exact writer variables carrying `key_terms` / `news` / the cast names.

## TESTS
- Extractor edge cases: empty news; all-cast-names; only dates; only generic terms;
  mixed proper nouns/numbers; duplicate casing -> anchors list correct, cast names
  filtered, cap respected, empty fallback.
- central_object: returns "" when no physical object; returns the object when one
  is clearly present; idempotent.
- Injection: the built `canon_header` contains the anchors block when meta has
  anchors (and omits it when empty); the outro `news_close_brief` carries the
  central object text when set.
