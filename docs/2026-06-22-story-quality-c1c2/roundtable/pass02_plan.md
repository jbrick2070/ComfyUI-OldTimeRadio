# Story-Quality R2 C1 + C2 -- fix plan (ARCH + CODING hardened, pass02)

> REVIEW FOCUS NEXT PASS (pass03): **WIRING.** The exact writer edit points
> (canon_header build @~2745, news_close_brief before the compose_announcer_outro
> call @~3275, the meta-absent derive guard), the sentinel anti-duplication, and
> the prompt-snapshot tests. Ground vs OTR_LedgerScriptWriter + _otr_line_composer.

## DECISION (pass01, unchanged): injection-only C1, deterministic, zero-ripple.
CUT the C1 reroll gate; CUT the outline nudge; CUT the LLM opt-in entirely (pure
deterministic only -- GPT); CUT the news proper-noun/number regex -- anchors come
SOLELY from the CURATED `key_terms` (Gemini: a capitalized-word regex over news is
brittle + injects garbage; key_terms are already LLM-distilled, clean).

## GROUNDED SEAMS (verified in OTR_LedgerScriptWriter.py)
- `key_terms_tuple = briefs.key_terms` (~2425) + `meta["news"]["key_terms"]`
  (~2968) -- the curated salient entities. SOURCE of both derivations.
- `canon_header` built ~2745 (render_episode_canon_header + .replace @2746),
  stamped `meta["canon_header"]` @3255, passed per line to `_build_user_prompt`.
- `news_close_brief` drives `compose_announcer_outro` post-loop (~3275).
- cast names = the locked cast rows (dicts with "name").

## INVARIANTS
meta free-form: `meta["specificity_anchors"]` = runtime list[str];
`meta["central_object"]` = str ("" = none). NO new Pydantic fields. Deterministic +
idempotent. Model-agnostic. NO LineRequest / compose_announcer_outro signature
change. NO workflow-JSON change. UTF-8 no BOM. Audio invariant = spine/no-JSON (the
script intentionally changes; the clean indextts2 byte-identical fixture is
unaffected by a meta/prompt-context add).

## PURE HELPERS -- new module `nodes/_otr_specificity.py` (stdlib only, testable)
`_cast_tokens(cast) -> set[str]`: flatten the cast roster to lowercased whole-word
tokens (handle dict `row.get("name")` + object `getattr`); include each full name
AND its individual tokens ("John Doe" -> {"john doe","john","doe"}) so a partial
cast name cannot leak into anchors. Never raises.

`derive_specificity_anchors(key_terms, cast) -> list[str]`:
- candidates = `key_terms` in original order (coerce each to str, trim, collapse
  whitespace).
- EXCLUDE: empty / < 2 chars; any candidate whose casefold == a cast token OR whose
  tokens intersect the cast token set.
- dedupe case-insensitively (keep first casing).
- cap = `MAX_ANCHORS = 5`.
- return list[str] (possibly empty). Pure; never raises.

`derive_central_object(key_terms, cast) -> str` (CONSERVATIVE, deterministic):
- candidates = `key_terms` in order. REJECT a candidate if ANY: it is ALL-CAPS
  (org acronym, e.g. NASA); it is a SINGLE capitalized word (bare proper-noun
  entity, e.g. Swift / Katalyst / Link); it matches a cast token; it contains an
  org/place suffix word in `_ENTITY_SUFFIX` = {agency, administration, university,
  institute, corp, corporation, inc, company, department, island, city, county,
  state, observatory, laboratory, lab, center, centre}; it is purely numeric.
- ACCEPT the FIRST surviving candidate (a descriptive object PHRASE with a lowercase
  common-noun head, e.g. "three robotic arms", "a $500M telescope"). Else return "".
- Pure; idempotent; never raises. (Imperfect by design -> OMIT when unsure, never
  invent -- the conservative resolution of the Gemini/DeepSeek "regex can't judge
  centrality" objection.)

`_sanitize_for_prompt(s) -> str`: collapse whitespace, strip newlines/control +
prompt delimiters, cap length (~80). Applied to every injected value.

## WIRING (idempotent, sentinel-guarded)
In the writer, AFTER canon_header is built and BEFORE the line loop:
- derive ONLY when absent: `if "specificity_anchors" not in meta:
  meta["specificity_anchors"] = derive_specificity_anchors(key_terms, cast)`.
- inject ONLY when truthy AND not already present (Gemini): `if meta.get(
  "specificity_anchors") and "Specificity anchors:" not in canon_header:` ->
  `canon_header += "\n\nSpecificity anchors (when natural, ground a line in one of
  these concrete details; do not force them into every line):\n- " +
  "\n- ".join(_sanitize_for_prompt(a) for a in meta["specificity_anchors"])`.
  (Sentinel + the once-before-loop placement => no duplication on resume/retry.)
For C2, BEFORE the compose_announcer_outro call:
- `if "central_object" not in meta: meta["central_object"] =
  derive_central_object(key_terms, cast)`.
- `if meta.get("central_object") and "Central object" not in news_close_brief:
  news_close_brief += " Central object, if useful: " +
  _sanitize_for_prompt(meta["central_object"]) + "."` (conditional => unchanged
  when "").

## TESTS
- _cast_tokens flattens names + tokens.
- anchors: key_terms in/out (cast names filtered incl. partial; dedupe; cap 5; empty).
- central_object: rejects NASA / single-cap entity / cast name / suffix / numeric;
  accepts the first descriptive phrase; returns "" when none; idempotent.
- idempotency: present key (incl. [] / "") is NOT recomputed; a second injection
  pass does NOT duplicate the block (sentinel).
- injection: canon_header carries the anchors block iff anchors truthy;
  news_close_brief carries the central object iff set; both unchanged when empty.
- sanitize: newlines/delimiters stripped, length capped.
